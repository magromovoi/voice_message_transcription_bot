import re
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, pipeline


@dataclass
class Config:
    sample_rate = 16_000
    n_fft = 2048
    hop_length = 1024
    n_mels = 128
    max_duration_sec = 11
    max_n_tokens = 500
    # max_n_samples = max_duration_sec * sample_rate
    max_n_chunks = int(max_duration_sec * sample_rate / hop_length) + 1
    batch_size = 256


config = Config()


def logsumexp(*args):
    return torch.logsumexp(torch.tensor(args), dim=-1)


def add(*args):
    return torch.sum(torch.tensor(args), dim=-1)


class Hypo(NamedTuple):
    tokens: torch.Tensor
    logprob: torch.Tensor

    @property
    def prob(self):
        return self.logprob.exp()

    def __repr__(self):
        return (f'Hypo('
                f'tokens={self.tokens!r}, '
                # f'text={self.text!r}, '
                # f'logprob={self.logprob.float().item()!r}, '
                f'prob={float(self.prob.float().item()):.6f}'
                f')')


class Hypotheses(list[Hypo]):
    beam_width = 3

    @classmethod
    def from_dict(cls, hypos_dict, beam_width=beam_width):
        hypos = [Hypo(tokens=torch.tensor(tokens_tuple, dtype=torch.int64), logprob=logprob) for tokens_tuple, logprob
                 in hypos_dict.items()]
        hypos = sorted(hypos, key=lambda hypo: hypo.logprob, reverse=True)
        return cls(hypos[:beam_width])


class ASR:
    def __init__(self, checkpoint='jonatasgrosman/wav2vec2-large-xlsr-53-russian'):
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(checkpoint)
        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.spellchecker = pipeline("text2text-generation", model="bond005/ruT5-ASR")

    def transcribe(self, path):
        wave, sample_rate = torchaudio.load(path)
        wave = torchaudio.functional.resample(wave, orig_freq=sample_rate, new_freq=config.sample_rate)
        wave = wave[0]
        x = self.processor(wave, sampling_rate=config.sample_rate, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = self.model(x).logits[0]
        logprobs = torch.softmax(logits, dim=-1).log()
        hypos = self.decode(logprobs)
        tokens = hypos[0].tokens
        text = ''.join(self.tokenizer.convert_ids_to_tokens(tokens.tolist())).replace('|', ' ')
        text = self.spellchecker(text)[0]['generated_text']
        words = text.split()
        text = ' '.join([word for word in words
                         if 'хуй' not in word
                         and not word.startswith('еб')
                         and not word.startswith('бля')])
        if re.sub(r'\s', '', text) == '':
            text = '...'
        return text

    def decode(self, logprobs):
        hypos = Hypotheses.from_dict({(): torch.tensor(0)})
        for i_chunk in range(logprobs.shape[0]):
            token_logprobs = logprobs[i_chunk]
            new_hypos = {}
            for hypo in hypos:
                old_tokens = hypo.tokens
                old_logprob = hypo.logprob
                for new_token in torch.argsort(token_logprobs, descending=True):
                    last_old_token = old_tokens[-1] if len(old_tokens) > 0 else self.tokenizer.pad_token_id
                    new_token_logprob = token_logprobs[new_token]
                    new_logprob = add(old_logprob, new_token_logprob)
                    new_tokens = None
                    if last_old_token == self.tokenizer.pad_token_id and new_token == self.tokenizer.pad_token_id:
                        new_tokens = old_tokens
                    elif last_old_token == self.tokenizer.pad_token_id and new_token != self.tokenizer.pad_token_id:
                        new_tokens = torch.cat([old_tokens[:-1], torch.tensor([new_token])])
                    elif last_old_token != self.tokenizer.pad_token_id and new_token == self.tokenizer.pad_token_id:
                        new_tokens = torch.cat([old_tokens, torch.tensor([new_token])])
                    elif last_old_token != self.tokenizer.pad_token_id and new_token != self.tokenizer.pad_token_id:
                        if last_old_token == new_token:
                            new_tokens = old_tokens
                        else:
                            new_tokens = torch.cat([old_tokens, torch.tensor([new_token])])
                    new_tokens_tuple = tuple(new_tokens.tolist())
                    new_hypos[new_tokens_tuple] = logsumexp(new_hypos.get(new_tokens_tuple, -float('inf')), new_logprob)
            hypos = Hypotheses.from_dict(new_hypos)
        return hypos
