from __future__ import annotations

import re
import time
from pathlib import Path
from dataclasses import dataclass
from pprint import pprint
from typing import NamedTuple
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from IPython.display import display, Audio, clear_output

import editdistance
import eng_to_ipa
import librosa
import librosa.feature
from librosa.feature import melspectrogram

import torch
import torchaudio
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset, DataLoader

sns.set_theme(font_scale=1.2)
tqdm.pandas()


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
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'


def show_mel_spec(mel_spec):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        librosa.amplitude_to_db(mel_spec, ref=np.max),
        ax=ax,
        x_axis='time',
        y_axis='mel',
        sr=config.sample_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
    )
    fig.colorbar(img, format="%2.f dB")
    plt.show()


def play_audio(wave, sample_rate):
    display(Audio(wave, rate=sample_rate))


def logsumexp(*args):
    return torch.logsumexp(torch.tensor(args), dim=-1)


def add(*args):
    return torch.sum(torch.tensor(args), dim=-1)


class Tokenizer:
    def __init__(self):
        self.blank_token = '_'
        self.alphabet = self.blank_token + ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        self.alphabet_size = len(self.alphabet)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.blank_idx = self.char_to_idx[self.blank_token]
        self.pad_idx = self.blank_idx
        self.cmu_to_rus = {'aa': 'а', 'ae': 'а', 'ah': 'э', 'ao': 'о', 'aw': 'ау', 'ay': 'ай', 'b': 'б', 'ch': 'ч',
                           'd': 'д', 'dh': 'з', 'eh': 'э', 'er': 'эр', 'ey': 'эй', 'f': 'ф', 'g': 'г', 'hh': 'х',
                           'ih': 'и', 'iy': 'и', 'jh': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'ng': 'нг',
                           'ow': 'оу', 'oy': 'ой', 'p': 'п', 'r': 'р', 's': 'с', 'sh': 'ш', 't': 'т', 'th': 'с',
                           'uh': 'у', 'uw': 'у', 'v': 'в', 'w': 'у', 'y': 'й', 'z': 'з', 'zh': 'ж'}

    def transcribe(self, word: str) -> str:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        cmu = eng_to_ipa.get_cmu([word])[0][0]
        if cmu.startswith('__IGNORE__'):
            return ''
        cmu = re.sub(r'[^ a-z]', '', cmu)
        cmu = cmu.split()
        rus = [self.cmu_to_rus[s] for s in cmu]
        transcription = ''.join(rus)
        return transcription

    def normalize(self, text: str) -> str:
        text = text.lower()
        words = re.split(r'[^а-яёa-z]', text)
        words = [w if re.sub(r'[а-яё]', '', w) == '' else self.transcribe(w) for w in words]
        words = [w for w in words if len(w) > 0]
        text = ' '.join(words)
        return text

    def separate_with_blanks(self, seq: list[int]) -> list[int]:
        mod_seq = [self.blank_idx] + [None, self.blank_idx] * len(seq)
        mod_seq[1::2] = seq
        return mod_seq

    def tokenize(self, text: str) -> list[int]:
        text = self.normalize(text)
        idxs = [self.char_to_idx[char] for char in text]
        return idxs

    def tokenize_with_blanks(self, text: str) -> list[int]:
        return self.separate_with_blanks(self.tokenize(text))

    def detokenize(self, idxs: list[int]) -> str:
        return ''.join(self.convert_ids_to_tokens(idxs))

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return self.tokenize(''.join(tokens))

    def convert_ids_to_tokens(self, idxs: list[int]) -> list[str]:
        return [self.idx_to_char[idx] for idx in idxs if idx != self.pad_idx]


tokenizer = Tokenizer()

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                     n_fft=config.n_fft, hop_length=config.hop_length)


class AudioDataset(Dataset):
    def __init__(self, *, metadata: pd.DataFrame, text_col: str,
                 audio_base_path: Path, audio_filename_col: str):
        self.metadata = metadata
        self.text_col = text_col
        self.audio_base_path = audio_base_path
        self.audio_filename_col = audio_filename_col
        self.mel_transform = mel_transform

    def __len__(self):
        return len(self.metadata)

    def get_wave(self, idx):
        item = self.metadata.iloc[idx]
        path = self.audio_base_path / item[self.audio_filename_col]
        wave, sample_rate = torchaudio.load(path)
        wave = torchaudio.functional.resample(wave, orig_freq=sample_rate, new_freq=config.sample_rate)
        wave = wave[0]
        return wave

    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]

        text = item[self.text_col]
        tokens = torch.tensor(tokenizer.tokenize(text))
        tokens = tokens[: config.max_n_tokens]
        tokens_padded = torch.full(size=(config.max_n_tokens,), fill_value=tokenizer.pad_idx, dtype=torch.int64)
        tokens_padded[: len(tokens)] = tokens

        wave = self.get_wave(idx)
        mel_spec = self.mel_transform(wave)
        mel_spec = mel_spec[:, : config.max_n_chunks]
        mel_spec_padded = torch.zeros((config.n_mels, config.max_n_chunks))
        mel_spec_padded[:, : mel_spec.shape[1]] = mel_spec

        batch = {'mel_spec': mel_spec_padded, 'tokens': tokens_padded, 'text': text,
                 'target_len': torch.tensor([tokens.shape[0]]), 'input_len': torch.tensor([mel_spec.shape[1]])}

        return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


class CtcTransformer(nn.Module):
    def __init__(self, *, n_mels, n_classes):
        super().__init__()
        self.n_mels = n_mels
        self.n_classes = n_classes
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_mels, nhead=2, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=10)  # input: (batch_size, seq_len, emb_size)
        self.head = nn.Linear(self.n_mels, self.n_classes)
        self.loss = nn.CTCLoss(blank=tokenizer.blank_idx, reduction='mean',
                               zero_infinity=True)  # input: (seq_len, batch_size, n_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logprobs = self.predict(batch['mel_spec'])
        loss = self.loss(
            log_probs=logprobs.transpose(0, 1), targets=batch['tokens'],
            input_lengths=batch['input_len'][:, 0], target_lengths=batch['target_len'][:, 0],
        )
        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch_size, n_mels, n_chunks)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.head(x)
        logprobs = f.softmax(x, dim=-1).log()  # (batch_size, n_chunks, n_classes)
        return logprobs


class CtcLoss(nn.Module):
    def forward(self, tokens, logprobs):
        len_tokens = len(tokens)
        n_chunks = logprobs.shape[0]
        alphas = torch.full(size=(n_chunks, len_tokens), fill_value=-float('inf'), dtype=torch.float).to(device)
        # alphas = torch.autograd.Variable(alphas.data, requires_grad=True)
        alphas[0, 0] = logprobs[0, tokens[0]]
        alphas[0, 1] = logprobs[0, tokens[1]]
        for t in range(1, n_chunks):
            for s in range(len_tokens):
                if s < len_tokens - 2 * (n_chunks - t):
                    pass
                elif s == 0:
                    alphas[t, s] = add(alphas[t - 1, s], logprobs[t, tokens[s]])
                elif s == 1:
                    alphas[t, s] = add(logsumexp(alphas[t - 1, s], alphas[t - 1, s - 1]), logprobs[t, tokens[s]])
                else:
                    if tokens[s] == tokenizer.blank_idx or tokens[s] == tokens[s - 2]:
                        alphas[t, s] = add(logsumexp(alphas[t - 1, s], alphas[t - 1, s - 1]), logprobs[t, tokens[s]])
                    else:
                        alphas[t, s] = add(logsumexp(alphas[t - 1, s], alphas[t - 1, s - 1], alphas[t - 1, s - 2]),
                                           logprobs[t, tokens[s]])
        loss = -logsumexp(alphas[-1, -1], alphas[-1, -2])
        loss = torch.autograd.Variable(loss.data, requires_grad=True)
        return loss


def word_error_rate(true: str, pred: str):
    true = true.split()
    pred = pred.split()
    return editdistance.eval(true, pred) / len(true)


def character_error_rate(true: str, pred: str):
    return editdistance.eval(true, pred) / len(true)


class Hypo(NamedTuple):
    tokens: torch.Tensor
    logprob: torch.Tensor

    @property
    def text(self):
        return ''.join(tokenizer.convert_ids_to_tokens(self.tokens.tolist())).replace('|', ' ')

    @property
    def prob(self):
        return self.logprob.exp()

    def __repr__(self):
        return (f'Hypo('
                # f'tokens={self.tokens!r}, '
                f'text={self.text!r}, '
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

