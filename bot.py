import asyncio
import datetime
import logging
import logging.config
import time
from pathlib import Path

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

from asr import ASR

asr = ASR()


def get_api_token():
    with open('../telegram_tokens/voice_message_transcription_bot', 'r') as file:
        token = file.read().strip()
    return token


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f'Got command /start')
    text = ('Это бот для расшифровки голосовых сообщений в текст. \n'
            'Просто перешлите сообщение сюда и бот переведёт его в текстовый вид. \n'
            'Работает только с русским языком.')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f'Got voice message')
    file = await context.bot.get_file(update.message.voice.file_id)
    folder = Path('voice') / f'{update.message.from_user.id}'
    path = folder / f'{update.message.id}.oga'
    folder.mkdir(parents=True, exist_ok=True)
    logger.info(f'Save audio file to {path}')
    await file.download_to_drive(custom_path=path)
    logger.info(f'Start transcribing')
    text = asr.transcribe(path)
    logger.info(f'Send transcription {text!r}')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_to_message_id=update.message.id)
    logger.info(f'Delete audio file {path}')
    path.unlink(missing_ok=True)


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': f'logs/log_{datetime.datetime.today():%Y-%m-%d}.txt',
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


def run():
    logger.info(f'Starting bot')
    application = ApplicationBuilder().token(get_api_token()).build()
    start_handler = CommandHandler('start', start)
    voice_handler = MessageHandler(filters.VOICE, voice)
    application.add_handler(start_handler)
    application.add_handler(voice_handler)
    logger.info(f'Bot is running')
    application.run_polling()


if __name__ == '__main__':
    run()
