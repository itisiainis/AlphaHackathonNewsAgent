"""Модель поста (новости) и нормализация ссылок на каналы.

Перенесено из Telegram-бота «Новостной агент» (collector.py). Чтение через Telethon
(личный аккаунт + файл сессии) намеренно не перенесено: в сдаваемом решении это риск.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime

MAX_PER_CHANNEL = 150   # максимум постов с одного канала за раз
MIN_TEXT_LEN = 30       # посты короче — пропускаем (картинки без текста и т.п.)


@dataclass
class Post:
    channel: str          # username канала (без @)
    channel_title: str
    msg_id: int
    text: str
    date: datetime
    views: int = 0
    forwards: int = 0
    doc_name: str = ""    # имя PDF-вложения, если есть
    doc_text: str = ""    # извлечённый текст PDF (только в полном режиме)
    doc_summary: str = "" # сводка PDF, заполняет ИИ
    url: str = ""         # прямая ссылка (для статей с сайтов)
    topics: list = field(default_factory=list)   # заполняет ИИ
    importance: int = 0                          # заполняет ИИ
    gist: str = ""                               # заполняет ИИ
    geo: str = "ru"       # "ru" — касается России, "world" — международная (ИИ)
    category: str = ""    # Политика/Экономика/Спорт и т.п. (ИИ)

    @property
    def link(self) -> str:
        return self.url or f"https://t.me/{self.channel}/{self.msg_id}"


def normalize_ref(ref: str) -> str:
    """@name / t.me/name / https://t.me/s/name -> name"""
    ref = ref.strip()
    ref = re.sub(r"^https?://", "", ref)
    ref = re.sub(r"^t\.me/(s/)?", "", ref)
    ref = ref.lstrip("@").split("/")[0].split("?")[0].strip()
    return ref
