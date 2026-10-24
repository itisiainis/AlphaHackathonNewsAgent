"""Роли и зоны ответственности — из одного конфига data/roles.json.

Его используют: краткий разбор (порядок каналов влияния под роль, agent/channels.py), агент (кто читает вывод,
за что отвечает и какие действия ему доступны, agent/prompts.py), бэкенд и интерфейс (веса тем — порядок ленты,
зоны — сводка «Долг: 1 риск» и фильтр ленты). Новая роль — запись в конфиге, без правки кода.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data" / "roles.json"


@lru_cache(maxsize=1)
def config() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def roles() -> list[dict]:
    return config()["roles"]


def zones() -> list[dict]:
    return config()["zones"]


def ids() -> tuple[str, ...]:
    return tuple(r["id"] for r in roles())


def get(role_id: str) -> dict | None:
    return next((r for r in roles() if r["id"] == role_id), None)


def zones_of(topics: list[str]) -> list[str]:
    """Зоны ответственности карточки — по её темам; карточка может попасть в несколько зон."""
    return [z["id"] for z in zones() if set(z["topics"]) & set(topics or [])]


def public() -> list[dict]:
    """Роли для интерфейса: без порядка каналов (он нужен только коду краткого разбора)."""
    return [{k: v for k, v in r.items() if k != "channels"} for r in roles()]


def prompt_block() -> str:
    """За что отвечает каждая роль — для промпта агента. Коротко: длинный блок с действиями заставлял черновую модель
    рассуждать до исчерпания лимита токенов (замер 23.10)."""
    return "; ".join(f"{r['id']} — {r['duties']}" for r in roles())
