"""Сквозной конвейер для одной компании: события → отбор правилами → разбор агентом → карточки.

Бюджет: агент запускается только на TOP_N лучших пар «событие × компания», прошедших порог
отбора, и только если карточки ещё нет в хранилище. Повторный показ ленты — бесплатный.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import okved, profiles
from .kb import BM25Index, load_bank_index
from .llm import LLM
from .loop import Agent, Result
from .matching import Match, match, rank

ROOT = Path(__file__).resolve().parents[1]
TOP_N = 3


def load_events(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "data" / "news" / "events.json"
    if not path.exists():
        path = ROOT / "data" / "news" / "events.sample.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def news_index(events: list[dict]) -> BM25Index:
    return BM25Index([{"doc_id": e["id"], "title": e["title"], "date": e["published"], "page": None,
                       "text": e.get("gist") or e["text"][:1500], "url": e["url"], "kind": e.get("kind", "news")}
                      for e in events])


def event_from_text(title: str, text: str, url: str = "", source: str = "Новость пользователя",
                    published: str | None = None) -> dict:
    """Произвольная новость (вставленная в интерфейсе) → событие того же формата, что и собранные."""
    import hashlib

    full = f"{title}\n{text}"
    return {"id": "ev-" + hashlib.sha1(full.encode(), usedforsecurity=False).hexdigest()[:10], "title": title.strip()[:200],
            "text": text.strip(), "published": published or date.today().isoformat(), "source": source,
            "kind": "news", "url": url, "topics": okved.topics_in(text, title=title),
            "sectors": okved.sectors_in(title + " " + text[:600]),
            "also": [], "importance": 5, "gist": ""}


class Pipeline:
    def __init__(self, llm: LLM, today: date | None = None):
        self.llm = llm
        self.today = today or date.today()
        self.bank = load_bank_index(ROOT / "data" / "bank" / "docs.json")
        self.reload_news()

    def reload_news(self):
        self.events = load_events()
        self.news = news_index(self.events)

    def ranked(self, company_id: str, limit: int = 20) -> list[tuple[dict, Match]]:
        p = profiles.get(company_id)
        return rank(self.events, profiles.features(p), p["bank_products"])[:limit]

    def analyze(self, event: dict, company_id: str) -> tuple[Result, Match]:
        p = profiles.get(company_id)
        m = match(event, profiles.features(p), p["bank_products"])
        agent = Agent(self.llm, self.bank, self.news, today=self.today)
        return agent.run(event, company_id, m), m
