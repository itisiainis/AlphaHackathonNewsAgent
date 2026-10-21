"""Сквозной конвейер для одной компании: события → отбор правилами → разбор агентом → карточки.

Бюджет: агент запускается только на TOP_N лучших пар «событие × компания», прошедших порог
отбора, и только если карточки ещё нет в хранилище. Повторный показ ленты — бесплатный.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import okved, profiles
from .kb import BM25Index, load_bank_index, load_bank_meta
from .llm import LLM
from .loop import Agent, Result
from .matching import Match, match, rank

ROOT = Path(__file__).resolve().parents[1]
TOP_N = 3
FULL_TEXT_MIN = 700  # короче — это анонс, перед разбором подтягиваем полный текст по ссылке


def load_events(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "data" / "news" / "events.json"
    if not path.exists():
        path = ROOT / "data" / "news" / "events.sample.json"
    from .ingest.collect import AD_RE

    events = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    # фильтр рекламы применяется и к уже сохранённым событиям: правка правил действует сразу
    return [retag(e) for e in events if not AD_RE.search(e.get("text", ""))]


def retag(e: dict) -> dict:
    """Темы и отрасли пересчитываются словарями при загрузке: правка словаря сразу действует на всю базу.
    Разметка модели (llm_topics, llm_sectors), если была при сборе, добавляется к словарной."""
    title, text = e.get("title", ""), e.get("text", "")
    return e | {"topics": sorted(set(okved.topics_in(text, title=title)) | set(e.get("llm_topics", []))),
                "sectors": sorted(set(okved.sectors_in(title + " " + text[:600])) | set(e.get("llm_sectors", [])))}


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
        self.bank_meta = load_bank_meta(ROOT / "data" / "bank" / "docs.json")
        self.reload_news()

    def reload_news(self):
        self.events = load_events()
        self.news = news_index(self.events)

    def ranked(self, company_id: str, limit: int = 20) -> list[tuple[dict, Match]]:
        p = profiles.get(company_id)
        return rank(self.events, profiles.features(p), p["bank_products"])[:limit]

    def full_text(self, event: dict) -> dict:
        """Дополняет короткую новость (анонс RSS) полным текстом статьи по ссылке; результат кешируется.

        Не получилось (сайт недоступен, текст собирается скриптом) — остаётся анонс, и это видно
        в основании карточки: «текст новости — анонс источника»."""
        if event.get("text_kind") or len(event.get("text", "")) >= FULL_TEXT_MIN or not event.get("url", "").startswith("http"):
            return event
        cache = ROOT / "data" / "cache" / "articles" / f"{event['id']}.json"
        if cache.exists():
            return event | json.loads(cache.read_text(encoding="utf-8"))
        try:
            from .ingest.article import fetch_article
            a = fetch_article(event["url"])
            extra = {"text": a.text, "text_kind": "full"} if len(a.text) > len(event.get("text", "")) else {"text_kind": "summary"}
        except (ValueError, OSError):
            extra = {"text_kind": "summary"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")
        return event | extra

    def analyze(self, event: dict, company_id: str, on_step=None) -> tuple[Result, Match]:
        """on_step(state) вызывается на каждом шаге агента — для показа трассы в реальном времени."""
        event = self.full_text(event)
        p = profiles.get(company_id)
        m = match(event, profiles.features(p), p["bank_products"])
        agent = Agent(self.llm, self.bank, self.news, today=self.today, bank_meta=self.bank_meta)
        agent.on_step = on_step
        return agent.run(event, company_id, m), m

    def rules(self, event: dict, company_id: str, fetch: bool = True) -> tuple[dict | None, list[dict], Match]:
        """Краткий разбор без модели (agent/offline.py). fetch=False — без загрузки полного текста (быстро, для ленты)."""
        from .offline import rules_card

        if fetch:
            event = self.full_text(event)
        p = profiles.get(company_id)
        m = match(event, profiles.features(p), p["bank_products"])
        card, trace = rules_card(event, company_id, m, self.bank, self.news, self.today, self.bank_meta)
        return card, trace, m
