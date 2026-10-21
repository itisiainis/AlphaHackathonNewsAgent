"""Сбор новостей и превращение потока постов в события.

Что делает: читает источники из data/sources.json (RSS и публичные Telegram-каналы через t.me/s),
склеивает дубли в события, размечает темы и отрасли, отсеивает шум и пишет data/news/events.json.

Как устроено:
1. Сбор — код Telegram-бота (feeds.py, webcollector.py): только публичные данные, без ключей.
2. Дедупликация: заголовок → множество основ слов (kb.tokens); посты с коэффициентом Жаккара
   |A∩B| / |A∪B| ≥ DUP_JACCARD считаются одним событием. Событие хранит все источники —
   это и объединение материалов (критерий 03), и сигнал важности (чем больше изданий, тем важнее).
3. Разметка без LLM: темы и отрасли по словарям okved.py. Посты без единой бизнес-темы
   и отрасли отбрасываются (политика, спорт, происшествия) — это дешёвый первый фильтр.
4. Если задан ключ модели — Gemma пачками по 20 уточняет важность, темы, отрасли и помечает
   рекламу (схема пачечной оценки — из бота). Без ключа важность считается эвристикой.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import okved
from ..kb import tokens
from ..llm import ROUTING, LLMUnavailable, extract_json
from .feeds import fetch_feed_posts
from .post import Post
from .webcollector import UA, fetch_recent_web

ROOT = Path(__file__).resolve().parents[2]
DUP_JACCARD = 0.5
KEEP_DAYS = 14
BATCH = 20
VOCAB = set(okved.TOPIC_STEMS)
# Признаки рекламы и промо без LLM: маркировка рекламы (erid), призывы зарегистрироваться,
# инвестидеи для частных инвесторов. Точнее рекламу отсеивает Gemma, если ключ задан.
AD_RE = re.compile(r"erid|реклама|промокод|розыгрыш|зарегистрир|займите место|подписывайтесь|"
                   r"идея может не сработать|скидк[аиу]", re.I)

TRIAGE_SYSTEM = """Ты — редактор деловых новостей для руководителей крупных компаний. Для КАЖДОЙ новости верни объект:
{"id": номер, "importance": 0-10, "is_ad": true|false, "topics": [...], "sectors": [...], "gist": "суть одним предложением"}
importance: 0-2 — реклама, розыгрыши, мелочи; 3-5 — рутина; 6-8 — значимо для бизнеса; 9-10 — особо важно.
topics — только из списка: ставка, долг, курс, ликвидность, отрасль, спрос, контрагенты.
sectors — двузначные коды классов ОКВЭД2 затронутых отраслей (например "24" металлургия, "41" строительство).
Ответ — строго JSON-объект {"items": [объекты по каждой новости]}."""


def _event_id(p: Post) -> str:
    return "ev-" + hashlib.sha1((p.link or p.text[:120]).encode(), usedforsecurity=False).hexdigest()[:10]


def _title(p: Post) -> str:
    """Заголовок — первая строка хотя бы из трёх слов; строки из эмодзи и хештегов пропускаем.

    В Telegram-каналах первая строка часто «🛢#нефть #иран», а смысл — во второй. Если брать её как
    заголовок, разные новости с одинаковыми хештегами склеились бы в одно событие."""
    for line in p.text.split("\n"):
        clean = re.sub(r"\s+", " ", re.sub(r"#\S+|@\S+", " ", line))
        clean = re.sub(r"^[^\wА-Яа-яЁё«\"]+", "", clean).strip()   # эмодзи и значки в начале строки
        if sum(1 for w in clean.split() if re.search(r"[A-Za-zА-Яа-яЁё]{2}", w)) >= 3:
            return clean[:200]
    return re.sub(r"\s+", " ", p.text).strip()[:200]


async def fetch_all(sources: dict, hours: int) -> tuple[list[tuple[Post, dict]], list[str]]:
    import httpx

    posts, failures = [], []
    async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True, timeout=20) as http:
        for src in sources.get("rss", []):
            try:
                posts += [(p, src) for p in await fetch_feed_posts(http, src["url"], src["name"], hours)]
            except Exception:
                failures.append(src["name"])
        for src in sources.get("telegram", []):
            try:
                posts += [(p, src) for p in await fetch_recent_web(http, src["channel"], hours)]
            except Exception:
                failures.append(src["name"])
    return posts, failures


def to_events(posts: list[tuple[Post, dict]]) -> list[dict]:
    """Посты → события: разметка, отсев шума, склейка дублей."""
    events: list[dict] = []
    sigs: list[set] = []
    for p, src in sorted(posts, key=lambda x: x[0].date):
        text = p.text
        topics = okved.topics_in(text, title=_title(p))
        sectors = okved.sectors_in(_title(p) + " " + text[:600])  # отрасль — по заголовку и началу текста
        if (not topics and not sectors) or AD_RE.search(text):
            continue
        sig = set(tokens(_title(p)))
        ref = {"name": src["name"], "url": p.link, "date": p.date.date().isoformat()}
        for ev, s in zip(events, sigs, strict=True):
            if sig and s and len(sig & s) / len(sig | s) >= DUP_JACCARD:
                ev["also"].append(ref)
                if len(text) > len(ev["text"]):
                    ev["text"] = text
                break
        else:
            events.append({"id": _event_id(p), "title": _title(p), "text": text,
                           "published": p.date.date().isoformat(), "source": src["name"],
                           "kind": src.get("kind", "news"), "url": p.link, "official": bool(src.get("official")),
                           "topics": topics, "sectors": sectors, "also": [], "importance": 0, "gist": ""})
            sigs.append(sig)
    for ev in events:
        # эвристика без LLM: чем больше изданий написали, тем важнее; регулятор и министерства — важнее
        ev["importance"] = min(10, 4 + len(ev["also"]) + (1 if {"ставка", "курс"} & set(ev["topics"]) else 0)
                               + (2 if ev.get("official") else 0))
    return events


def triage(llm, events: list[dict]) -> int:
    """Уточнение моделью Gemma. Возвращает число размеченных событий (0 — модель недоступна)."""
    done = 0
    for i in range(0, len(events), BATCH):
        batch = events[i:i + BATCH]
        user = "\n\n".join(f"[{j}] {e['source']}: {e['title']}\n{e['text'][:500]}" for j, e in enumerate(batch))
        try:
            c = llm.complete(ROUTING["triage"], [{"role": "system", "content": TRIAGE_SYSTEM},
                                                 {"role": "user", "content": user}], max_tokens=6000, json_mode=True)
            data = extract_json(c.text)
            items = data.get("items", []) if isinstance(data, dict) else data
        except LLMUnavailable:
            return done
        except ValueError:
            continue
        for it in items if isinstance(items, list) else []:
            j = it.get("id") if isinstance(it, dict) else None
            if not isinstance(j, int) or not 0 <= j < len(batch):
                continue
            e = batch[j]
            try:
                e["importance"] = 0 if it.get("is_ad") else max(0, min(10, int(it.get("importance", e["importance"]))))
            except (TypeError, ValueError):
                pass
            e["topics"] = sorted(set(e["topics"]) | {t for t in it.get("topics", []) if t in VOCAB})
            e["sectors"] = sorted(set(e["sectors"]) | {s for s in it.get("sectors", []) if s in okved.SHORT_NAMES})
            e["gist"] = str(it.get("gist", ""))[:300]
            done += 1
    return done


def merge_store(path: Path, fresh: list[dict], today: datetime) -> list[dict]:
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_id = {e["id"]: e for e in old}
    by_id.update({e["id"]: e for e in fresh})
    cutoff = (today - timedelta(days=KEEP_DAYS)).date().isoformat()
    events = sorted((e for e in by_id.values() if e["published"] >= cutoff), key=lambda e: e["published"], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")
    return events


def collect(hours: int = 48, llm=None) -> dict:
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    posts, failures = asyncio.run(fetch_all(sources, hours))
    events = to_events(posts)
    triaged = triage(llm, events) if llm is not None and getattr(llm, "online", False) else 0
    events = [e for e in events if e["importance"] >= 3]
    store = merge_store(ROOT / "data" / "news" / "events.json", events, datetime.now(timezone.utc))
    return {"posts": len(posts), "events": len(events), "triaged": triaged, "stored": len(store), "failures": failures}


def as_posts_debug(posts):  # для отладки из консоли
    return [asdict(p) for p, _ in posts]


if __name__ == "__main__":
    import sys

    from ..llm import LLM

    print(collect(int(sys.argv[1]) if len(sys.argv) > 1 else 48, LLM(ROOT / "data" / "cache" / "llm")))
