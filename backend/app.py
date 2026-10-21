"""API сервиса и раздача интерфейса.

Запуск: uvicorn backend.app:app  (в Docker — docker compose up).
Интерфейс — frontend/template.html: без вшитых данных он сам загружает их из /api/bootstrap.
Контракт карточки — agent/contract/card.schema.json; OpenAPI — /docs.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import impact, profiles
from agent.chat import answer
from agent.contract import validate_card
from agent.llm import LLM, LLMUnavailable
from agent.offline import NOTICE as OFFLINE_NOTICE
from agent.pipeline import Pipeline, event_from_text
from agent.slices import SUGGESTED, data_answer

from .store import Store

ROOT = Path(__file__).resolve().parents[1]
MOCK = json.loads((ROOT / "frontend" / "mock_data.json").read_text(encoding="utf-8"))
DEMO_IDS = [c["id"] for c in MOCK["companies"]]
TODAY = date.fromisoformat(os.environ["APP_TODAY"]) if os.environ.get("APP_TODAY") else date.today()

CACHE = Path(os.environ.get("APP_CACHE_DIR", ROOT / "data" / "cache"))
llm = LLM(CACHE / "llm")
pipe = Pipeline(llm, today=TODAY)
store = Store(CACHE / "app.db")
_collect_lock = threading.Lock()
log = logging.getLogger("newsagent")

app = FastAPI(title="Новостная аналитика для Альфа-Бизнеса", version="0.1.0",
              description="Отбор и интерпретация событий под профиль компании. Карточка — agent/contract/card.schema.json.")

# Платные эндпоинты (вызывают модель) — не чаще RATE_LIMIT запросов в минуту с одного адреса:
# защита общего ключа с лимитом 500 ₽/день от случайных и намеренных повторов.
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
COSTLY = ("/api/analyze", "/chat", "/events/", "/api/news/refresh")
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.method == "POST" and any(k in request.url.path for k in COSTLY) and not request.url.path.endswith("/feedback"):
        ip = request.client.host if request.client else "?"
        q, now = _hits[ip], time.monotonic()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse({"detail": "слишком много запросов к модели, попробуйте через минуту"}, status_code=429)
        q.append(now)
    resp = await call_next(request)
    # базовые заголовки безопасности; inline-скрипты нужны прототипу, внешних ресурсов нет
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                               "style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
    return resp


def _seed() -> None:
    """Карточки из mock_data.json — примеры, подготовленные командой; помечаются как seed."""
    if store.has_seed():
        return
    for cid, cards in MOCK["insights"].items():
        for c in cards:
            store.put_card(cid, impact.attach(c, profiles.get(cid), TODAY.isoformat()), "seed")


_seed()


def _load_precomputed() -> None:
    """Разборы агента, посчитанные заранее (scripts/run_agent.py --save → data/cards/<компания>/<id>.json).
    Экономит бюджет: готовую карточку не нужно пересчитывать при каждом запуске."""
    for f in sorted((ROOT / "data" / "cards").glob("*/*.json")):
        cid = f.parent.name
        if profiles.get(cid) is None or store.card(cid, f.stem):
            continue
        try:
            card = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            log.warning("не читается %s", f)
            continue
        if validate_card(card, today=TODAY).ok:
            store.put_card(cid, card, "agent")


_load_precomputed()


COLLECT_EVERY_MIN = int(os.environ.get("COLLECT_EVERY_MIN", "60"))
# Разметка новостей моделью Gemma при сборе: ~0,003 ₽ за пачку из 20 новостей. 0 — только словари, без расхода
COLLECT_TRIAGE = os.environ.get("COLLECT_TRIAGE", "1") == "1"


def _collect_once() -> None:
    from agent.ingest.collect import collect

    if not _collect_lock.acquire(blocking=False):
        return
    try:
        collect(48, llm if COLLECT_TRIAGE else None)
        pipe.reload_news()
    except Exception:  # нет сети — работаем на сохранённой выборке data/news/events.sample.json
        log.warning("сбор новостей не удался, используется сохранённая выборка", exc_info=True)
    finally:
        _collect_lock.release()


def _collect_loop() -> None:
    """Сбор свежих новостей при старте и затем каждые COLLECT_EVERY_MIN минут (критерий 04).

    Уже собранные события не размечаются повторно: хранилище объединяет их по id."""
    while True:
        _collect_once()
        time.sleep(max(COLLECT_EVERY_MIN, 5) * 60)


if os.environ.get("COLLECT_ON_START", "0") == "1":
    threading.Thread(target=_collect_loop, daemon=True).start()


def _company(cid: str) -> dict:
    p = profiles.get(cid)
    if not p:
        raise HTTPException(404, "компания не найдена")
    mock = next((c for c in MOCK["companies"] if c["id"] == cid), None)
    v = profiles.view(p)
    if mock:
        v["display"] = mock["display"]  # подписи демо-компаний команда сформулировала вручную
    return v


def _status() -> dict:
    return {"llm_online": llm.online, "spent_today_rub": round(llm.spent_today(), 2), "budget_rub": llm.budget,
            "news_events": len(pipe.events), "bank_fragments": len(pipe.bank.items), "today": TODAY.isoformat()}


# ---------- чтение ----------
@app.get("/api/health")
def health():
    return {"ok": True} | _status()


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Проверка живости для Docker и облачных платформ."""
    return {"ok": True}


@app.get("/api/bootstrap")
def bootstrap():
    ids = list(dict.fromkeys(DEMO_IDS + store.companies_with_cards()))
    insights, origins = {}, {}
    for cid in ids:
        rows = store.cards(cid)
        insights[cid] = [c for c, _ in rows]
        origins[cid] = {c["id"]: o for c, o in rows}
    all_c = sorted(({"id": i, "display": profiles.short_industry(profiles.get(i)), "region": profiles.get(i)["region"],
                     "revenue_bn": round(profiles.get(i)["revenue_rub"] / 1e9, 1)} for i in profiles.all_ids()),
                   key=lambda x: -x["revenue_bn"])
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "roles": MOCK["roles"],
            "companies": [_company(i) for i in ids], "all_companies": all_c,
            "insights": insights, "origins": origins, "status": _status(), "chat_suggestions": SUGGESTED}


@app.get("/api/companies/{cid}")
def company(cid: str):
    rows = store.cards(cid)
    return {"company": _company(cid), "insights": [c for c, _ in rows], "origins": {c["id"]: o for c, o in rows}}


@app.get("/api/companies/{cid}/events")
def company_events(cid: str, limit: int = 10):
    """Отбор без LLM: события, прошедшие правила, с причинами (критерий 01)."""
    _company(cid)
    have = {c["id"] for c, _ in store.cards(cid)}
    return [{"id": e["id"], "title": e["title"], "published": e["published"], "source": e["source"], "url": e["url"],
             "topics": e["topics"], "also": len(e.get("also", [])), "score": m.score, "matched": m.matched,
             "reasons": m.reasons, "analyzed": f"{e['id']}-{cid}" in have}
            for e, m in pipe.ranked(cid, limit)]


@app.get("/api/metrics")
def metrics():
    """Метрики качества (критерий 12): обратная связь, доля карточек, прошедших проверку, расход."""
    return store.metrics() | {"status": _status()}


# ---------- действия ----------
def _rules(event: dict, cid: str, reason: str) -> dict:
    """Модель недоступна — черновой разбор правилами (agent/offline.py), явно помеченный в ответе и в карточке."""
    card, trace, m = pipe.rules(event, cid)
    if card is None:
        raise HTTPException(503, reason)
    if not store.card(cid, card["id"]):  # готовый разбор агентом не затираем черновиком
        store.put_card(cid, card, "rules")
    return {"ok": True, "card": card, "mode": "rules", "notice": OFFLINE_NOTICE, "reason": reason, "warnings": [], "tool_log": []}


def _run(event: dict, cid: str, on_step=None) -> dict:
    res, m = pipe.analyze(event, cid, on_step)
    if res.unavailable:  # нет ключа, бюджет исчерпан, сбой провайдера
        return _rules(event, cid, res.error)
    cost = sum(t.get("cost", 0) for t in res.trace)
    ms = sum(t.get("ms", 0) for t in res.trace)
    store.log_run(cid, event["id"], res.card is not None, res.report.errors, cost, ms, len(res.tool_log),
                  sum(1 for t in res.trace if t["title"] == "Правка черновика"))
    if res.card is None:
        return {"ok": False, "errors": res.report.errors, "trace": res.trace, "matched": m.matched}
    store.put_card(cid, res.card, "agent")
    return {"ok": True, "card": res.card, "warnings": res.report.warnings, "tool_log": res.tool_log}


SSE_PING_SEC = 15


def _stream(event: dict, cid: str) -> StreamingResponse:
    """Запуск агента с потоком шагов (Server-Sent Events): step — текущая трасса, result — итог, error — отказ.

    Агент работает в отдельном потоке и кладёт шаги в очередь; генератор отдаёт их клиенту по мере
    появления. Пока модель думает дольше SSE_PING_SEC, уходит комментарий-пинг, чтобы прокси не
    закрыли соединение."""
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            q.put({"type": "result", **_run(event, cid, on_step=lambda st: q.put({"type": "step", **st}))})
        except HTTPException as e:
            q.put({"type": "error", "detail": e.detail})
        except Exception:
            log.exception("агент упал на событии %s", event.get("id"))
            q.put({"type": "error", "detail": "внутренняя ошибка агента"})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            try:
                item = q.get(timeout=SSE_PING_SEC)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is None:
                return
            yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _event(cid: str, eid: str) -> dict:
    _company(cid)
    ev = next((e for e in pipe.events if e["id"] == eid), None)
    if not ev:
        raise HTTPException(404, "событие не найдено")
    return ev


@app.post("/api/companies/{cid}/events/{eid}/analyze")
def analyze_event(cid: str, eid: str):
    return _run(_event(cid, eid), cid)


@app.post("/api/companies/{cid}/events/{eid}/analyze/stream")
def analyze_event_stream(cid: str, eid: str):
    return _stream(_event(cid, eid), cid)


class NewsIn(BaseModel):
    company_id: str
    title: str = Field(default="", max_length=300)
    text: str = Field(default="", max_length=20000)
    url: str = Field(default="", max_length=500)


def _news_event(body: NewsIn) -> dict:
    """Новость пользователя: текст, или ссылка (статья, пост Telegram, PDF), или и то и другое."""
    title, text = body.title.strip(), body.text.strip()
    if body.url and len(text) < 200:
        from agent.ingest.article import fetch_article
        try:
            a = fetch_article(body.url)
        except ValueError as e:  # UnsafeURL — тоже ValueError: сообщение понятное, без деталей сервера
            raise HTTPException(422, f"Не удалось прочитать ссылку: {e}") from e
        title, text = title or a.title, (text + "\n\n" + a.text).strip()
        ev = event_from_text(title or text[:120], text, body.url, source=urlparse(body.url).netloc or "Ссылка",
                             published=a.published or None)
        return ev | {"text_kind": "full"}
    if len(title) < 5 and len(text) < 40:
        raise HTTPException(422, "Нужен заголовок, текст новости или ссылка на неё")
    return event_from_text(title or text[:120], text, body.url)


@app.post("/api/analyze")
def analyze_news(body: NewsIn):
    """Разбор новости, которой нет в базе (критерий 04: ранее не показанные новости)."""
    _company(body.company_id)
    return _run(_news_event(body), body.company_id)


@app.post("/api/analyze/stream")
def analyze_news_stream(body: NewsIn):
    """То же, что /api/analyze, но шаги агента приходят потоком SSE."""
    _company(body.company_id)
    return _stream(_news_event(body), body.company_id)


class ChatIn(BaseModel):
    company_id: str
    q: str = Field(min_length=2, max_length=500)


@app.post("/api/cards/{card_id}/chat")
def chat(card_id: str, body: ChatIn):
    card = store.card(body.company_id, card_id)
    if not card:
        raise HTTPException(404, "карточка не найдена")
    data = data_answer(body.company_id, body.q)  # «другой разрез» из профиля — без модели и мгновенно
    if data:
        return data
    try:
        res = answer(llm, pipe.bank, card, body.company_id, body.q, TODAY)
    except LLMUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(502, f"ответ модели не разобран: {e}") from e
    return {k: v for k, v in res.items() if not k.startswith("_")}


class FeedbackIn(BaseModel):
    company_id: str
    role: str = Field(pattern="^(cfo|ceo|treasurer)$")
    vote: str = Field(pattern="^(up|down)$")


@app.post("/api/cards/{card_id}/feedback")
def feedback(card_id: str, body: FeedbackIn):
    store.put_feedback(body.company_id, card_id, body.role, body.vote)
    return {"ok": True}


@app.post("/api/news/refresh")
def refresh_news(hours: int = 48):
    """Сбор свежих новостей из data/sources.json (занимает до минуты)."""
    from agent.ingest.collect import collect

    if not _collect_lock.acquire(blocking=False):
        raise HTTPException(409, "сбор уже идёт")
    try:
        stats = collect(min(max(hours, 1), 168), llm if COLLECT_TRIAGE else None)
    finally:
        _collect_lock.release()
    pipe.reload_news()
    return stats


# ---------- интерфейс ----------
@app.get("/", response_class=HTMLResponse)
def index():
    from frontend.page import render

    return render("null")
