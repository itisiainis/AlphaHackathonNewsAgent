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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import impact, profiles, roles
from agent.chat import FOLLOW_UPS, answer, card_answer
from agent.contract import validate_card
from agent.learning import Learner
from agent.llm import LLM, LLMUnavailable
from agent.loop import refresh_saved
from agent.offline import CHANNELS_STEP
from agent.offline import UNAVAILABLE as OFFLINE_NOTICE
from agent.okved import short_name
from agent.pipeline import Pipeline, event_from_text
from agent.sensitivity import diverse as diverse_events
from agent.sensitivity import factors as factors_of
from agent.sensitivity import hits as factor_hits_of
from agent.slices import SUGGESTED, data_answer

from .admin import admin as admin_guard
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
MOCK = json.loads((ROOT / "frontend" / "mock_data.json").read_text(encoding="utf-8"))
DEMO_IDS = [c["id"] for c in MOCK["companies"]]
TODAY = date.fromisoformat(os.environ["APP_TODAY"]) if os.environ.get("APP_TODAY") else date.today()

CACHE = Path(os.environ.get("APP_CACHE_DIR", ROOT / "data" / "cache"))
llm = LLM(CACHE / "llm")
pipe = Pipeline(llm, today=TODAY)
store = Store(CACHE / "app.db")
learner = Learner(store)  # опыт сервиса: память разборов, оценки клиентов, рутина в потоке, уроки проверки
pipe.learner = learner
learner.observe(pipe.events)
_collect_lock = threading.Lock()
log = logging.getLogger("newsagent")

app = FastAPI(title="Новостная аналитика для Альфа-Бизнеса", version="0.1.0",
              description="Отбор и интерпретация событий под профиль компании. Карточка — agent/contract/card.schema.json.")

# Платные эндпоинты (вызывают модель) — не чаще RATE_LIMIT запросов в минуту с одного адреса:
# защита общего ключа с лимитом 500 ₽/день от случайных и намеренных повторов.
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
# Любые POST без модели (краткий разбор, чтение ссылки, оценки, запросы менеджеру) — тоже с потолком: база и чтение
# чужих сайтов по ссылке не должны быть доступны для перебора
POST_LIMIT = int(os.environ.get("POST_LIMIT_PER_MIN", "60"))
COSTLY = ("/api/analyze", "/chat", "/events/")
FREE = ("/feedback", "/brief", "/request")  # без вызова модели: лимит платных запросов на них не распространяется
_hits: dict[str, deque] = defaultdict(deque)
_posts: dict[str, deque] = defaultdict(deque)
_last_sweep = [0.0]


def _over(hits: dict[str, deque], ip: str, limit: int, now: float) -> bool:
    q = hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


def _sweep(now: float) -> None:
    """Раз в минуту — забыть адреса без запросов за минуту: словари счётчиков не растут без предела."""
    if now - _last_sweep[0] < 60:
        return
    _last_sweep[0] = now
    for hits in (_hits, _posts):
        for ip in [ip for ip, q in hits.items() if not q or now - q[-1] > 60]:
            del hits[ip]


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if request.method in ("POST", "PUT", "DELETE"):
        ip, now = request.client.host if request.client else "?", time.monotonic()
        _sweep(now)
        if _over(_posts, ip, POST_LIMIT, now):
            return JSONResponse({"detail": "слишком много запросов, попробуйте через минуту"}, status_code=429)
        if any(k in path for k in COSTLY) and not path.endswith(FREE) and _over(_hits, ip, RATE_LIMIT, now):
            return JSONResponse({"detail": "слишком много запросов к модели, попробуйте через минуту"}, status_code=429)
    resp = await call_next(request)
    # заголовки безопасности; inline-скрипты нужны странице, внешних ресурсов нет; ссылки — только своего сайта
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                               "style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; "
                                               "base-uri 'none'; object-src 'none'; form-action 'self'")
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
    Экономит бюджет: готовую карточку не нужно пересчитывать при каждом запуске. Кеш (APP_CACHE_DIR) переживает
    перезапуск, поэтому проверка идёт и по уже сохранённой карточке: разбор агентом на сервере новее файла — берём его,
    файл новее (пересчитали) — берём файл."""
    for f in sorted((ROOT / "data" / "cards").glob("*/*.json")):
        cid = f.parent.name
        if profiles.get(cid) is None:
            continue
        meta = store.card_meta(cid, f.stem)
        stored = store.card(cid, f.stem) if meta and meta[0] == "agent" else None
        if stored and meta[1] >= f.stat().st_mtime:
            card = stored
        else:
            try:
                card = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                log.warning("не читается %s", f)
                continue
        rep = validate_card(card, today=TODAY, check_revised=True)
        if rep.ok:
            # по правилам нынешнего кода: итог трассы — из её же шагов (разборы до правки итога писали «Все проверки
            # пройдены» после замечаний), без кодов отраслей в выводах ИИ
            card = refresh_saved(card)
            if card != stored:
                store.put_card(cid, card, "agent")
        else:
            # разбор, посчитанный до пересмотра прогноза банка (или не прошедший проверку), не показываем:
            # неверная позиция банка хуже, чем краткий разбор того же события; пересчитать — scripts/run_agent.py
            SKIPPED_PRECOMPUTED[f"{cid}/{f.stem}"] = rep.errors[:3]
            log.warning("готовый разбор %s/%s не загружен: %s", cid, f.stem, "; ".join(rep.errors[:2]))
            if stored:
                store.drop_card(cid, f.stem)  # сохранён прошлым запуском, до проверки — убираем из ленты


SKIPPED_PRECOMPUTED: dict[str, list[str]] = {}  # в панели управления: что пересчитать
_load_precomputed()


def _refresh_rules() -> None:
    """Краткие разборы из кеша прошлого запуска, собранные до каналов влияния (вывод-ярлык вместо вывода), пересобираются
    по тому же событию: из потока новостей или из новостей, которые прислал клиент. Событие выпало из потока — разбор
    остаётся как был."""
    for cid, card, _, _ in store.all_cards("rules"):
        if any(st.get("title") == CHANNELS_STEP for st in card.get("trace", [])):
            continue
        eid = card["id"][: -len(cid) - 1] if card["id"].endswith("-" + cid) else card["id"]
        ev = next((e for e in pipe.events if e["id"] == eid), None) or store.user_event(cid, eid)
        if not ev or profiles.get(cid) is None:
            continue
        try:
            fresh, _, _ = pipe.rules(ev, cid, fetch=False)
        except Exception:  # пересборка одного разбора не должна мешать запуску
            log.exception("пересборка краткого разбора %s", card["id"])
            continue
        if fresh:
            store.put_card(cid, fresh, "rules")


_refresh_rules()


COLLECT_EVERY_MIN = int(os.environ.get("COLLECT_EVERY_MIN", "60"))
# Разметка новостей моделью Gemma при сборе: ~0,003 ₽ за пачку из 20 новостей. 0 — только словари, без расхода
COLLECT_TRIAGE = os.environ.get("COLLECT_TRIAGE", "1") == "1"
SOURCES_OFF: list[str] = []  # источники, отключённые в панели управления


LAST_COLLECT: dict = {}  # сводка последнего сбора — в панели управления


def _collect_once() -> None:
    from agent.ingest.collect import collect

    if not _collect_lock.acquire(blocking=False):
        return
    try:
        stats = collect(48, llm if COLLECT_TRIAGE else None, disabled=set(SOURCES_OFF))
        LAST_COLLECT.clear()
        LAST_COLLECT.update(stats | {"at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        pipe.reload_news()
        learner.observe(pipe.events)
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
    v["factors"] = factors_of(p)  # факторы чувствительности по данным банка (agent/sensitivity.py) — блок на главной
    return v


def _factor_hits(company: dict, cards: list[dict]) -> dict[str, list[str]]:
    """Какие факторы чувствительности затрагивает каждая карточка — для блока «Чувствительность бизнеса» и метки карточки."""
    return {c["id"]: h for c in cards if (h := factor_hits_of(c, company["factors"]))}


def _status() -> dict:
    return {"llm_online": llm.online, "spent_today_rub": round(llm.spent_today(), 2), "budget_rub": llm.budget,
            "news_events": len(pipe.events), "bank_fragments": len(pipe.bank.items), "today": TODAY.isoformat(),
            "agent_avg_ms": store.recent_run_ms()}


# ---------- чтение ----------
@app.get("/api/health")
def health():
    return {"ok": True} | _status()


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Проверка живости для Docker и облачных платформ."""
    return {"ok": True}


def _visible(cid: str) -> list[tuple[dict, str]]:
    """Карточки компании без тех, что сотрудник банка скрыл в панели управления."""
    hidden = {k for _, k in store.hidden(cid)}
    return [(c, o) for c, o in store.cards(cid) if c["id"] not in hidden]


@app.get("/api/bootstrap")
def bootstrap():
    ids = list(dict.fromkeys(DEMO_IDS + store.companies_with_cards()))
    insights, origins, user_cards = {}, {}, {}
    for cid in ids:
        _ensure_briefs(cid)
        rows = _visible(cid)
        insights[cid] = [c for c, _ in rows]
        origins[cid] = {c["id"]: o for c, o in rows}
        user_cards[cid] = store.user_card_ids(cid)  # прислал сам клиент: не в «Важном» на главной, отдельной группой
    all_c = sorted(({"id": i, "display": profiles.short_industry(profiles.get(i)), "region": profiles.get(i)["region"],
                     "sector": short_name(profiles.get(i)["okved2"]),
                     "revenue_bn": round(profiles.get(i)["revenue_rub"] / 1e9, 1)} for i in profiles.all_ids()),
                   key=lambda x: -x["revenue_bn"])
    companies = [_company(i) for i in ids]
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "roles": roles.public(), "zones": roles.zones(),
            "companies": companies, "all_companies": all_c,
            "factor_hits": {c["id"]: _factor_hits(c, insights[c["id"]]) for c in companies},
            "insights": insights, "origins": origins, "user_cards": user_cards, "status": _status(), "chat_suggestions": SUGGESTED,
            "follow_ups": FOLLOW_UPS, "learned": learner.topic_weights()}


BRIEFS = int(os.environ.get("BRIEFS_PER_COMPANY", "3"))


def _ensure_briefs(cid: str) -> None:
    """У каждой компании в ленте сразу есть разборы: для лучших по отбору событий — краткий разбор без модели
    (бесплатно, миллисекунды). Полный разбор ИИ — по кнопке. Свежие события важнее примеров команды."""
    have = {c["id"] for c, o in store.cards(cid) if o != "seed"}
    for ev, _ in diverse_events(pipe.ranked(cid, 12), BRIEFS):  # лучшие события по разным факторам чувствительности
        if f"{ev['id']}-{cid}" in have:
            continue
        try:
            card, _, _ = pipe.rules(ev, cid, fetch=False)
        except Exception:  # разбор одного события не должен ломать открытие компании
            log.exception("краткий разбор %s для %s", ev.get("id"), cid)
            continue
        if card:
            store.put_card(cid, card, "rules")
    _precompute(cid)


# Главное — без нажатия: для лучших событий компании полный разбор ИИ готовится заранее, в фоне, по очереди.
# Быстрый вывод клиент видит сразу; когда разбор ИИ готов, он заменяет быстрый вывод на том же месте.
PRECOMPUTE_TOP = int(os.environ.get("PRECOMPUTE_TOP", "3"))          # сколько событий компании углублять заранее
PRECOMPUTE_DAILY = int(os.environ.get("PRECOMPUTE_DAILY", "15"))     # не больше разборов в день — бюджет общего ключа
PRECOMPUTE_BUDGET_SHARE = 0.6  # заранее тратим не больше 60% дневного лимита: остальное — на разборы по кнопке
_pre_queue: queue.Queue = queue.Queue()
_pre_seen: set[str] = set()
_pre_day = {"day": "", "n": 0}
_pre_lock = threading.Lock()


def _precompute_worker() -> None:
    while True:
        ev, cid = _pre_queue.get()
        try:
            if {c["id"]: o for c, o in store.cards(cid)}.get(f"{ev['id']}-{cid}") != "rules":
                continue  # уже разобрано ИИ (по кнопке или из памяти) или скрыто
            if not llm.online or llm.spent_today() >= PRECOMPUTE_BUDGET_SHARE * llm.budget:
                continue
            _run(ev, cid)
        except Exception:
            log.exception("заранее: разбор %s для %s", ev.get("id"), cid)
        finally:
            _pre_queue.task_done()


def _precompute(cid: str) -> None:
    if PRECOMPUTE_TOP <= 0 or not llm.online:
        return
    origins = {c["id"]: o for c, o in store.cards(cid)}
    with _pre_lock:
        today = date.today().isoformat()
        if _pre_day["day"] != today:
            _pre_day.update(day=today, n=0)
        for ev, _ in diverse_events(pipe.ranked(cid, 12), PRECOMPUTE_TOP):
            key = f"{ev['id']}-{cid}"
            if origins.get(key) != "rules" or key in _pre_seen or _pre_day["n"] >= PRECOMPUTE_DAILY:
                continue
            _pre_seen.add(key)
            _pre_day["n"] += 1
            _pre_queue.put((ev, cid))


threading.Thread(target=_precompute_worker, daemon=True).start()


@app.get("/api/companies/{cid}")
def company(cid: str):
    _company(cid)
    _ensure_briefs(cid)
    rows = _visible(cid)
    comp = _company(cid)
    return {"company": comp, "insights": [c for c, _ in rows], "origins": {c["id"]: o for c, o in rows},
            "user_cards": store.user_card_ids(cid), "factor_hits": _factor_hits(comp, [c for c, _ in rows])}


def _role_weights(role: str) -> dict:
    return (roles.get(role) or {}).get("weights", {})


@app.get("/api/companies/{cid}/events")
def company_events(cid: str, limit: int = 10, role: str = ""):
    """Отбор без LLM: события, прошедшие правила, с причинами (критерий 01). role — порядок под роль:
    финансовому директору раньше ставка и долг, генеральному — спрос и отрасль."""
    _company(cid)
    have = {c["id"] for c, _ in store.cards(cid)}
    learned = learner.topic_weights()
    own, common = learned["companies"].get(cid, {}).get(role, {}), learned["roles"].get(role, {})
    # опыт самой компании (если она оценивала тему) заменяет общий опыт роли — он уже учтён в нём как априорный
    w = {t: v + own.get(t, common.get(t, 0)) for t, v in _role_weights(role).items()}
    ranked = pipe.ranked(cid, limit)
    if w:
        ranked = sorted(ranked, key=lambda x: -(x[1].score + sum(w.get(t, 0) for t in x[0]["topics"]) / 2))
    return [{"id": e["id"], "title": e["title"], "published": e["published"], "source": e["source"], "url": e["url"],
             "topics": e["topics"], "also": len(e.get("also", [])), "score": m.score, "matched": m.matched,
             "reasons": m.reasons, "analyzed": f"{e['id']}-{cid}" in have}
            for e, m in ranked]


MACRO_MIN = 3  # меньше отобранных событий — лента дополняется главным в экономике


@app.get("/api/companies/{cid}/macro")
def company_macro(cid: str):
    """«Значимых событий за сутки нет — главное в экономике» (docs/product.md, «Нет важных событий»): когда отбор
    дал меньше MACRO_MIN событий, — до трёх новостей о ставке, курсе и инфляции за последние дни, по важности.
    Без модели. needed — лента тонкая; items в формате /events (без признаков компании) плюс macro (тема) и why."""
    from agent.matching import macro_events

    _company(cid)
    ranked = pipe.ranked(cid, 1000)
    needed = len(ranked) < MACRO_MIN
    have = {c["id"] for c, _ in store.cards(cid)}
    items = macro_events(pipe.events, TODAY, exclude={e["id"] for e, _ in ranked}) if needed else []
    return {"needed": needed, "selected": len(ranked),
            "items": [{"id": e["id"], "title": e["title"], "published": e["published"], "source": e["source"],
                       "url": e["url"], "topics": e["topics"], "also": len(e.get("also", [])), "macro": kind,
                       "why": why, "analyzed": f"{e['id']}-{cid}" in have} for e, kind, why in items]}


@app.get("/api/metrics")
def metrics():
    """Метрики качества (критерий 12): обратная связь, доля карточек, прошедших проверку, расход."""
    from scripts.eval_matching import evaluate

    ev = evaluate()  # размеченный набор: правила отбора без модели, считается за миллисекунды
    return store.metrics() | {"status": _status(),
                              "matching_eval": {k: ev[k] for k in ("pairs", "precision", "recall", "f1")}}


# ---------- действия ----------
def _rules(event: dict, cid: str, reason: str) -> dict:
    """Модель недоступна — черновой разбор правилами (agent/offline.py), явно помеченный в ответе и в карточке."""
    card, trace, m = pipe.rules(event, cid)
    if card is None:
        raise HTTPException(503, reason)
    if not store.card(cid, card["id"]):  # готовый разбор агентом не затираем черновиком
        store.put_card(cid, card, "rules")
    return {"ok": True, "card": card, "mode": "rules", "notice": OFFLINE_NOTICE, "reason": reason, "warnings": [], "tool_log": []}


def _run(event: dict, cid: str, on_step=None, fresh: bool = False) -> dict:
    if not fresh:
        dup = learner.remembered(cid, event, TODAY)
        if dup:  # тот же сюжет из другого источника уже разобран — ответ из памяти, без модели и расхода
            store.log_reuse(cid, dup["id"])
            return {"ok": True, "card": dup, "reused": True, "warnings": [], "tool_log": []}
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


def _event_from_card(cid: str, eid: str) -> dict | None:
    """Событие выпало из потока новостей (поток пересобран после перезапуска), а краткий разбор по нему остался в кеше:
    восстанавливаем событие из карточки — заголовок, факты новости, ссылка на первоисточник. Полный текст по ссылке
    агент подтянет сам (pipeline.full_text)."""
    card = store.card(cid, f"{eid}-{cid}")
    if not card:
        return None
    news = next((s for s in card.get("sources", []) if s.get("kind") == "news"), {})
    facts = [s["t"] for s in card["blocks"].get("what", []) if s.get("p") == "fact"]
    return event_from_text(card["title"], " ".join(facts) or card["title"], news.get("url", ""),
                           source=news.get("name") or "Новость", published=card.get("published")) | {"id": eid}


def _event(cid: str, eid: str) -> dict:
    _company(cid)
    # событие из потока или новость, которую клиент прислал сам (её полный разбор запускают позже)
    ev = next((e for e in pipe.events if e["id"] == eid), None) or store.user_event(cid, eid) or _event_from_card(cid, eid)
    if not ev:
        raise HTTPException(404, "событие не найдено: возможно, оно выпало из потока новостей — пришлите ссылку ещё раз")
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


_ARTICLES: dict[str, object] = {}  # ссылка → прочитанная статья: краткий и полный разбор видят один и тот же текст
ARTICLE_CACHE_MAX = 200


def _news_event(body: NewsIn) -> dict:
    """Новость пользователя: текст, или ссылка (статья, пост Telegram, PDF), или и то и другое."""
    title, text = body.title.strip(), body.text.strip()
    if body.url and len(text) < 200:
        from agent.ingest.article import NotArticle, fetch_article
        a = _ARTICLES.get(body.url)
        if a is None:
            try:
                a = fetch_article(body.url)
            except NotArticle as e:  # раздел или главная сайта: просим ссылку на саму новость
                raise HTTPException(422, str(e)) from e
            except ValueError as e:  # UnsafeURL — тоже ValueError: сообщение понятное, без деталей сервера
                raise HTTPException(422, f"Не удалось прочитать ссылку: {e}") from e
            if len(_ARTICLES) >= ARTICLE_CACHE_MAX:
                _ARTICLES.pop(next(iter(_ARTICLES)))
            _ARTICLES[body.url] = a
        title, text = title or a.title, (text + "\n\n" + a.text).strip()
        ev = event_from_text(title or text[:120], text, body.url, source=urlparse(body.url).netloc or "Ссылка",
                             published=a.published or None)
        return ev | {"text_kind": "full"}
    if len(title) < 5 and len(text) < 40:
        raise HTTPException(422, "Нужен заголовок, текст новости или ссылка на неё")
    return event_from_text(title or text[:120], text, body.url)


def _brief(ev: dict, cid: str) -> dict:
    """Краткий разбор за секунды и без модели. Готовый разбор ИИ по тому же событию не затирается."""
    existing = next(((c, o) for c, o in store.cards(cid) if c["id"] == f"{ev['id']}-{cid}"), None)
    if existing and existing[1] == "agent":
        return {"card": existing[0], "origin": "agent", "llm_online": llm.online}
    dup = learner.remembered(cid, ev, TODAY)
    if dup:  # тот же сюжет уже разобран ИИ — сразу полный разбор из памяти
        store.log_reuse(cid, dup["id"])
        return {"card": dup, "origin": "agent", "llm_online": llm.online, "reused": True}
    card, _, _ = pipe.rules(ev, cid, fetch=False)
    if card is None:
        raise HTTPException(422, "Не удалось разобрать новость")
    store.put_card(cid, card, "rules")
    return {"card": card, "origin": "rules", "llm_online": llm.online}


def _user_event(body: NewsIn) -> dict:
    """Новость клиента запоминается: полный разбор ИИ по ней можно запустить и после перезагрузки страницы."""
    _company(body.company_id)
    ev = _news_event(body)
    store.put_user_event(body.company_id, ev)
    return ev


@app.post("/api/brief")
def brief_news(body: NewsIn):
    """Своя новость: краткий разбор сразу; полный разбор ИИ клиент запускает следом
    (/api/companies/{cid}/events/{event_id}/analyze/stream — event_id приходит в ответе)."""
    ev = _user_event(body)
    return _brief(ev, body.company_id) | {"event_id": ev["id"], "user": True}


@app.post("/api/companies/{cid}/events/{eid}/brief")
def brief_event(cid: str, eid: str):
    return _brief(_event(cid, eid), cid)


@app.post("/api/analyze")
def analyze_news(body: NewsIn):
    """Разбор новости, которой нет в базе (критерий 04: ранее не показанные новости)."""
    return _run(_user_event(body), body.company_id)


@app.post("/api/analyze/stream")
def analyze_news_stream(body: NewsIn):
    """То же, что /api/analyze, но шаги агента приходят потоком SSE."""
    return _stream(_user_event(body), body.company_id)


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
    except (LLMUnavailable, ValueError):
        # модель недоступна или ответила не по делу — отвечаем материалами самой карточки, с источниками
        return card_answer(card, body.q)
    return {k: v for k, v in res.items() if not k.startswith("_")}


class FeedbackIn(BaseModel):
    company_id: str
    role: str = Field(pattern="^(cfo|ceo|treasurer|accountant)$")
    vote: str = Field(pattern="^(up|down)$")


@app.post("/api/cards/{card_id}/feedback")
def feedback(card_id: str, body: FeedbackIn):
    # оценка меняет порядок ленты (agent/learning.py) — только по существующей карточке существующей компании
    _company(body.company_id)
    if not store.card(body.company_id, card_id):
        raise HTTPException(404, "карточка не найдена")
    store.put_feedback(body.company_id, card_id, body.role, body.vote)
    return {"ok": True}


class RequestIn(BaseModel):
    company_id: str
    role: str = Field(pattern="^(cfo|ceo|treasurer|accountant)$")
    kind: str = Field(pattern="^(manager|product)$")
    label: str = Field(max_length=120)
    comment: str = Field(default="", max_length=1000)


@app.post("/api/cards/{card_id}/request")
def manager_request(card_id: str, body: RequestIn):
    """Запрос менеджеру из карточки. Карточка прикладывается к запросу, поэтому она должна существовать."""
    _company(body.company_id)
    if not store.card(body.company_id, card_id):
        raise HTTPException(404, "карточка не найдена")
    store.put_request(body.company_id, card_id, body.role, body.kind, body.label, body.comment.strip())
    return {"ok": True}


@app.post("/api/news/refresh", dependencies=[Depends(admin_guard)])
def refresh_news(hours: int = 48):
    """Сбор свежих новостей из data/sources.json (занимает до минуты). Только для сотрудников банка: сбор ходит в сеть
    и размечает новости моделью — за счёт общего ключа."""
    from agent.ingest.collect import collect

    if not _collect_lock.acquire(blocking=False):
        raise HTTPException(409, "сбор уже идёт")
    try:
        stats = collect(min(max(hours, 1), 168), llm if COLLECT_TRIAGE else None, disabled=set(SOURCES_OFF))
    finally:
        _collect_lock.release()
    pipe.reload_news()
    return stats


# ---------- интерфейс ----------
@app.get("/", response_class=HTMLResponse)
def index():
    from frontend.page import render

    return render("null")


# ---------- панель управления банка (backend/admin.py) ----------
from .admin import apply_saved  # noqa: E402
from .admin import router as admin_router  # noqa: E402

app.include_router(admin_router)
apply_saved()
