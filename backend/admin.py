"""Панель управления для банка: как работает сервис, как его воспринимают клиенты, что можно поменять.

Доступ — по токену сотрудника (заголовок X-Admin-Token), который задаётся переменной ADMIN_TOKEN. Без неё
панель закрыта. Токен идёт в заголовке, а не в cookie, поэтому чужой сайт не отправит запрос от имени
сотрудника (CSRF); неверные попытки ограничены и пишутся в журнал сервера без самого токена.
Каждое изменение записывается в журнал действий (audit) с именем сотрудника из X-Admin-Name.

Настройки меняются на лету и сохраняются в базе: после перезапуска применяются снова. Пределы у
настроек жёсткие — лимит расхода не выше доли сервиса в общем ключе, модели — только из белого списка.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from agent import learning, matching, stories
from agent.llm import ALLOWED, ROUTING

log = logging.getLogger("newsagent")
FAIL_LIMIT = 10  # неверных токенов в минуту с одного адреса — дальше 429: подбор токена не пройдёт
_fails: dict[str, deque] = defaultdict(deque)


def admin(request: Request, x_admin_token: str = Header(default=""), x_admin_name: str = Header(default="")) -> str:
    """Единая проверка доступа ко всем маршрутам панели. Без ADMIN_TOKEN панель закрыта (запрет по умолчанию)."""
    token = os.environ.get("ADMIN_TOKEN", "").strip()
    if not token:
        raise HTTPException(503, "панель управления выключена: задайте ADMIN_TOKEN в .env")
    ip = request.client.host if request.client else "?"
    now = time.monotonic()
    if len(_fails) > 10_000:  # адреса с давними ошибками забываем: словарь не растёт без предела
        for k in [k for k, v in _fails.items() if not v or now - v[-1] > 60]:
            del _fails[k]
    q = _fails[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= FAIL_LIMIT:
        raise HTTPException(429, "слишком много попыток входа, попробуйте через минуту")
    if not x_admin_token or not secrets.compare_digest(x_admin_token.encode(), token.encode()):
        q.append(now)
        log.warning("панель управления: отказ в доступе, адрес %s, путь %s", ip, request.url.path)  # сам токен не пишем
        raise HTTPException(401, "нужен токен сотрудника банка")
    return (unquote(x_admin_name).strip() or "сотрудник банка")[:40]  # имя в заголовке — в URL-кодировке (кириллица)


router = APIRouter(prefix="/api/admin", tags=["панель управления"], dependencies=[Depends(admin)])


def _app():
    import backend.app as a  # ленивый импорт: app подключает этот модуль в конце своей загрузки

    return a


# ---------- настройки ----------
BUDGET_MAX = 500.0  # дневной лимит общего ключа команды: больше нет смысла — дальше прокси организаторов сам отключит ключ


def _sources() -> list[str]:
    a = _app()
    s = json.loads((a.ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    return sorted({x["name"] for kind in ("rss", "telegram") for x in s.get(kind, [])})


def _current() -> dict:
    a = _app()
    return {"budget_rub": a.llm.budget, "rate_limit": a.RATE_LIMIT, "select_min": matching.SELECT_MIN,
            "briefs": a.BRIEFS, "collect_every_min": a.COLLECT_EVERY_MIN, "collect_triage": a.COLLECT_TRIAGE,
            "learning": a.learner.enabled, "routing": dict(ROUTING), "sources_off": list(a.SOURCES_OFF)}


def _num(v, lo, hi, cast):
    try:
        x = cast(v)
    except (TypeError, ValueError):
        raise ValueError(f"нужно число от {lo} до {hi}") from None
    if not lo <= x <= hi:
        raise ValueError(f"допустимо от {lo} до {hi}")
    return x


def _routing(v) -> dict:
    if not isinstance(v, dict) or set(v) - set(ROUTING):
        raise ValueError("задачи: " + ", ".join(ROUTING))
    bad = [m for m in v.values() if m not in ALLOWED]
    if bad:
        raise ValueError("модели только из списка кейса: " + ", ".join(ALLOWED))
    return v


def _sources_off(v) -> list[str]:
    known = set(_sources())
    if not isinstance(v, list) or any(x not in known for x in v):
        raise ValueError("неизвестный источник")
    return sorted(set(v))


def _bool(v) -> bool:
    if not isinstance(v, bool):
        raise ValueError("нужно да или нет")
    return v


SETTINGS = {  # ключ → (проверка, применение)
    "budget_rub": (lambda v: _num(v, 0, BUDGET_MAX, float), lambda a, v: setattr(a.llm, "budget", v)),
    "rate_limit": (lambda v: _num(v, 1, 60, int), lambda a, v: setattr(a, "RATE_LIMIT", v)),
    "select_min": (lambda v: _num(v, 1, 6, float), lambda a, v: setattr(matching, "SELECT_MIN", v)),
    "briefs": (lambda v: _num(v, 0, 6, int), lambda a, v: setattr(a, "BRIEFS", v)),
    "collect_every_min": (lambda v: _num(v, 5, 1440, int), lambda a, v: setattr(a, "COLLECT_EVERY_MIN", v)),
    "collect_triage": (_bool, lambda a, v: setattr(a, "COLLECT_TRIAGE", v)),
    "learning": (_bool, lambda a, v: setattr(a.learner, "enabled", v)),
    "routing": (_routing, lambda a, v: ROUTING.update(v)),
    "sources_off": (_sources_off, lambda a, v: a.SOURCES_OFF.__setitem__(slice(None), v)),
}


def apply_saved() -> None:
    """При запуске: настройки, сохранённые сотрудниками, поверх значений из окружения."""
    a = _app()
    for k, v in a.store.settings().items():
        if k not in SETTINGS:
            continue
        check, put = SETTINGS[k]
        try:
            put(a, check(v))
        except ValueError:
            log.warning("сохранённая настройка %s=%r не прошла проверку и не применена", k, v)


@router.get("/settings")
def get_settings():
    return {"values": _current(), "sources": _sources(), "models": {k: v[0] for k, v in ALLOWED.items()},
            "tasks": list(ROUTING), "limits": {"budget_rub": BUDGET_MAX}}


@router.put("/settings")
def put_settings(body: dict, who: str = Depends(admin)):
    a = _app()
    unknown = set(body) - set(SETTINGS)
    if unknown:
        raise HTTPException(422, "неизвестные настройки: " + ", ".join(sorted(unknown)))
    checked = {}
    for k, v in body.items():
        try:
            checked[k] = SETTINGS[k][0](v)
        except ValueError as e:
            raise HTTPException(422, f"{k}: {e}") from None
    before = _current()
    for k, v in checked.items():  # всё проверено — применяем разом
        SETTINGS[k][1](a, v)
        a.store.set_setting(k, v)
        if before[k] != v:
            a.store.audit(who, "настройка", f"{k}: {json.dumps(before[k], ensure_ascii=False)} → {json.dumps(v, ensure_ascii=False)}")
    return {"values": _current()}


# ---------- обзор ----------
def _attention(m: dict, st: dict, fresh_h: float | None, new_req: int, skipped: dict | None = None) -> list[dict]:
    """Что требует внимания сотрудника — правила простые и видны в коде."""
    out = []
    f, r = m["feedback"], m["agent_runs"]
    if f["useful_share"] is not None and f["useful"] + f["not_relevant"] >= 5 and f["useful_share"] < 0.6:
        out.append({"level": "high", "text": f"Клиенты отмечают «не про нас» чаще обычного: полезных {round(f['useful_share'] * 100)}%. "
                    "Посмотрите оценки на вкладке «Клиенты» и порог отбора в настройках."})
    if r["pass_rate"] is not None and r["total"] >= 3 and r["pass_rate"] < 0.8:
        out.append({"level": "high", "text": f"Проверку проходят {round(r['pass_rate'] * 100)}% разборов ИИ — частые замечания на вкладке «Обучение»."})
    if st["budget_rub"] and st["spent_today_rub"] >= 0.8 * st["budget_rub"]:
        out.append({"level": "medium", "text": f"Израсходовано {st['spent_today_rub']} из {st['budget_rub']} ₽ дневного лимита: "
                    "при исчерпании клиенты получат краткий разбор без ИИ."})
    if not st["llm_online"]:
        out.append({"level": "medium", "text": "Модель не подключена — клиенты видят краткий разбор без ИИ."})
    if fresh_h is None or fresh_h > 24:
        out.append({"level": "medium", "text": "Свежих новостей больше суток нет — проверьте сбор и источники."})
    # необработанные запросы менеджеру — счётчиком на вкладке «Клиенты», а не здесь
    if skipped:
        names = "; ".join(_company_name(k.split("/")[0]) for k in sorted(skipped)[:3]) + (" и др." if len(skipped) > 3 else "")
        out.append({"level": "medium", "text": f"Разборов ИИ, посчитанных заранее, спрятано от клиентов: {len(skipped)} ({names}). "
                    "С тех пор Альфа-Банк изменил прогноз, на который они опирались (или разбор не прошёл проверку), — "
                    "устаревший вывод показывать нельзя, поэтому клиенты видят по этим событиям краткий разбор. "
                    "Вернуть: «Пересчитать ИИ» на вкладке «Разборы»."})
    return out


@router.get("/overview")
def overview():
    a = _app()
    m = a.metrics()
    st = a._status()
    last = max((e["published"] for e in a.pipe.events), default=None)
    fresh_h = None
    if last:
        try:
            dt = datetime.fromisoformat(last)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            fresh_h = round((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 1)
        except ValueError:
            pass
    new_req = a.store.rows("SELECT COUNT(*) AS n FROM requests WHERE COALESCE(status,'new')='new'")[0]["n"]
    clients = a.store.rows("""SELECT company_id, SUM(vote='up') AS up, SUM(vote='down') AS down, MAX(ts) AS ts
                              FROM feedback GROUP BY company_id""")
    return {"metrics": m, "status": st, "news": {"last_published": last, "hours_ago": fresh_h, "events": len(a.pipe.events)},
            "collect": a.LAST_COLLECT, "requests_new": new_req, "hidden": len(a.store.hidden()),
            "reused": a.store.rows("SELECT COUNT(*) AS n FROM reuse")[0]["n"], "clients_with_votes": len(clients),
            "attention": _attention(m, st, fresh_h, new_req, a.SKIPPED_PRECOMPUTED),
            "skipped_precomputed": a.SKIPPED_PRECOMPUTED}


# ---------- клиенты ----------
def _titles() -> dict:
    return {(c, card["id"]): card.get("title", card["id"]) for c, card, _, _ in _app().store.all_cards()}


def _company_name(cid: str) -> str:
    from agent import profiles

    p = profiles.get(cid)
    return profiles.short_industry(p, limit=200) if p else cid  # в панели — полное название, таблица переносит строки


@router.get("/feedback")
def feedback_list(limit: int = 200):
    a, titles = _app(), _titles()
    rows = a.store.rows("SELECT company_id, card_id, role, vote, ts FROM feedback ORDER BY ts DESC LIMIT ?", (min(limit, 1000),))
    return [r | {"title": titles.get((r["company_id"], r["card_id"]), r["card_id"]), "company": _company_name(r["company_id"])} for r in rows]


@router.get("/requests")
def requests_list(limit: int = 200):
    a, titles = _app(), _titles()
    rows = a.store.rows("""SELECT rowid AS id, ts, company_id, card_id, role, kind, label, comment, COALESCE(status,'new') AS status
                           FROM requests ORDER BY ts DESC LIMIT ?""", (min(limit, 1000),))
    return [r | {"title": titles.get((r["company_id"], r["card_id"]), r["card_id"]), "company": _company_name(r["company_id"])} for r in rows]


class StatusIn(BaseModel):
    status: str = Field(pattern="^(new|in_work|done)$")


@router.post("/requests/{rowid}")
def request_status(rowid: int, body: StatusIn, who: str = Depends(admin)):
    a = _app()
    if not a.store.rows("SELECT 1 AS x FROM requests WHERE rowid=?", (rowid,)):
        raise HTTPException(404, "запрос не найден")
    a.store.set_request_status(rowid, body.status)
    a.store.audit(who, "запрос менеджеру", f"№{rowid} → {body.status}")
    return {"ok": True}


# ---------- разборы ----------
@router.get("/runs")
def runs(limit: int = 100):
    rows = _app().store.rows("SELECT ts, company_id, event_id, ok, errors, cost, ms, tools, revisions FROM runs ORDER BY ts DESC LIMIT ?",
                             (min(limit, 500),))
    return [r | {"errors": json.loads(r["errors"] or "[]"), "company": _company_name(r["company_id"])} for r in rows]


@router.get("/cards")
def cards(company: str = ""):
    a = _app()
    hidden = a.store.hidden()
    votes: dict = {}
    for f in a.store.rows("SELECT company_id, card_id, vote FROM feedback"):
        v = votes.setdefault((f["company_id"], f["card_id"]), {"up": 0, "down": 0})
        v[f["vote"]] += 1
    out, in_feed = [], {e["id"] for e in a.pipe.events}
    for cid, card, origin, created in a.store.all_cards():
        if company and cid != company:
            continue
        out.append({"company_id": cid, "company": _company_name(cid), "id": card["id"], "title": card.get("title"),
                    "type": card.get("type"), "impact": card.get("impact"), "origin": origin, "created": created,
                    "published": card.get("published"), "hidden": (cid, card["id"]) in hidden,
                    "votes": votes.get((cid, card["id"]), {"up": 0, "down": 0}), "rerun": _RERUNS.get((cid, card["id"])),
                    "rerunnable": card["id"].removesuffix("-" + cid) in in_feed})  # пересчитать можно только событие из потока
    return out


@router.get("/cards/{cid}/{card_id}")
def card_detail(cid: str, card_id: str):
    card = _app().store.card(cid, card_id)
    if not card:
        raise HTTPException(404, "карточка не найдена")
    return card


class HideIn(BaseModel):
    hidden: bool
    note: str = Field(default="", max_length=300)


@router.post("/cards/{cid}/{card_id}/hide")
def hide(cid: str, card_id: str, body: HideIn, who: str = Depends(admin)):
    a = _app()
    if not a.store.card(cid, card_id):
        raise HTTPException(404, "карточка не найдена")
    a.store.moderate(cid, card_id, body.hidden, body.note.strip())
    a.store.audit(who, "скрыт разбор" if body.hidden else "разбор возвращён", f"{cid}/{card_id}" + (f": {body.note.strip()}" if body.note.strip() else ""))
    return {"ok": True}


_RERUNS: dict[tuple[str, str], str] = {}  # (компания, карточка) → running | done | failed


@router.post("/cards/{cid}/{card_id}/rerun")
def rerun(cid: str, card_id: str, who: str = Depends(admin)):
    """Новый разбор ИИ по тому же событию — минуя память разборов. Расход идёт из общего дневного лимита."""
    a = _app()
    ev = next((e for e in a.pipe.events if f"{e['id']}-{cid}" == card_id), None)
    if not ev:
        raise HTTPException(409, "события нет в текущем потоке новостей — пересчитать можно только разбор из ленты")
    if _RERUNS.get((cid, card_id)) == "running":
        raise HTTPException(409, "разбор уже идёт")
    _RERUNS[(cid, card_id)] = "running"
    a.store.audit(who, "пересчёт разбора", f"{cid}/{card_id}")

    def work():
        try:
            res = a._run(ev, cid, fresh=True)
            _RERUNS[(cid, card_id)] = "done" if res.get("ok") and res.get("mode") != "rules" else "failed"
        except Exception:
            log.exception("пересчёт %s/%s", cid, card_id)
            _RERUNS[(cid, card_id)] = "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "status": "running"}


# ---------- сбор новостей ----------
@router.post("/collect")
def collect_now(who: str = Depends(admin)):
    a = _app()
    if a._collect_lock.locked():
        raise HTTPException(409, "сбор уже идёт")
    a.store.audit(who, "сбор новостей", "вручную")
    threading.Thread(target=a._collect_once, daemon=True).start()
    return {"ok": True}


# ---------- обучение ----------
@router.get("/learning")
def learning_summary():
    a = _app()
    s = a.learner.summary()
    roles = {r["id"]: r["name"] for r in a.MOCK["roles"]}
    s["weights_readable"] = [{"scope": roles.get(role, role), "topic": t, "delta": d}
                             for role, ts in s["weights"]["roles"].items() for t, d in ts.items() if d]
    s["constants"] = {"routine_days": stories.TEMPLATE_DAYS, "routine_factor": learning.ROUTINE_FACTOR,
                      "dup_jaccard": stories.SAME_STORY, "dup_days": learning.DUP_DAYS, "min_lesson": learning.MIN_LESSON}
    s["today"] = a.TODAY.isoformat()
    return s


# ---------- журнал ----------
@router.get("/audit")
def audit(limit: int = 200):
    return _app().store.rows("SELECT ts, actor, action, detail FROM audit ORDER BY ts DESC LIMIT ?", (min(limit, 1000),))
