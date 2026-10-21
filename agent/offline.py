"""Черновой разбор правилами, когда модель недоступна (нет ключа, исчерпан бюджет, сбой провайдера).

Зачем: сервис должен оставаться полезным и честным без LLM — например, когда проект запускают
без ключа (ключ команды публиковать нельзя) или общий дневной бюджет закончился.
Карточка собирается теми же инструментами, что и у агента, но без модели:
- что произошло — информативные предложения новости (факт, источник S1);
- почему важно — признаки компании, на которые сработал отбор (факт из профиля, источник «статистика»);
- позиция банка — фрагмент аналитики, только свежий и только если совпадает с темой события;
- сколько это в деньгах — расчёт кодом (agent/impact.py).
Выводов, рисков и действий «от ИИ» здесь нет: их без модели не пишем. Карточка помечается как
разбор правилами, и полный разбор агентом её заменит, когда модель станет доступна.
"""
from __future__ import annotations

import re
import time
from datetime import date

from . import profiles
from .contract import validate_card
from .impact import _sentences
from .kb import BM25Index
from .matching import Match
from .okved import plain, topics_in, who
from .tools import ToolSession

TOPICS = {"ставка", "долг", "курс", "ликвидность", "отрасль", "спрос", "контрагенты"}
NOTICE = "Краткий разбор: факты новости, данные компании, позиция банка и расчёт — без выводов ИИ."
UNAVAILABLE = "ИИ-аналитик сейчас недоступен. " + NOTICE


# частые слова, совпадение по которым ничего не говорит о теме
GENERIC = {"росси", "банка", "банко", "сохра", "рынка", "рынке", "компа", "сообщ", "заяви", "новос",
           "эконо", "прогн", "ожида", "данны", "октяб", "сентя", "недел", "месяц", "будет", "также", "после"}


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _by_kind(match: Match) -> dict[str, list[dict]]:
    """Причины отбора по видам признака (kind из agent/matching.py); понижающие пометки — не причина."""
    by: dict[str, list[dict]] = {}
    for r in match.reasons:
        if r["weight"] > 1 and r.get("kind"):
            by.setdefault(r["kind"], []).append(r)
    return by


# отрасль события — своя, отрасль товара или отрасль группы (role из agent/matching.py)
OWN_HEAD = {"own": "Событие в вашей отрасли ({}) — касается вас напрямую",
            "goods": "Событие в отрасли вашего товара ({}) — может изменить ваши продажи и цены",
            "group": "Событие в отрасли вашей группы ({}) — касается компаний группы"}
OWN_WHY = {"own": "Это ваша отрасль — {}", "goods": "Вы продаёте товар этой отрасли — {}",
           "group": "В этой отрасли работают компании вашей группы — {}"}


def _pct(r: dict) -> str:
    return f"{round(r['share'] * 100)}%"


def headline(match: Match, order: tuple[str, ...]) -> str:
    """Главный вывод без модели простой причинной фразой: что в бизнесе компании затрагивает событие и почему."""
    by = _by_kind(match)
    for k in order:
        if k not in by:
            continue
        r = by[k][0]
        if k == "debt":
            return f"Это меняет стоимость вашего долга: {' и '.join(x['feature'] for x in by[k])} зависят от ставки"
        if k == "deposits":
            return "Это меняет доход по вашим депозитам: он следует за ставкой"
        if k == "fx":
            return f"Это меняет стоимость ваших расчётов с зарубежными партнёрами — {r['feature']}"
        if k == "customers":
            return f"{_cap(who(r['sector']))} — ваши покупатели ({_pct(r)} поступлений): что меняется у них, скажется на ваших продажах"
        if k == "suppliers":
            return f"{_cap(who(r['sector']))} — ваши поставщики ({_pct(r)} платежей): что меняется у них, скажется на ваших закупках"
        if k == "own":
            return OWN_HEAD[r.get("role", "own")].format(plain(r["sector"]))
        if k == "demand":
            return "Это меняет спрос ваших покупателей"
    return f"Касается вас: {match.reasons[0]['feature']}" if match.reasons else "Прямой связи с профилем компании не найдено"


def _why(by: dict[str, list[dict]]) -> list[str]:
    """«Почему это важно вам» без модели: признаки компании из профиля, по одной фразе на вид признака."""
    out = []
    for k, rs in by.items():
        if k == "debt":
            out.append(f"У вас в банке {' и '.join(x['feature'] for x in rs)}: платежи по ним зависят от ставки")
        elif k == "customers":
            out.extend(f"{_pct(x)} ваших поступлений через банк — от покупателей: {who(x['sector'])}. "
                       "Что меняется у них, скажется на ваших продажах" for x in rs)
        elif k == "suppliers":
            out.extend(f"{_pct(x)} ваших платежей через банк — поставщикам: {who(x['sector'])}. "
                       "Что меняется у них, скажется на ваших закупках" for x in rs)
        elif k == "own":
            out.extend(OWN_WHY[x.get("role", "own")].format(plain(x["sector"])) for x in rs)
        else:
            out.extend(f"{_cap(x['feature'])}: {x['why']}" for x in rs)
    return out


def _take(c: dict) -> str:
    """Главный вывод из расчёта кодом: сначала деньги, потом почему."""
    title, value = c["title"], c["value"]
    if title.startswith("Выручка под влиянием") and " — " in title:
        return f"{_cap(title.split(' — ', 1)[1])} — ваши покупатели: от них {value.replace(' в год', '')} выручки в год"
    if title.startswith("Выручка от покупателей") and " — " in title:
        return f"{_cap(title.split(' — ', 1)[1])} — ваши покупатели: выручка от них изменится на {value}"
    if title.startswith("Выручка:"):
        return f"Ваша выручка может измениться на {value}"
    return f"{title}: {value}"


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яё]{5,}", text.lower())}  # 5 букв: «ставку» и «ставки» — одна основа


def _repeats(s: str, title: str) -> bool:
    """Предложение пересказывает заголовок: почти все его основы слов уже есть в заголовке."""
    st = _stems(s)
    return bool(st) and len(st & _stems(title)) / len(st) >= 0.7


def _facts(event: dict) -> list[str]:
    """2–3 информативных предложения новости: с числами или достаточно длинные, без повторов заголовка."""
    title = event.get("title", "").strip()
    out = []
    for s in _sentences(event.get("text", "")):
        if len(s) < 40 or s.strip(" .") == title.strip(" .") or _repeats(s, title) or len(s) > 400:
            continue
        if re.search(r"\d", s) or len(out) < 1:
            out.append(s)
        if len(out) == 3:
            break
    return out or [title]


def _bank_quote(text: str, event_stems: set[str]) -> str:
    """Два предложения фрагмента, сильнее всего совпадающие с темой события, в исходном порядке."""
    text = re.sub(r"(^|(?<=[.!?] ))[А-ЯA-Z] (?=[А-Я])", "", text)  # одиночная буква — остаток буквицы после OCR
    sents = [s for s in _sentences(text) if 30 <= len(s) <= 400]
    scored = sorted(range(len(sents)), key=lambda k: -len(_stems(sents[k]) & event_stems))[:2]
    return " ".join(sents[k] for k in sorted(scored))


def rules_card(event: dict, company_id: str, match: Match, bank: BM25Index, news: BM25Index,
               today: date, bank_meta: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Карточка контракта без LLM и трасса её сборки."""
    ses = ToolSession(company_id, event, bank, news, today, bank_meta or [])
    trace: list[dict] = []

    def step(kind, title, detail, t0):
        trace.append({"kind": kind, "title": title, "detail": detail, "ms": int((time.monotonic() - t0) * 1000), "cost": 0})

    t0 = time.monotonic()
    facts = _facts(event)
    step("plan", "План (правила)", "модель недоступна: факты новости → признаки компании → позиция банка → расчёт", t0)

    t0 = time.monotonic()
    prof = ses.profile_source()
    p = profiles.get(company_id)
    why = [{"t": t, "p": "fact", "src": prof.label} for t in _why(_by_kind(match))[:3]]
    step("tool", "get_company_profile()", f"признаков компании: {len(why)}", t0)

    t0 = time.monotonic()
    ev_stems = _stems(event.get("title", "") + " " + " ".join(facts))
    res, _ = ses.call("search_bank_analytics", {"query": event.get("title", "")[:200]})
    ev_topics = set(event.get("topics") or topics_in(event.get("title", "") + " " + event.get("text", ""))) - {"отрасль"}
    title_stems = _stems(event.get("title", "")) - GENERIC
    # фрагмент банка берём, только если он про ту же тему, что и событие, и совпадает с заголовком по смыслу
    frag = next((f for f in res.get("fragments", []) if not f["stale"] and not f["too_old_for_conclusions"]
                 and set(topics_in(f["doc"] + " " + f["text"])) & ev_topics
                 and len(title_stems & _stems(f["doc"] + " " + f["text"])) >= 2), None)
    bank_stmt, bank_block = None, None
    if frag:
        src = next(s for s in ses.sources if s.label == frag["label"])
        quote = _bank_quote(src.text, ev_stems)
        if quote:
            bank_stmt = {"t": quote, "p": "bank", "src": src.label}
            bank_block = {"quote": quote, "doc": src.doc or src.name, "date": src.date, "stale": src.stale}
    step("tool", f"search_bank_analytics(\"{event.get('title', '')[:60]}\")",
         "свежий фрагмент по теме события найден" if bank_block else "свежей позиции банка по теме события нет", t0)

    t0 = time.monotonic()
    calcs = ses.calculate_impact()
    step("tool", "calculate_impact()", "; ".join(f"{c['title']}: {c['value']}" for c in calcs)[:200] or
         "числового шока в новости нет — расчёта нет", t0)

    # метки S… → индексы sources[] в порядке появления в карточке
    cited: list[str] = ["S1"]

    def idx(label: str) -> int:
        if label not in cited:
            cited.append(label)
        return cited.index(label)

    def fix(s: dict) -> dict:
        return {**s, "src": idx(s["src"])}

    what = [fix({"t": f, "p": "fact", "src": "S1"}) for f in facts]
    if bank_stmt:
        bank_stmt = fix(bank_stmt)
    why = [fix(s) for s in why]
    calc = [c | {"inputs": [fix(i) for i in c["inputs"]]} for c in calcs]
    by_label = {s.label: s for s in ses.sources}
    sources = [{"name": by_label[lbl].name, "kind": by_label[lbl].kind, "date": by_label[lbl].date or today.isoformat(),
                "url": by_label[lbl].url} for lbl in cited]

    def pick(prefer: tuple[str, ...], order: tuple[str, ...]) -> str:
        c = next((c for c in calcs if c["title"].startswith(prefer)), calcs[0] if calcs else None)
        return _take(c) if c else headline(match, order)

    signs = {c["sign"] for c in calcs}
    card = {
        "id": f"{event['id']}-{company_id}",
        "type": "risk" if "risk" in signs else "opportunity" if "opportunity" in signs else "info",
        "impact": "high" if match.score >= 6 else "medium" if match.score >= 3.5 else "low",
        "topics": [t for t in event.get("topics", []) if t in TOPICS] or ["отрасль"],
        "is_new": True, "published": event.get("published") or today.isoformat(),
        "title": event.get("title", "")[:200],
        "takeaway": {"cfo": pick(("Стоимость долга", "Валютные", "Выручка"), ("debt", "fx", "customers", "suppliers", "own")),
                     "ceo": pick(("Выручка", "Закупки", "Стоимость долга"), ("customers", "own", "suppliers", "demand", "debt", "fx")),
                     "treasurer": pick(("Стоимость долга", "Валютные", "Доход по депозитам"), ("debt", "fx", "deposits", "customers"))},
        # короткий слой: почему вам, позиция банка, факт новости — заголовок события уже виден в карточке
        "key_points": (why[:1] + ([bank_stmt] if bank_stmt else []) + what[:1])[:3],
        "why_short": match.reasons[0]["why"] if match.reasons else "Событие из отобранных для компании новостей",
        "matched": match.matched, "sources": sources,
        "blocks": {"what": what, "why": why, "risks": [], "opportunities": [], "actions": [],
                   **({"calc": calc} if calc else {}),
                   "bank": bank_block or {"quote": "", "doc": "Свежей позиции банка по теме события не найдено",
                                          "date": today.isoformat(), "stale": False},
                   "basis": {"features": match.matched + [f"Продукты банка: {', '.join(p['bank_products']) or 'нет'}"],
                             "coverage": " ".join(x for x in [NOTICE, ses.industry["note"], "; ".join(profiles.data_quality(p))] if x)}},
        "trace": [], "chat": [],
    }
    if len(card["key_points"]) < 2:
        card["key_points"] = (what + why)[:2] if len(what + why) >= 2 else what * 2
    rep = validate_card(card, today=today, source_texts={i: by_label[lbl].text for i, lbl in enumerate(cited)})
    trace.append({"kind": "check", "title": "Самопроверка (код)", "detail": rep.summary(), "ms": 0, "cost": 0})
    trace.append({"kind": "done", "title": "Черновой разбор готов" if rep.ok else "Разбор не прошёл проверку",
                  "detail": NOTICE, "ms": 0, "cost": 0})
    card["trace"] = trace
    return (card if rep.ok else None), trace
