"""Уточняющий вопрос по карточке (критерий 09): ответ с той же разметкой факт / банк / ИИ.

Как устроено: код сам подбирает контекст — карточку, профиль компании и фрагменты аналитики банка,
найденные BM25 по тексту вопроса, — и один раз вызывает модель (Qwen3.6). Ответ проходит те же
правила происхождения, что и карточка: bank/fact без ссылки на источник понижаются до ai.
Если вопрос про «другой разрез» данных (доли, динамика), модель может вернуть небольшой график.

Цифры ответа сверяет код, как у карточки: модели даётся «паспорт цифр» компании — что именно значит каждое число
(46% поступлений — не 46% выручки; 27% — крупнейший получатель платежей, а не покупатель). Утверждение с числом,
которого нет ни в карточке, ни в профиле, ни в найденной аналитике банка, или с «N% выручки», которых в данных нет,
из ответа убирается — и об этом пишется. Неподтверждённая цифра хуже, чем её отсутствие.
"""
from __future__ import annotations

import json
import re

from . import impact, profiles
from .contract.validator import numbers_in
from .kb import BM25Index
from .llm import LLM, ROUTING, extract_json

CHAT_SYSTEM = """Ты — аналитик Альфа-Банка. Отвечай на вопрос руководителя по карточке события коротко и по делу.
Верни ОДИН JSON: {"a": [{"t": "...", "p": "fact|bank|ai", "src": "S…"}],
 "chart": {"title": "...", "unit": "...", "points": [["подпись", число], ...], "note": "источник данных"} или null}
fact и bank — только с меткой источника из списка; всё остальное — ai. Цифры — только из источников или профиля.
Если данных для ответа нет — так и скажи. График — только если вопрос просит данные в другом разрезе.
«Подробнее» — глубже про причины и механизм влияния на эту компанию; «С другой стороны» — аргументы против главного
вывода карточки и что может пойти иначе; «Что делать сейчас» — 2–3 конкретных шага на ближайшие недели.
«Что если не действовать» — что произойдёт с деньгами и рисками компании, если ничего не менять, и к какому сроку.
Не повторяй дословно то, что уже есть в карточке: добавляй новое.
Цифры компании бери только из «Паспорта цифр» и строго в том смысле, что там написан: доли поступлений и платежей
через банк — это не доли выручки; крупнейший получатель платежей — это поставщик, а не покупатель."""

# Уточнения, которые интерфейс предлагает под каждой карточкой: глубина — по запросу руководителя
FOLLOW_UPS = ["Что делать сейчас?", "С другой стороны", "Что если не действовать?", "Подробнее"]


def card_answer(card: dict, question: str) -> dict:
    """Ответ без модели — из материалов самой карточки, с теми же метками происхождения и источниками.
    Используется, когда модель недоступна: человек всё равно получает ответ, а не отказ."""
    q = question.lower()
    b = card["blocks"]
    if any(w in q for w in ("сторон", "против", "иначе", "возможн", "плюс")):
        picked, lead = b["opportunities"] + b["risks"], "С другой стороны — возможности и риски из разбора:"
    elif any(w in q for w in ("дела", "делать", "шаг", "действ", "предприн")):
        picked, lead = b["actions"], "Что можно сделать:"
    elif "банк" in q:
        bank = b["bank"]
        picked = [{"t": bank["quote"], "p": "bank"}] if bank["quote"] else []
        lead = f"Позиция банка — {bank['doc']}" if bank["quote"] else bank["doc"]
    elif any(w in q for w in ("риск", "опасн", "угроз")):
        picked, lead = b["risks"], "Риски:"
    else:
        picked, lead = b["why"] + b["what"], "Подробнее — почему это важно и что произошло:"
    calc = [{"t": f"{c['title']}: {c['value']} ({c['formula']})", "p": "ai"} for c in b.get("calc", [])]
    if any(w in q for w in ("подробн", "почему", "сколько", "деньг")):
        picked = picked + calc
    note = {"t": "Ответ собран из материалов разбора (ИИ-аналитик сейчас не ответил).", "p": "ai"}
    body = picked[:5] or [{"t": "В разборе нет сведений по этому вопросу.", "p": "ai"}]
    return {"q": question, "a": [{"t": lead, "p": "ai"}, *body, note]}


def _pp(x: float | None) -> str:
    return profiles.pct(x)


def figures(p: dict) -> str:
    """Паспорт цифр компании: каждое число профиля с его смыслом — чтобы модель не выдала долю поступлений за долю выручки."""
    v = profiles.view(p)
    lines = [f"- выручка за год: {impact.num(p['revenue_rub'] / 1e9)} млрд ₽"]
    for side, flow, top in (("customers", "поступлений", "покупатель (плательщик)"), ("suppliers", "платежей", "поставщик (получатель)")):
        c = v["counterparties"][side]
        if not c.get("n"):
            continue
        lines += [f"- {_pp(t['share'])} {flow} через банк — {t.get('who') or t['name']}" for t in c.get("top_industries", [])]
        lines.append(f"- отрасль известна лишь для {_pp(c['known_share'])} {flow} (это покрытие данных, а не доля отрасли)")
        if "top1_share" in c:
            lines.append(f"- крупнейший {top}: {_pp(c['top1_share'])} {flow}; пять крупнейших: {_pp(c['top5_share'])}")
    return "Паспорт цифр компании (по платежам через Альфа-Банк; доля выручки по отраслям не известна):\n" + "\n".join(lines)


SMALL = 12  # «2–3 шага», «6 месяцев» — счёт и сроки, а не цифры компании: их не сверяем
REVENUE_SHARE = re.compile(r"(\d+(?:[.,]\d+)?)\s?%\s+(?:[а-яё]+\s+){0,2}выручк", re.I)


def unsupported(text: str, grounding: str, grounding_nums: set[str]) -> str:
    """Почему утверждение нельзя показывать: число не из данных или «N% выручки», которых в данных нет. Пусто — можно."""
    for m in REVENUE_SHARE.finditer(text):
        if not re.search(re.escape(m.group(1)) + r"\s?%\s+(?:[а-яё]+\s+){0,2}выручк", grounding, re.I):
            return f"«{m.group(0)}» — доли выручки в данных нет"
    for n in numbers_in(text) - grounding_nums:
        try:
            v = float(n)
        except ValueError:
            continue
        if v > SMALL and not (2020 <= v <= 2035):
            return f"число {n} не найдено в разборе, профиле и аналитике банка"
    return ""


def answer(llm: LLM, bank: BM25Index, card: dict, company_id: str, question: str, today) -> dict:
    hits = bank.search(question + " " + card["title"], top_k=3, today=today)
    labels, extra = {}, {}
    lines = []
    for i, s in enumerate(card["sources"]):
        labels[f"S{i + 1}"] = i
        lines.append(f"[S{i + 1}] {s['kind']} · {s['name']} · {s['date']}")
    for h in hits:
        lbl = f"S{len(labels) + 1}"
        labels[lbl] = None  # новый источник: в ответе ссылка станет подписью, а не индексом карточки
        extra[lbl] = f"Альфа-Банк: «{h.title[:70]}», {h.date}"
        lines.append(f"[{lbl}] bank · {h.title} · {h.date}{' · УСТАРЕЛО' if h.stale else ''}\n{h.text[:900]}")
    p = profiles.get(company_id)
    profile = profiles.view(p)
    card_text = json.dumps({k: card.get(k) for k in ('title', 'takeaway', 'key_points', 'blocks')}, ensure_ascii=False)
    user = (f"Вопрос: {question}\n\nКарточка: {card_text[:5000]}\n\n{figures(p)}\n\n"
            f"Профиль: {json.dumps(profile, ensure_ascii=False)}\n\nИсточники:\n" + "\n".join(lines))
    grounding = "\n".join([card_text, figures(p), impact.profile_text(p), json.dumps(profile, ensure_ascii=False),
                            *(h.text for h in hits)])
    grounding_nums = numbers_in(grounding)
    c = llm.complete(ROUTING["chat"], [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": user}],
                     max_tokens=6000, json_mode=True, reasoning=False)
    data = extract_json(c.text)
    out, dropped = [], []
    for s in data.get("a", []):
        if not isinstance(s, dict) or not s.get("t"):
            continue
        why = unsupported(str(s["t"]), grounding, grounding_nums)
        if why:
            dropped.append(why)
            continue
        st = {"t": str(s["t"]), "p": s.get("p") if s.get("p") in ("fact", "bank", "ai") else "ai"}
        lbl = s.get("src") or ""
        if labels.get(lbl) is not None:
            st["src"] = labels[lbl]
        elif lbl in extra:
            st["t"] += f" [{extra[lbl]}]"  # фрагмент аналитики, которого нет в карточке
        elif st["p"] != "ai":
            st["p"] = "ai"  # нет источника — это не факт
        out.append(st)
    chart = data.get("chart")
    ok_chart = isinstance(chart, dict) and isinstance(chart.get("points"), list) and all(
        isinstance(p, list) and len(p) == 2 and isinstance(p[1], (int, float)) for p in chart["points"])
    if not out:
        raise ValueError("модель не дала ответа по существу" if not dropped else "цифры ответа не подтвердились: " + dropped[0])
    if dropped:
        out.append({"t": f"Часть ответа убрана: цифры не подтвердились данными компании и разбора ({dropped[0]}).", "p": "ai"})
    return {"q": question, "a": out,
            **({"chart": chart} if ok_chart and chart["points"] else {}), "_cost": c.cost_rub, "_model": c.model,
            "_dropped": dropped}
