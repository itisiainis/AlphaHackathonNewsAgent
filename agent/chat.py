"""Уточняющий вопрос по карточке (критерий 09): ответ с той же разметкой факт / банк / ИИ.

Как устроено: код сам подбирает контекст — карточку, профиль компании и фрагменты аналитики банка,
найденные BM25 по тексту вопроса, — и один раз вызывает модель (Qwen3.6). Ответ проходит те же
правила происхождения, что и карточка: bank/fact без ссылки на источник понижаются до ai.
Если вопрос про «другой разрез» данных (доли, динамика), модель может вернуть небольшой график.
"""
from __future__ import annotations

import json

from . import profiles
from .kb import BM25Index
from .llm import LLM, ROUTING, extract_json

CHAT_SYSTEM = """Ты — аналитик Альфа-Банка. Отвечай на вопрос руководителя по карточке события коротко и по делу.
Верни ОДИН JSON: {"a": [{"t": "...", "p": "fact|bank|ai", "src": "S…"}],
 "chart": {"title": "...", "unit": "...", "points": [["подпись", число], ...], "note": "источник данных"} или null}
fact и bank — только с меткой источника из списка; всё остальное — ai. Цифры — только из источников или профиля.
Если данных для ответа нет — так и скажи. График — только если вопрос просит данные в другом разрезе."""


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
    profile = profiles.view(profiles.get(company_id))
    user = (f"Вопрос: {question}\n\nКарточка: {json.dumps({k: card[k] for k in ('title', 'takeaway', 'blocks')}, ensure_ascii=False)[:5000]}\n\n"
            f"Профиль: {json.dumps(profile, ensure_ascii=False)}\n\nИсточники:\n" + "\n".join(lines))
    c = llm.complete(ROUTING["chat"], [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": user}],
                     max_tokens=6000, json_mode=True)
    data = extract_json(c.text)
    out = []
    for s in data.get("a", []):
        if not isinstance(s, dict) or not s.get("t"):
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
    return {"q": question, "a": out or [{"t": "Не удалось сформулировать ответ по имеющимся данным.", "p": "ai"}],
            **({"chart": chart} if ok_chart and chart["points"] else {}), "_cost": c.cost_rub, "_model": c.model}
