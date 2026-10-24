"""Качество карточки: «нельзя ли этот вывод скопировать другой компании».

Вывод для руководителя ценен, только если он про его компанию. Проверка без модели:
- приметы компании в главном выводе — числа из её профиля по данным банка (выручка, доли поступлений и платежей),
  кто её покупатели и поставщики, страны внешней торговли;
- вывод «копируется» другой компании, если ни одна из его примет не отличает эту компанию от той;
- позиция банка с датой и конкретное действие — две другие обязательные части хорошей карточки.
Используют самопроверка агента (agent/loop.py: шаблонный вывод уходит на правку) и scripts/eval_cards.py.
"""
from __future__ import annotations

import re

from . import impact, profiles
from .okved import country_name, who

FLOW_MIN = 0.03


def marks(p: dict) -> set[str]:
    """Приметы компании: выручка, доли поступлений и платежей по отраслям, кто её покупатели и поставщики, страны."""
    out = {"n:" + impact.num(p["revenue_rub"] / 1e9)}
    for side in ("customers", "suppliers"):
        for b in p[side]["by_industry"]:
            if b["share"] >= FLOW_MIN:
                out |= {f"p:{round(b['share'] * 100)}", "w:" + who(b["okved2"]).lower()}
    out |= {"w:" + country_name(c).lower() for c in p["foreign_trade"]["countries"]}
    return out


def found(text: str, p: dict) -> set[str]:
    """Какие приметы компании есть в тексте: «46%», «22,3 млрд», «автодилеры», «Китай»."""
    low = text.lower()

    def hit(m: str) -> bool:
        kind, v = m[:2], m[2:]
        if kind == "p:":
            return re.search(rf"(?<![\d,.]){v}\s?%", text) is not None
        if kind == "n:":
            return re.search(rf"(?<![\d,.]){re.escape(v)}(?![\d,])", text) is not None
        return v in low
    return {m for m in marks(p) if hit(m)}


def takeaway_text(card: dict) -> str:
    return " ".join(str(t) for t in (card.get("takeaway") or {}).values())


def copyable_to(card: dict, company_id: str, others: list[str]) -> list[str]:
    """Компании, которым главный вывод подошёл бы без правки: в нём нет примет, отличающих эту компанию от них."""
    own = found(takeaway_text(card), profiles.get(company_id))
    return [o for o in others if o != company_id and not (own - marks(profiles.get(o)))]


def report(card: dict, company_id: str, others: list[str]) -> dict:
    b = card["blocks"]["bank"]
    acts = [a for a in card["blocks"]["actions"] if a.get("p") in ("ai", "rule")]
    return {"company_numbers": sorted(found(takeaway_text(card), profiles.get(company_id))),
            # позиция банка с датой; устаревшая аналитика с честной пометкой тоже засчитывается — так требует кейс
            "bank_dated": bool(b.get("date") and (b.get("quote") or b.get("stale"))),
            "action": bool(acts), "copyable_to": copyable_to(card, company_id, others)}


def style_issue(card: dict, company_id: str) -> str:
    """Замечание для правки агентом: в главном выводе нет ни одной приметы компании."""
    if found(takeaway_text(card), profiles.get(company_id)):
        return ""
    return ("takeaway: вывод не опирается на данные именно этой компании — его можно скопировать любой другой. "
            "Добавь цифру из профиля (выручка, доля поступлений или платежей) или назови её покупателей, поставщиков, страны")
