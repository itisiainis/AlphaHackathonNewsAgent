"""«Данные в другом разрезе» (критерий 09): ответы на вопросы о структуре бизнеса клиента — без LLM.

Что делает: если вопрос в чате про покупателей, поставщиков, крупнейших контрагентов или состав
группы, ответ и график строятся прямо из профиля компании (платёжные данные банка). Это мгновенно,
бесплатно, работает без ключа модели и не может выдумать цифру.

Как устроено: намерение определяется по основам слов в вопросе (первое совпадение по порядку
INTENTS); для каждого намерения — функция, которая берёт нужный раздел профиля и возвращает
утверждения и точки графика. Если ни одно намерение не подошло — вопрос уходит модели (chat.py).
Доля потока с неизвестной отраслью показывается отдельным столбцом: неполные данные не прячем.
"""
from __future__ import annotations

import re

from . import profiles
from .okved import short_name

LABEL_MAX = 32  # подписи длинные — интерфейс рисует такие графики горизонтальными столбцами


def _short(label: str) -> str:
    return label if len(label) <= LABEL_MAX else label[:LABEL_MAX - 1].rstrip() + "…"


def _pct(x: float) -> float:
    return round(x * 100, 1)


def num(x: float, digits: int = 1) -> str:
    """Число по-русски: 45.8 → «45,8»."""
    return f"{x:.{digits}f}".replace(".", ",")


def _by_industry(p: dict, side: str, title: str) -> dict | None:
    s = p[side]
    if not s["n"]:
        return {"a": [{"t": f"В платёжной выборке банка нет данных о {'покупателях' if side == 'customers' else 'поставщиках'} компании.", "p": "ai"}]}
    rows = [b for b in s["by_industry"] if b["share"] >= 0.005][:4]
    points = [[_short(short_name(b["okved2"])), _pct(b["share"])] for b in rows]
    unknown = 1 - (s["known_industry_share"] or 0)
    if unknown > 0.005:
        points.append(["Отрасль неизвестна", _pct(unknown)])
    top = rows[0] if rows else None
    a = [{"t": f"{'Входящий' if side == 'customers' else 'Исходящий'} поток за период выборки — "
               f"{num(s['total_rub'] / 1e9)} млрд ₽ от {s['n']} контрагентов.", "p": "ai"}]
    if top:
        a.append({"t": f"Крупнейшая отрасль — {short_name(top['okved2']).lower()} (ОКВЭД {top['okved2']}): "
                       f"{num(_pct(top['share']))}% потока.", "p": "ai"})
    a.append({"t": f"Отрасль известна только для {num(_pct(s['known_industry_share'] or 0))}% потока: остальные "
                   "контрагенты вне базы банка, вывод по отраслям ограничен.", "p": "ai"})
    if s.get("intragroup_share"):
        a.append({"t": f"Внутри группы — {num(_pct(s['intragroup_share']))}% потока.", "p": "ai"})
    return {"a": a, "chart": {"title": title, "unit": "%", "points": points,
                              "note": "Платёжные данные банка (обезличенные), доля в сумме платежей"}}


def _top_counterparties(p: dict) -> dict:
    top = p["customers"]["top"][:5]
    if not top:
        return {"a": [{"t": "В платёжной выборке нет данных о покупателях компании.", "p": "ai"}]}
    points = [[f"{k}. " + (_short(short_name(t["okved"][:2])) if t["okved"] else "отрасль неизвестна"),
               round(t["amount_rub"] / 1e6)] for k, t in enumerate(top, 1)]
    total = p["customers"]["total_rub"] or 1
    share = sum(t["amount_rub"] for t in top) / total
    return {"a": [{"t": f"Пять крупнейших покупателей дают {num(_pct(share))}% входящего потока.", "p": "ai"},
                  {"t": "Названия контрагентов в данных кейса обезличены, поэтому показаны их отрасли.", "p": "ai"}],
            "chart": {"title": "Крупнейшие покупатели", "unit": "млн ₽", "points": points,
                      "note": "Платёжные данные банка (обезличенные), сумма платежей"}}


def _group(p: dict) -> dict:
    g = p["group"]
    if g["size"] <= 1:
        return {"a": [{"t": "Компания не входит в группу: в данных банка у неё нет связанных юрлиц.", "p": "ai"}]}
    points = [[_short(short_name(x["okved2"])), _pct(x["share"])] for x in g["industries"][:5]]
    return {"a": [{"t": f"В группе {g['size']} юрлиц; ниже — доли отраслей по числу юрлиц.", "p": "ai"}],
            "chart": {"title": "Состав группы по отраслям", "unit": "%", "points": points,
                      "note": "Данные банка о группе компаний (GROUP_ID)"}}


# порядок важен: «крупнейшие покупатели» должны сработать раньше, чем просто «покупатели»
INTENTS = [
    (r"крупнейш|топ|главн\w* (?:клиент|покупател|контрагент)", _top_counterparties),
    (r"поставщ|закуп|исходящ", lambda p: _by_industry(p, "suppliers", "Поставщики по отраслям")),
    (r"покупател|клиент|выручк|входящ|продаж", lambda p: _by_industry(p, "customers", "Покупатели по отраслям")),
    (r"групп|холдинг|юрлиц|дочерн", _group),
]

SUGGESTED = ["Покупатели по отраслям", "Поставщики по отраслям", "Крупнейшие покупатели"]


def data_answer(company_id: str, question: str) -> dict | None:
    """Ответ из профиля компании или None, если вопрос не про структуру бизнеса."""
    q = question.lower()
    p = profiles.get(company_id)
    if not p:
        return None
    for pattern, fn in INTENTS:
        if re.search(pattern, q):
            res = fn(p)
            return {"q": question, **res}
    return None
