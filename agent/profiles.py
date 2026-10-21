"""Профили компаний: загрузка, вид для интерфейса, признаки для отбора и пометки о качестве данных.

Источник — data/company_profiles.json (115 компаний с выручкой от 15 млрд ₽, см. docs/data.md).
Здесь нет LLM: всё, что можно посчитать по данным, считается кодом.

Пометки о качестве данных (ограничение кейса 06 — неполные и зашумлённые данные) выводятся
правилами: низкая доля контрагентов с известной отраслью, отсутствие поставщиков в выборке,
выручка, нетипичная для численности, мега-группа, потенциальный (не действующий) клиент.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .okved import country_name, label, related_sectors, short_name

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "data" / "company_profiles.json"

LOW_COVERAGE = 0.35       # доля потока с известной отраслью, ниже которой предупреждаем
MEGA_GROUP = 50           # юрлиц в группе, начиная с которого показываем только агрегат
TOP_SHARE_MIN = 0.01
CONCENTRATION_MIN_N = 10  # меньше контрагентов — доля крупнейшего ничего не говорит о бизнесе
TRADE = {"45", "46", "47"}
TRADE_WEIGHT = 0.4        # вес «своего» класса торговли, когда известна отрасль товара      # отрасли контрагентов меньше 1% потока в признаки не берём

EMPLOYEES_RU = {"Более 251": "более 250", "101-250": "101–250", "16-100": "16–100", "Менее 15": "менее 15"}


@lru_cache(maxsize=1)
def _all() -> dict[str, dict]:
    return {p["id"]: p for p in json.loads(PROFILES_PATH.read_text(encoding="utf-8"))}


def get(company_id: str) -> dict | None:
    return _all().get(company_id)


def all_ids() -> list[str]:
    return list(_all())


def pct(x: float | None) -> str:
    if x is None:
        return "н/д"
    return "менее 1%" if 0 < x < 0.005 else f"{round(x * 100)}%"


def _side_view(side: dict, with_intragroup: bool) -> dict:
    tops = [{"name": label(b["okved2"]), "share": round(b["share"], 2)}
            for b in side["by_industry"] if b["share"] >= TOP_SHARE_MIN][:3]
    v = {"n": side["n"], "known_share": None if not side["n"] else round(side["known_industry_share"], 2),
         "top_industries": tops, "total_bn": round((side.get("total_rub") or 0) / 1e9, 2)}
    total, top = side.get("total_rub") or 0, side.get("top") or []
    if side["n"] >= CONCENTRATION_MIN_N and total and top:
        # зависимость от крупнейших контрагентов — по платежам через банк; при малом числе контрагентов не считаем
        v["top1_share"] = round(top[0]["amount_rub"] / total, 2)
        v["top5_share"] = round(sum(t["amount_rub"] for t in top[:5]) / total, 2)
    if with_intragroup:
        v["intragroup_share"] = round(side.get("intragroup_share") or 0, 2)  # в данных бывает null
    return v


def short_industry(p: dict, limit: int = 48) -> str:
    """Вид деятельности для списка компаний: «Торговля оптовая металлами и металлическими рудами»."""
    name = re.sub(r"\s*\(.*?\)|,? кроме .*$", "", p["industry"]).strip()
    return name if len(name) <= limit else name[:limit].rsplit(" ", 1)[0].rstrip(",") + "…"


def data_quality(p: dict) -> list[str]:
    notes = []
    if p["employees"] == "Менее 15" and p["revenue_rub"] >= 15e9:
        notes.append(f"Выручка {p['revenue_rub'] / 1e9:.1f} млрд ₽ при штате менее 15 человек — нетипично, "
                     "возможна агентская или транзитная модель".replace(".", ",", 1))
    for side, name in (("customers", "покупателях"), ("suppliers", "поставщиках")):
        if not p[side]["n"]:
            notes.append(f"Нет данных о {name} в платёжной выборке")
    cust = p["customers"]
    if cust["n"] and cust["known_industry_share"] < LOW_COVERAGE:
        notes.append(f"Отрасль известна лишь для {pct(cust['known_industry_share'])} входящего потока")
    if p["group"]["size"] >= MEGA_GROUP:
        notes.append(f"Группа из {p['group']['size']} юрлиц: анализ по агрегату долей отраслей, не по каждому юрлицу")
    if p["client_type"] == "PROSPECT":
        notes.append("Потенциальный клиент: продуктов банка нет, выводы о долговой нагрузке недоступны")
    return notes


def view(p: dict) -> dict:
    """Компания в формате frontend/mock_data.json → companies[]."""
    v = {
        "id": p["id"],
        "display": short_name(p["okved2"]),
        "okved": p["okved"],
        "industry": p["industry"],
        "revenue_bn": round(p["revenue_rub"] / 1e9, 1),
        "region": p["region"],
        "employees": EMPLOYEES_RU.get(p["employees"], p["employees"]),
        "products": p["bank_products"],
        "ved": [country_name(c) for c in p["foreign_trade"]["countries"]],
        "group_size": p["group"]["size"],
        "counterparties": {
            "customers": _side_view(p["customers"], p["group"]["size"] > 1),
            "suppliers": _side_view(p["suppliers"], p["group"]["size"] > 1),
        },
        "data_quality": data_quality(p),
    }
    if p["group"]["size"] > 1:
        v["group_industries"] = [{"name": label(g["okved2"]), "share": round(g["share"], 2)}
                                 for g in p["group"]["industries"][:4]]
    return v


def features(p: dict) -> dict:
    """Признаки для отбора событий: что у компании есть и от чего она зависит."""
    # торговля «наследует» отрасль товара (46.72 → металлургия); сам класс 46/47 у 62 из 115 компаний,
    # поэтому любая новость про опт или розницу не должна весить как новость про отрасль компании
    related = related_sectors(p["okved"])
    sectors = {p["okved2"]: 1.0}
    for g in p["group"]["industries"]:
        sectors[g["okved2"]] = max(sectors.get(g["okved2"], 0), g["share"])
    if related and p["okved2"] in TRADE:
        sectors[p["okved2"]] = min(sectors[p["okved2"]], TRADE_WEIGHT)
    for s in related:
        sectors.setdefault(s, 0.8)  # отрасль товара, которым компания торгует
    return {
        "own_sectors": sectors,  # отрасль компании и её группы с долями
        "customer_sectors": {b["okved2"]: b["share"] for b in p["customers"]["by_industry"]},
        "supplier_sectors": {b["okved2"]: b["share"] for b in p["suppliers"]["by_industry"]},
        "has_debt": bool({"кредит", "лизинг"} & set(p["bank_products"])),
        "has_deposits": "депозиты" in p["bank_products"],
        "has_guarantees": "гарантии" in p["bank_products"],
        "ved": p["foreign_trade"]["active"],
        "countries": p["foreign_trade"]["countries"],
        "revenue_bn": p["revenue_rub"] / 1e9,
    }
