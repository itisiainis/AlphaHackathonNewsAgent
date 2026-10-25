"""Чувствительность бизнеса: факторы, через которые внешние события влияют на компанию, — по данным банка, без модели.

Руководителю нужен не поток новостей, а ответ, какие факторы его бизнеса затронуты сейчас. Факторы по данным банка:
- долг под ставку — кредит или лизинг в банке (или доход по депозитам, если долга нет);
- покупатели одной отрасли — от неё заметная доля поступлений через банк;
- деньги от немногих — доля крупнейшего плательщика и пяти крупнейших;
- поставщики одной отрасли — их цены становятся вашими расходами;
- внешняя торговля — курс, пошлины и ограничения;
- налоги и сборы на товар — утильсбор, акцизы, пошлины, если отрасль ими облагается;
- свой рынок — спрос и цены в отрасли компании и её товара.
У каждого фактора — цифры профиля и честная оговорка, если данным мало можно верить. Какие события затрагивают какой фактор,
считает hits(): по видам признаков отбора, расчётам кодом и темам карточки — одинаково для разбора правилами, разбора
ИИ и примеров команды.
"""
from __future__ import annotations

import re

from . import impact, profiles
from .matching import FLOW_MIN_N
from .okved import country_name, plain, related_sectors

BUYERS_MIN = 0.2        # концентрация выручки — фактор, если от неё не меньше 20% поступлений через банк
SUPPLIERS_MIN = 0.3     # отрасль поставщиков — если ей уходит не меньше 30% платежей
TOP1_MIN, TOP5_MIN = 0.2, 0.5      # зависимость от крупнейших плательщиков
LOW_COVERAGE = 0.35     # отрасль известна для меньшей доли потока — оценка по известной части, так и пишем
# Отрасли, где налоги и сборы на товар — отдельная статья расходов (код ОКВЭД2 своей отрасли или отрасли товара)
TAXED = {"29": "утильсбор и пошлины на ввоз автомобилей", "45": "ввозные пошлины на автомобили и запчасти, утильсбор на автомобили",
         "19": "акцизы на топливо и экспортные пошлины", "06": "налог на добычу и экспортные пошлины",
         "11": "акцизы на алкоголь", "12": "акцизы на табак"}
LEVELS = {"high": 0, "medium": 1, "low": 2}


def _factor(pid: str, title: str, detail: str, level: str, roles: tuple[str, ...], watch: str) -> dict:
    return {"id": pid, "title": title, "detail": detail, "level": level, "roles": list(roles), "watch": watch}


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _pct(x: float) -> str:
    return profiles.pct(x)


def factors(p: dict) -> list[dict]:
    """Факторы чувствительности компании по убыванию значимости. Пусто не бывает: рыночный риск есть у всех."""
    v = profiles.view(p)
    products = set(p["bank_products"])
    out: list[dict] = []

    debt = [x for x in ("кредит", "лизинг") if x in products]
    if debt:
        out.append(_factor("rate", "Процентный риск", f"{_cap(' и '.join(debt))} в Альфа-Банке: стоимость обслуживания зависит от ключевой ставки",
                         "high" if len(debt) > 1 else "medium", ("cfo", "treasurer", "ceo", "accountant"), "решения ЦБ по ставке"))
    elif "депозиты" in products:
        out.append(_factor("deposits", "Процентный доход", "Депозиты в Альфа-Банке: доходность размещения следует за ключевой ставкой",
                         "low", ("treasurer", "cfo"), "решения ЦБ по ставке"))

    cust, supp = v["counterparties"]["customers"], v["counterparties"]["suppliers"]
    note = (f" (отрасль известна лишь для {_pct(cust['known_share'])} поступлений — оценка по известной части)"
            if cust["known_share"] is not None and cust["known_share"] < LOW_COVERAGE else "")
    if cust["n"] >= FLOW_MIN_N and cust["top_industries"] and cust["top_industries"][0]["share"] >= BUYERS_MIN:
        b = cust["top_industries"][0]
        out.append(_factor("buyers", f"Концентрация выручки: {b['who']}", f"{_pct(b['share'])} поступлений через Альфа-Банк{note}",
                         "high" if b["share"] >= 0.4 else "medium", ("ceo", "treasurer", "cfo"),
                         "спрос и неплатежи в их отрасли"))
    top1, top5 = cust.get("top1_share"), cust.get("top5_share")
    if top1 is not None and (top1 >= TOP1_MIN or top5 >= TOP5_MIN):
        out.append(_factor("concentration", "Зависимость от крупных плательщиков",
                         f"Крупнейший плательщик — {_pct(top1)} поступлений через Альфа-Банк, пять крупнейших — {_pct(top5)}",
                         "high" if top1 >= 0.3 or top5 >= 0.6 else "medium", ("treasurer", "cfo", "ceo", "accountant"),
                         "новости о крупных покупателях и их отраслях"))
    if supp["n"] >= FLOW_MIN_N and supp["top_industries"] and supp["top_industries"][0]["share"] >= SUPPLIERS_MIN:
        s = supp["top_industries"][0]
        out.append(_factor("suppliers", f"Зависимость от поставщиков: {s['who']}",
                         f"{_pct(s['share'])} платежей через Альфа-Банк: их цены определяют себестоимость",
                         "medium", ("cfo", "ceo"), "цены и перебои у поставщиков"))

    if p["foreign_trade"]["active"] and p["foreign_trade"]["countries"]:
        names = ", ".join(country_name(c) for c in p["foreign_trade"]["countries"][:3])
        out.append(_factor("fx", "Валютный и внешнеторговый риск", f"Контрагенты: {names} — курс, пошлины и ограничения",
                         "medium", ("treasurer", "accountant", "cfo", "ceo"), "курс рубля, ограничения, пошлины"))

    goods = [p["okved2"], *related_sectors(p["okved"])]
    taxed = next((TAXED[s] for s in goods if s in TAXED), None)
    if p["okved"].startswith("47.30"):
        taxed = TAXED["19"].split(" и ")[0]  # АЗС: акцизы на топливо, экспортные пошлины их не касаются
    if taxed:
        out.append(_factor("taxes", "Отраслевые налоги и сборы", _cap(taxed), "medium",
                         ("accountant", "cfo", "ceo"), "изменения налогов, сборов и пошлин"))

    own = plain(p["okved2"])
    extra = [plain(s) for s in related_sectors(p["okved"]) if plain(s) != own]
    out.append(_factor("market", f"Рыночный риск: {own}",
                     "Спрос и цены в отрасли" + (f"; связанные рынки — {', '.join(extra)}" if extra else ""),
                     "low", ("ceo", "cfo"), "спрос, цены и регулирование в отрасли"))
    return sorted(out, key=lambda x: LEVELS[x["level"]])


# ---------- какие события затрагивают какой фактор ----------
# по видам признаков отбора (agent/matching.py, reasons[].kind) и по расчётам кодом (agent/impact.py, заголовки)
KIND_FACTOR = {"debt": "rate", "deposits": "deposits", "macro": "rate", "fx": "fx", "sanctions": "fx",
             "customers": "buyers", "suppliers": "suppliers", "own": "market", "demand": "market"}
CALC_FACTOR = (("Стоимость долга", "rate"), ("Доход по депозитам", "deposits"), ("Валютные расчёты", "fx"),
             ("Утильсбор", "taxes"), ("Пошлина", "taxes"), ("Налог", "taxes"), ("НДС", "taxes"),
             ("Выручка от покупателей", "buyers"), ("Выручка под влиянием", "buyers"),
             ("Закупки у поставщиков", "suppliers"), ("Закупки", "suppliers"), ("Маржа", "suppliers"),
             ("Выручка", "market"))
# темы карточки — запасной путь для разборов, у которых нет признаков отбора (примеры команды)
TOPIC_FACTOR = {"ставка": "rate", "долг": "rate", "курс": "fx", "контрагенты": "buyers", "ликвидность": "concentration",
              "спрос": "market", "отрасль": "market"}
MATCHED_FACTOR = ((r"^(?:кредит|лизинг)$|решение Банка России", "rate"), (r"^депозиты$", "deposits"),
                (r"^ВЭД|страна в новости|ограничения в торговле", "fx"), (r"поступлений", "buyers"),
                (r"платежей", "suppliers"), (r"ограничения на ваш товар|ваша отрасль|ваш товар", "market"))


def hits(card: dict, company_factors: list[dict], reasons: list[dict] | None = None) -> list[str]:
    """Какие факторы чувствительности компании затрагивает карточка — id из company_factors, по порядку важности.

    Источники: признаки отбора (reasons, если есть), подписи признаков (matched), расчёты кодом (blocks.calc) и темы.
    Зависимость от крупных плательщиков — если событие несёт риск для покупателей: их неплатежи сокращают поступления."""
    have = {x["id"] for x in company_factors}
    found: list[str] = []
    for r in reasons or []:
        if r.get("kind") in KIND_FACTOR and r.get("weight", 0) > 1:
            found.append(KIND_FACTOR[r["kind"]])
    for m in card.get("matched", []):
        found += [pid for rx, pid in MATCHED_FACTOR if re.search(rx, m)]
    for c in card.get("blocks", {}).get("calc", []):
        title = impact.base_title(c["title"])
        found += [next((pid for t, pid in CALC_FACTOR if title.lower().startswith(t.lower())), "market")]
    if not found:
        found = [TOPIC_FACTOR[t] for t in card.get("topics", []) if t in TOPIC_FACTOR]
    if "buyers" in found and card.get("type") == "risk":
        found.append("concentration")
    if "rate" in found and "rate" not in have:
        found.append("deposits")
    return [pid for pid in dict.fromkeys(found) if pid in have]


def main_factor(reasons: list[dict]) -> str:
    """Фактор, который событие затрагивает сильнее всего: по самому весомому признаку отбора."""
    best = max((r for r in reasons if r.get("kind") in KIND_FACTOR and r.get("weight", 0) > 1),
               key=lambda r: r["weight"], default=None)
    return KIND_FACTOR[best["kind"]] if best else "market"


def diverse(pairs: list, k: int) -> list:
    """k лучших событий по разным факторам: сначала лучшее по каждому фактору, потом по рейтингу. Иначе три новости
    о рынке топлива подряд вытесняют решение ЦБ, хотя у компании кредит и лизинг."""
    first, seen = [], set()
    for ev, m in pairs:
        pid = main_factor(m.reasons)
        if pid not in seen:
            seen.add(pid)
            first.append((ev, m))
    chosen = first[:k]
    return chosen + [x for x in pairs if x not in chosen][: k - len(chosen)]
