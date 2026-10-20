"""Отбор и ранжирование событий под компанию — правилами по признакам профиля, без LLM.

Что делает: для пары «событие × компания» считает релевантность и объясняет её списком
причин (matched[] в карточке) — критерий 01 кейса: отбор с учётом релевантных признаков.

Как устроено: у события есть темы (ставка, долг, курс, …), отрасли ОКВЭД2 и страны.
У компании — признаки из profiles.features(). Каждое совпадение добавляет причину с весом:
  • ставка/долг × есть кредит или лизинг            → 3   (стоимость долга)
  • ставка × нет долга, есть депозиты                → 1.5 (доходность остатков — возможность)
  • ставка × отрасль, где спрос идёт в кредит        → 2   (ипотека, автокредиты, лизинг)
  • курс × ВЭД                                       → 3, ещё +2 если страна ВЭД упомянута в новости
  • отрасль события = отрасль компании или группы    → 4 × доля отрасли в группе
  • отрасль события = отрасль покупателей/поставщиков → 6 × доля в денежном потоке (от 3% потока)
  • спрос × компания работает на конечного потребителя → 1.5
Сумма весов — релевантность. Порог SELECT_MIN отсекает события «не про нас»: на такие пары
агент не запускается, и бюджет LLM не тратится. Порядок в ленте дальше уточняется весами ролей
(score = сила влияния × 10 + сумма весов тем роли — как во фронте).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .okved import RATE_SENSITIVE, country_name, label

SELECT_MIN = 2.5
FLOW_MIN = 0.03
CONSUMER_SECTORS = {"10", "11", "29", "41", "45", "47", "55", "56", "68", "79"}


@dataclass
class Match:
    score: float
    matched: list[str] = field(default_factory=list)       # короткие признаки для чипов
    reasons: list[dict] = field(default_factory=list)      # {feature, why, weight} — в трассу

    @property
    def selected(self) -> bool:
        return self.score >= SELECT_MIN


def _add(m: Match, chip: str, why: str, w: float) -> None:
    m.score += w
    if chip not in m.matched:
        m.matched.append(chip)
    m.reasons.append({"feature": chip, "why": why, "weight": round(w, 2)})


def match(event: dict, feats: dict, products: list[str]) -> Match:
    m = Match(0.0)
    topics = set(event.get("topics", []))
    sectors = set(event.get("sectors", []))
    text = (event.get("title", "") + " " + event.get("text", "")).lower()

    if topics & {"ставка", "долг"}:
        if feats["has_debt"]:
            for prod in ("кредит", "лизинг"):
                if prod in products:
                    _add(m, prod, "стоимость обслуживания долга зависит от ставки", 1.5)
        elif feats["has_deposits"] and "ставка" in topics:
            _add(m, "депозиты", "доходность размещения остатков зависит от ставки", 1.5)
        if "ставка" in topics and set(feats["own_sectors"]) & RATE_SENSITIVE:
            _add(m, "спрос клиентов в кредит", "покупатели компании финансируются кредитами или ипотекой", 2)

    if "курс" in topics and feats["ved"]:
        countries = feats["countries"]
        _add(m, "ВЭД: " + ", ".join(country_name(c) for c in countries[:3]),
             "внешнеторговые расчёты зависят от курса", 3)
        hit = [c for c in countries if c.lower()[:5] in text]
        if hit:
            _add(m, "страна в новости: " + country_name(hit[0]), "новость прямо касается страны ВЭД", 2)

    for s in sectors:
        own = feats["own_sectors"].get(s)
        if own:
            _add(m, "отрасль: " + label(s), "событие в отрасли компании, её группы или её товара", 4 * own)
        for side, name in (("customer_sectors", "входящего"), ("supplier_sectors", "исходящего")):
            share = feats[side].get(s, 0)
            if share >= FLOW_MIN:
                _add(m, f"{round(share * 100)}% {name} потока — {label(s)}",
                     "событие в отрасли контрагентов", 6 * share)

    if "спрос" in topics and set(feats["own_sectors"]) & CONSUMER_SECTORS:
        _add(m, "зависимость от потребительского спроса", "компания работает на конечного потребителя", 1.5)

    m.score = round(m.score, 2)
    return m


def rank(events: list[dict], feats: dict, products: list[str]) -> list[tuple[dict, Match]]:
    """События по убыванию релевантности, только прошедшие порог."""
    pairs = [(e, match(e, feats, products)) for e in events]
    pairs = [(e, mt) for e, mt in pairs if mt.selected]
    return sorted(pairs, key=lambda x: (-(x[1].score * (1 + x[0].get("importance", 5) / 10))))
