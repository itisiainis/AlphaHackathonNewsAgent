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
Понижающие множители: рынок другой страны без ВЭД с ней (×0.15), подборка из нескольких сюжетов (×0.3),
криминальная хроника (×0.4), ежедневный официальный курс (×0.3).
Сумма весов — релевантность. Порог SELECT_MIN отсекает события «не про нас»: на такие пары
агент не запускается, и бюджет LLM не тратится. Порядок в ленте дальше уточняется весами ролей
(score = сила влияния × 10 + сумма весов тем роли — как во фронте).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .impact import FOREIGN
from .okved import RATE_SENSITIVE, country_name, plain, who

SELECT_MIN = 2.5
FLOW_MIN = 0.03
CONSUMER_SECTORS = {"10", "11", "29", "41", "45", "47", "55", "56", "68", "79"}
DOMESTIC = r"росси|\bрф\b|рубл|\bцб\b|москв|отечествен|минфин росс|минэк|минпромторг"
FOREIGN_FACTOR = 0.15
# пункты подборки: «1️⃣», «1.», «1)» или строка, начинающаяся со значка-маркера («🔴 », «• », «▪️ »)
DIGEST_RE = re.compile(r"\d\ufe0f?\u20e3|^\s*\d{1,2}[.)]\s|^\s*[^\w\s«\"(\[-]{1,3}\s", re.M)
DIGEST_FACTOR = 0.3
MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
ROUTINE = (r"установил\w* официальн\w* курс|сохраняет (?:официальн\w* )?курс|"
           r"официальн\w* курс\w* [^.]{0,40}на (?:сегодня|завтра|\d{1,2} (?:" + MONTHS + "))")
ROUTINE_FACTOR = 0.3
CRIME = (r"взятк|приговор|колони[июя] |уголовн\w* дел|осужден|мошеннич|задержан|"
         r"верховн\w* суд|суд\w* (?:защитил|обязал|признал|отказал|запретил)")
CRIME_FACTOR = 0.4


@dataclass
class Match:
    score: float
    matched: list[str] = field(default_factory=list)       # короткие признаки для чипов
    reasons: list[dict] = field(default_factory=list)      # {feature, why, weight} — в трассу

    @property
    def selected(self) -> bool:
        return self.score >= SELECT_MIN


def _add(m: Match, chip: str, why: str, w: float, **meta) -> None:
    """meta — вид признака (kind), отрасль и доля: по ним разбор без модели пишет вывод, не разбирая текст чипа."""
    m.score += w
    if chip not in m.matched:
        m.matched.append(chip)
    m.reasons.append({"feature": chip, "why": why, "weight": round(w, 2), **meta})


# чем отрасль события приходится компании → подпись чипа и объяснение простыми словами
OWN_ROLE = {"own": ("ваша отрасль", "событие в вашей отрасли"),
            "goods": ("отрасль вашего товара", "событие в отрасли, чей товар вы продаёте"),
            "group": ("отрасль вашей группы", "событие в отрасли компаний вашей группы")}


def match(event: dict, feats: dict, products: list[str]) -> Match:
    m = Match(0.0)
    topics = set(event.get("topics", []))
    sectors = set(event.get("sectors", []))
    text = (event.get("title", "") + " " + event.get("text", "")).lower()

    if topics & {"ставка", "долг"}:
        if feats["has_debt"]:
            # «ставка» — решение ЦБ; одно слово «кредит» (потребкредиты, суды о кредитах) весит вдвое меньше
            w = 1.5 if "ставка" in topics else 0.75
            for prod in ("кредит", "лизинг"):
                if prod in products:
                    _add(m, prod, "платежи по долгу зависят от ставки", w, kind="debt")
        elif feats["has_deposits"] and "ставка" in topics:
            _add(m, "депозиты", "доход по депозитам зависит от ставки", 1.5, kind="deposits")
        if "ставка" in topics and set(feats["own_sectors"]) & RATE_SENSITIVE:
            _add(m, "спрос клиентов в кредит", "ваши покупатели берут кредиты и ипотеку — ставка меняет их спрос", 2,
                 kind="demand")

    if "курс" in topics and feats["ved"]:
        countries = feats["countries"]
        _add(m, "ВЭД: " + ", ".join(country_name(c) for c in countries[:3]),
             "расчёты с зарубежными партнёрами зависят от курса", 3, kind="fx")
        hit = [c for c in countries if c.lower()[:5] in text]
        if hit:
            _add(m, "страна в новости: " + country_name(hit[0]), "новость прямо про страну, с которой вы торгуете", 2,
                 kind="fx")

    roles = feats.get("sector_role", {})
    for s in sectors:
        own = feats["own_sectors"].get(s)
        if own:
            role = roles.get(s, "own")
            chip, why = OWN_ROLE[role]
            _add(m, f"{chip}: {plain(s)}", why, 4 * own, kind="own", sector=s, role=role)
        for side, name, kind, why in (
                ("customer_sectors", "поступлений", "customers", "ваши покупатели: что меняется у них, отразится на ваших продажах"),
                ("supplier_sectors", "платежей", "suppliers", "ваши поставщики: что меняется у них, отразится на ваших закупках")):
            share = feats[side].get(s, 0)
            if share >= FLOW_MIN:
                _add(m, f"{round(share * 100)}% {name} — {who(s)}", why, 6 * share, kind=kind, sector=s, share=round(share, 2))

    if "спрос" in topics and set(feats["own_sectors"]) & CONSUMER_SECTORS:
        _add(m, "зависимость от потребительского спроса", "вы продаёте конечным потребителям", 1.5, kind="demand")

    # Рынок другой страны, с которой компания не работает: «ипотека в Европе», «Минфин США»
    abroad = [c for c, rx in FOREIGN.items() if re.search(rx, text)]
    if abroad and not re.search(DOMESTIC, text) and not set(abroad) & set(feats["countries"]):
        m.score *= FOREIGN_FACTOR
        m.reasons.append({"feature": "зарубежный рынок", "why": "событие о рынке другой страны: вес снижен", "weight": FOREIGN_FACTOR})
    # Уголовные дела и суды над людьми — не бизнес-событие для компании (если не о её контрагенте, а этого мы не знаем)
    if re.search(CRIME, text):
        m.score *= CRIME_FACTOR
        m.reasons.append({"feature": "суд или уголовное дело", "weight": CRIME_FACTOR,
                          "why": "частный спор или уголовное дело, не рыночное событие: вес снижен"})
    # Ежедневная рутина (официальный курс на завтра) — не событие: изменения курса ловятся по другим новостям
    if re.search(ROUTINE, text):
        m.score *= ROUTINE_FACTOR
        m.reasons.append({"feature": "рутинная публикация", "why": "ежедневный официальный курс, не событие: вес снижен", "weight": ROUTINE_FACTOR})
    # Подборка из нескольких новостей («1️⃣ … 2️⃣ … 3️⃣ …») совпадает со всем понемногу — это не одно событие
    if len(DIGEST_RE.findall(event.get("text", ""))) >= 3:
        m.score *= DIGEST_FACTOR
        m.reasons.append({"feature": "подборка новостей", "why": "пост из нескольких сюжетов: вес снижен", "weight": DIGEST_FACTOR})

    m.score = round(m.score, 2)
    return m


def rank(events: list[dict], feats: dict, products: list[str]) -> list[tuple[dict, Match]]:
    """События по убыванию релевантности, только прошедшие порог."""
    pairs = [(e, match(e, feats, products)) for e in events]
    pairs = [(e, mt) for e, mt in pairs if mt.selected]
    return sorted(pairs, key=lambda x: (-(x[1].score * (1 + x[0].get("importance", 5) / 10))))
