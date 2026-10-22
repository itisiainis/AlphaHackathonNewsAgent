"""Отбор и ранжирование событий под компанию — правилами по признакам профиля, без LLM.

Что делает: для пары «событие × компания» считает релевантность и объясняет её списком
причин (matched[] в карточке) — критерий 01 кейса: отбор с учётом релевантных признаков.

Как устроено: у события есть темы (ставка, долг, курс, …), отрасли ОКВЭД2 и страны.
У компании — признаки из profiles.features(). Каждое совпадение добавляет причину с весом:
  • ставка/долг × есть кредит или лизинг            → 3   (стоимость долга)
  • ставка × нет долга, есть депозиты                → 1.5 (доходность остатков — возможность)
  • ставка × своя отрасль, где спрос идёт в кредит   → 2   (ипотека, автокредиты, лизинг; отрасль группы не в счёт)
  • решение Банка России по ключевой ставке          → 2.5 (касается всех компаний, даже без кредитов)
  • курс × ВЭД                                       → 3, ещё +2 если страна ВЭД упомянута в новости
  • отрасль события = отрасль компании или группы    → 4 × доля отрасли в группе
  • отрасль события = отрасль покупателей/поставщиков → 6 × доля в денежном потоке (от 3% потока и от 10 контрагентов)
  • санкции, запреты, пошлины на ваш товар или страну ВЭД → 3
  • спрос × компания работает на конечного потребителя → 1.5
Понижающие множители: отрасль упомянута вскользь (×0.5), рынок другой страны без ВЭД с ней (×0.15; биржевые
цены — мировой ориентир, не чужой рынок), подборка из нескольких сюжетов (×0.3; пост экономиста банка —
не подборка), криминальная хроника (×0.4), ежедневный официальный курс (×0.3).
Сумма весов — релевантность. Порог SELECT_MIN отсекает события «не про нас»: на такие пары
агент не запускается, и бюджет LLM не тратится. Порядок в ленте дальше уточняется весами ролей
(score = сила влияния × 10 + сумма весов тем роли — как во фронте).

Когда отобранного мало (меньше трёх событий), лента дополняется главным в экономике — macro_events():
ставка, курс, инфляция за последние дни, по важности (docs/product.md, «Нет важных событий»).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

from . import okved
from .impact import FOREIGN
from .kb import ECONOMIST_RE, bank_voice_sources
from .okved import RATE_SENSITIVE, country_name, plain, who

SELECT_MIN = 2.5
FLOW_MIN = 0.03
FLOW_MIN_N = 10  # меньше контрагентов в выборке банка — доля отрасли в потоке ничего не говорит о бизнесе
CONSUMER_SECTORS = {"10", "11", "29", "41", "45", "47", "55", "56", "68", "79"}
DOMESTIC = r"росси|\bрф\b|рубл|\bцб\b|москв|отечествен|минфин росс|минэк|минпромторг"
FOREIGN_FACTOR = 0.15
# Биржевая цена — мировой ориентир для всех, а не рынок другой страны: «медь на LME», «нефть Brent»
BENCHMARK = r"\blme\b|\bлбм\b|лондонск\w* бирж\w* металл|\bcomex\b|\bbrent\b|\burals\b|биржев\w* цен"
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
# Отрасль названа один раз в тексте и ни разу в заголовке — новость не про неё (okved.passing_sectors)
PASSING_FACTOR = 0.5
# Санкции, запреты и пошлины, названные в одном предложении с товаром компании или страной её ВЭД
SANCTIONS = re.compile(r"санкц|запрет|эмбарго|потол\w* цен|пошлин")
SANCTIONS_W = 3
# страна ВЭД засчитывается, только если ограничение — в торговле с Россией: «американские пошлины… продажи
# в Китае» (Volkswagen) — не про ваши поставки из Китая
RUSSIA = re.compile(r"росси|\bрф\b|отечествен")
# ограничение снимают или смягчают — для разбора это возможность, а не риск
EASED = re.compile(r"сниз|сниж|отмен|снял|снят|смягч|ослаб|исключ")
# Решение Банка России по ключевой ставке: «сохранил», «снизил», «принял решение повысить». Прогноз («сохранит»)
# и анонс заседания решением не считаются
CB_BANK = re.compile(r"банк\w* росси|\bцб\b|центробанк|регулятор")
CB_VERB = re.compile(r"(?:сохранил|оставил|снизил|понизил|повысил|поднял)\w*|принял\w* решени|"
                     r"решени\w* (?:о |об )?(?:сохран|оставить|сниз|сниж|повыс|повыш)")
CB_BASE = SELECT_MIN
BANK_VOICE = bank_voice_sources()  # каналы самого банка: bank_voice в data/sources.json


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


def bank_voice(event: dict) -> bool:
    """Пост канала банка, где говорит его экономист, — аналитика банка с одним тезисом, а не подборка новостей,
    хотя доводы в нём и идут списком («🟥 Инфляция выше ожиданий…»). Реклама и обзоры канала — обычные посты."""
    own = event.get("bank_voice") or event.get("source") in BANK_VOICE
    return bool(own and ECONOMIST_RE.search(event.get("text", "")))


def cb_decision(title: str, text: str) -> bool:
    """Решение Банка России по ключевой ставке в заголовке или начале текста (а не прогноз и не анонс заседания)."""
    head = (title + ".\n" + text[:600]).lower().replace("ё", "е")
    for s in okved.SENTENCE_END.split(head):
        if okved.FOREIGN_CB.search(s) and not okved.RUS_CB.search(s):
            continue  # ФРС, ЕЦБ, «ЦБ Турции»
        if CB_BANK.search(s) and re.search(r"ключев\w* ставк", s) and CB_VERB.search(s):
            return True
    return False


def _country_rx(c: str) -> str:
    """Страна ВЭД в тексте новости: «Китай» → «Китая, КНР, Пекин»; для стран без словаря — первые 5 букв."""
    return FOREIGN.get(c) or re.escape(c.lower()[:5])


@lru_cache(maxsize=4096)
def _restrictions(title: str, body: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Предложения новости с санкциями, запретами или пошлинами и отрасли, названные в каждом из них."""
    sentences = okved.SENTENCE_END.split(title.lower() + ".\n" + body.lower())
    return tuple((s, frozenset(okved.sectors_in(s))) for s in sentences if SANCTIONS.search(s))


def _sanctions(title: str, body: str, goods: set[str], countries: list[str]) -> tuple[str, str, bool] | None:
    """Ограничения на товар компании или в торговле России со страной её ВЭД — в одном предложении с самим
    ограничением. Третий элемент — ограничение смягчают («снизил пошлины», «снял запрет»)."""
    for s, secs in _restrictions(title, body):
        hit = sorted(secs & goods)
        if hit:
            return "goods", hit[0], bool(EASED.search(s))
        c = next((c for c in countries if re.search(_country_rx(c), s)), None)
        if c and RUSSIA.search(s):
            return "country", c, bool(EASED.search(s))
    return None


def match(event: dict, feats: dict, products: list[str]) -> Match:
    m = Match(0.0)
    topics = set(event.get("topics", []))
    sectors = set(event.get("sectors", []))
    title, body = event.get("title", ""), event.get("text", "")
    text = (title + " " + body).lower()
    roles = feats.get("sector_role", {})
    # своя отрасль и отрасль товара; отрасль компаний группы — не про спрос и не про товар самой компании
    own_goods = {s for s in feats["own_sectors"] if roles.get(s, "own") in ("own", "goods")}

    if topics & {"ставка", "долг"}:
        debt = [prod for prod in ("кредит", "лизинг") if prod in products]
        if feats["has_debt"] and debt:
            # «ставка» — решение ЦБ; одно слово «кредит» (потребкредиты, суды о кредитах) весит вдвое меньше.
            # Вес — на долг в целом, а не на каждый продукт: компания с одним кредитом зависит от ставки не вдвое меньше
            w = (3 if "ставка" in topics else 1.5) / len(debt)
            for prod in debt:
                _add(m, prod, "платежи по долгу зависят от ставки", w, kind="debt")
        elif feats["has_deposits"] and "ставка" in topics:
            _add(m, "депозиты", "доход по депозитам зависит от ставки", 1.5, kind="deposits")
        if "ставка" in topics and own_goods & RATE_SENSITIVE:
            _add(m, "спрос клиентов в кредит", "ваши покупатели берут кредиты и ипотеку — ставка меняет их спрос", 2,
                 kind="demand")
    # Решение по ключевой ставке касается всех: стоимость денег меняется и у покупателей, и у поставщиков
    if "ставка" in topics and cb_decision(title, body):
        if feats["has_debt"] or feats["has_deposits"]:
            _add(m, "решение Банка России по ставке", "ставка Банка России меняет стоимость денег для всех компаний",
                 CB_BASE, kind="macro")
        else:
            _add(m, "решение Банка России по ставке", "кредитов в Альфа-Банке у вас нет — влияние через покупателей и курс",
                 CB_BASE, kind="macro")

    if "курс" in topics and feats["ved"]:
        countries = feats["countries"]
        _add(m, "ВЭД: " + ", ".join(country_name(c) for c in countries[:3]),
             "расчёты с зарубежными партнёрами зависят от курса", 3, kind="fx")
        hit = [c for c in countries if re.search(_country_rx(c), text)]
        if hit:
            _add(m, "страна в новости: " + country_name(hit[0]), "новость прямо про страну, с которой вы торгуете", 2,
                 kind="fx")

    # отрасль, названная один раз в тексте и ни разу в заголовке, весит вдвое меньше; разметка модели — как заголовок
    passing = okved.passing_sectors(title, body) - set(event.get("llm_sectors", []))
    for s in sectors:
        k = PASSING_FACTOR if s in passing else 1.0
        note = " (отрасль упомянута вскользь — вес снижен)" if s in passing else ""
        own = feats["own_sectors"].get(s)
        if own:
            role = roles.get(s, "own")
            chip, why = OWN_ROLE[role]
            _add(m, f"{chip}: {plain(s)}", why + note, 4 * own * k, kind="own", sector=s, role=role)
        for side, name, kind, why in (
                ("customer_sectors", "поступлений", "customers", "ваши покупатели: что меняется у них, отразится на ваших продажах"),
                ("supplier_sectors", "платежей", "suppliers", "ваши поставщики: что меняется у них, отразится на ваших закупках")):
            share = feats[side].get(s, 0)
            # по нескольким контрагентам доля отрасли в потоке случайна — такой признак не считаем
            enough = feats.get(kind + "_n", FLOW_MIN_N) >= FLOW_MIN_N
            if share >= FLOW_MIN and enough:
                _add(m, f"{round(share * 100)}% {name} — {who(s)}", why + note, 6 * share * k, kind=kind, sector=s,
                     share=round(share, 2))

    hit = _sanctions(title, body, own_goods, feats["countries"])
    if hit and hit[0] == "goods":
        _add(m, f"ограничения на ваш товар: {plain(hit[1])}", "ограничения касаются вашего товара", SANCTIONS_W,
             kind="sanctions", sector=hit[1], eased=hit[2])
    elif hit:
        _add(m, f"ограничения в торговле: {country_name(hit[1])}", "ограничения касаются страны, с которой вы торгуете",
             SANCTIONS_W, kind="sanctions", country=hit[1], eased=hit[2])

    if "спрос" in topics and own_goods & CONSUMER_SECTORS:
        _add(m, "зависимость от потребительского спроса", "вы продаёте конечным потребителям", 1.5, kind="demand")

    for f, why, w in penalties(event, feats["countries"]):
        m.score *= w
        m.reasons.append({"feature": f, "why": why, "weight": w})

    m.score = round(m.score, 2)
    return m


def penalties(event: dict, countries: list[str] = ()) -> list[tuple[str, str, float]]:
    """Понижающие множители события: (признак, объяснение, множитель). Общие для отбора и для главного в экономике."""
    text = (event.get("title", "") + " " + event.get("text", "")).lower()
    out = []
    # Рынок другой страны, с которой компания не работает: «ипотека в Европе», «Минфин США».
    # Биржевая цена (LME, Brent) — мировой ориентир, а не чужой рынок
    abroad = [c for c, rx in FOREIGN.items() if re.search(rx, text)]
    if abroad and not re.search(DOMESTIC, text) and not set(abroad) & set(countries) and not re.search(BENCHMARK, text):
        out.append(("зарубежный рынок", "событие о рынке другой страны: вес снижен", FOREIGN_FACTOR))
    # Уголовные дела и суды над людьми — не бизнес-событие для компании (если не о её контрагенте, а этого мы не знаем)
    if re.search(CRIME, text):
        out.append(("суд или уголовное дело", "частный спор или уголовное дело, не рыночное событие: вес снижен",
                     CRIME_FACTOR))
    # Ежедневная рутина (официальный курс на завтра) — не событие: изменения курса ловятся по другим новостям
    if re.search(ROUTINE, text):
        out.append(("рутинная публикация", "ежедневный официальный курс, не событие: вес снижен", ROUTINE_FACTOR))
    # Подборка из нескольких новостей («1️⃣ … 2️⃣ … 3️⃣ …») совпадает со всем понемногу — это не одно событие.
    # Пост экономиста банка — один тезис с доводами списком, а не подборка
    if len(DIGEST_RE.findall(event.get("text", ""))) >= 3 and not bank_voice(event):
        out.append(("подборка новостей", "пост из нескольких сюжетов: вес снижен", DIGEST_FACTOR))
    return out


def rank(events: list[dict], feats: dict, products: list[str]) -> list[tuple[dict, Match]]:
    """События по убыванию релевантности, только прошедшие порог."""
    pairs = [(e, match(e, feats, products)) for e in events]
    pairs = [(e, mt) for e, mt in pairs if mt.selected]
    return sorted(pairs, key=lambda x: (-(x[1].score * (1 + x[0].get("importance", 5) / 10))))


# Главное в экономике для тонкой ленты: ставка, курс, инфляция — касаются любой компании
MACRO_DAYS = 3
MACRO_WHY = {"ставка": "ключевая ставка меняет стоимость денег для всех компаний",
             "курс": "курс рубля меняет цены импорта и выручку от экспорта",
             "инфляция": "инфляция меняет цены закупок, зарплаты и спрос"}
# официальный курс на день — не событие, но в тонкой ленте это единственный ориентир по валюте
ROUTINE_FX_WHY = "официальный курс Банка России — ориентир для расчётов в валюте"
INFLATION = re.compile(r"инфляц|потребительск\w* цен|рост\w* цен")


def macro_kind(event: dict) -> str | None:
    """Тема события из главного в экономике: ставка, курс или инфляция; None — событие не макро."""
    topics = set(event.get("topics", []))
    if "ставка" in topics:
        return "ставка"
    if "курс" in topics:
        return "курс"
    if INFLATION.search((event.get("title", "") + " " + event.get("text", "")[:600]).lower()):
        return "инфляция"
    return None


def macro_events(events: list[dict], today: date, exclude: set[str] | frozenset = frozenset(),
                 limit: int = 3) -> list[tuple[dict, str, str]]:
    """До limit макрособытий за последние MACRO_DAYS дней: (событие, тема, почему важно). Сначала важнее,
    при равной важности — пост экономиста банка и решение ЦБ, затем свежее; разные темы — раньше повторов.
    Подборки, суды и новости чужих рынков (ставка ФРС) не берутся. Ежедневный официальный курс — только
    если других новостей о курсе нет, и не больше одного. exclude — события, которые уже есть в ленте."""
    since, until = (today - timedelta(days=MACRO_DAYS)).isoformat(), today.isoformat()
    main, routine = [], []
    for e in events:
        kind = macro_kind(e)
        if not kind or e["id"] in exclude or not since <= (e.get("published") or "") <= until:
            continue
        head = (e.get("title", "") + ".\n" + e.get("text", "")[:600]).lower()
        if okved.FOREIGN_CB.search(head) and not okved.RUS_CB.search(head):
            continue
        pen = {f for f, _, _ in penalties(e)}
        if not pen:
            main.append((e, kind, MACRO_WHY[kind]))
        elif pen == {"рутинная публикация"} and kind == "курс":
            routine.append((e, kind, ROUTINE_FX_WHY))

    def order(cands: list) -> list:
        cands = sorted(cands, key=lambda c: c[0].get("published") or "", reverse=True)  # при прочих равных — свежее
        return sorted(cands, key=lambda c: (-(c[0].get("importance") or 0),
                                            not (bank_voice(c[0]) or cb_decision(c[0].get("title", ""), c[0].get("text", ""))),
                                            list(MACRO_WHY).index(c[1])))

    main = order(main)
    picked: list[tuple[dict, str, str]] = []
    for c in main:  # сначала по одному событию на тему, потом остальные по порядку
        if c[1] not in {k for _, k, _ in picked} and len(picked) < limit:
            picked.append(c)
    for c in main:
        if all(c[0] is not e for e, _, _ in picked) and len(picked) < limit:
            picked.append(c)
    if routine and len(picked) < limit and "курс" not in {k for _, k, _ in picked}:
        picked.append(order(routine)[0])
    return picked
