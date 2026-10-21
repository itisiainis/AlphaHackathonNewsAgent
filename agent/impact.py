"""Калькулятор эффекта события для компании: формулы в коде, без LLM.

Зачем: «как событие повлияет на компанию» — в рублях, а не словами. Модель хорошо объясняет,
но плохо считает, поэтому цифры эффекта считает код по явным формулам, а модель только
ссылается на результат. Каждый расчёт показывает формулу, входные данные с происхождением
и допущения — чтобы руководитель мог проверить и пересчитать под себя.

Что умеет (шок берётся из текста новости правилами; нет шока — нет расчёта):
- ключевая ставка → стоимость плавающего долга и доход по депозитам на каждый 1 млрд ₽;
- курс рубля → стоимость импортных закупок (или валютной выручки) на каждый 1 млрд ₽;
- пошлина / утильсбор / налог на прибыль → доп. расходы на 1 млрд ₽ базы (экспорт и импорт — раздельно);
- цена того, что компания закупает (топливо для АЗС, целлюлоза для картона, металл для стройки) →
  закупки на 1 млрд ₽, а если в новости двигаются и закупочные, и розничные цены — одна оценка маржи;
- изменение спроса или цен в отрасли → выручка от покупателей этой отрасли по доле поступлений через банк
  и расходы на закупки у поставщиков этой отрасли.

Экономика, а не совпадение слов:
- прогноз, ожидание, предложение или условие («аналитики ожидают», «может», «если») — это сценарий
  «Если …» с низкой надёжностью и без знака риска / возможности, а не свершившийся факт;
- изменение за месяц или неделю не переносится на годовую выручку; месяц к тому же месяцу прошлого года —
  с оговоркой «если так будет весь год»;
- знак — по экономике: рост расхода (долг, закупки, пошлина, налог) — риск, падение выручки — риск;
- одна оценка выручки на событие; фраза, из которой взят шок ставки, курса или пошлины, второй раз не считается.

Честность (ограничение кейса 06): суммы долга, депозитов и импорта в данных кейса нет — есть только
факт наличия продукта и ВЭД. Поэтому эффект даётся на единицу базы с полем «введите свою сумму».
Доли отраслей — по платёжной выборке банка, где отрасль известна лишь для части потока:
это оценка по известной части, а не полный эффект. Меньше 10 плательщиков в выборке — оценку по долям не даём.
"""
from __future__ import annotations

import re

from .okved import country_name, label, related_sectors, sectors_in, who
from .profiles import pct

BN = 1e9
FLOW_MIN = 0.02     # отрасль контрагентов меньше 2% потока не считаем
EXPOSURE_MIN = 0.10  # «выручка под влиянием» — только для отраслей от 10% поступлений
FLOW_N_MIN = 10     # меньше 10 плательщиков в выборке банка — доли отраслей случайны, оценку по ним не даём
NUM = r"(\d+(?:[.,]\d+)?)"
PCT = NUM + r"\s*%"


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def num(x: float, digits: int = 1) -> str:
    """45.80 → «45,8», 20.0 → «20»."""
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{x:.0f}"
    return s.replace(".", ",")


def rub(x: float) -> str:
    """Сумма по-русски с разумной точностью: «1,3 млрд ₽», «20 млн ₽», «−350 тыс. ₽»."""
    sign, a = ("−" if x < 0 else ""), abs(x)
    if a >= BN:
        return f"{sign}{num(a / BN, 1)} млрд ₽"
    if a >= 1e6:
        return f"{sign}{num(a / 1e6, 0 if a >= 1e7 else 1)} млн ₽"
    return f"{sign}{num(a / 1e3, 0)} тыс. ₽"


def pp(x: float) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "") + num(abs(x), 2) + " п.п."


def _pc(x: float) -> str:
    """+8 → «+8%», −1,5 → «−1,5%»."""
    return ("+" if x > 0 else "−") + num(abs(x), 2) + "%"


def _plus(x: float) -> str:
    return "+" if x > 0 else ""


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+(?=[А-ЯA-Z«\"(\d])", text) if s.strip()]


def _low(s: str) -> str:
    return s.lower().replace("ё", "е")


def _blank(m: re.Match) -> str:
    return " " * len(m.group())  # вырезаем оборот, не сдвигая позиции в строке


DOWN = r"сниз|понизи|снижен|смягч|уменьш|сократ|сокращ|упал|упад|подешев|обвал|рухн|падени"
UP = r"повыс|повыш|поднял|подня|повышен|ужесточ|увелич|вырос|вырас|рост|подорож|ускор"
HOLD = r"сохрани|оставил|без изменени|не стал менять|паузу"

# Прогноз, ожидание, предложение или условие — не свершившийся факт: расчёт становится сценарием «Если …»
MODAL = re.compile(r"ожида|прогноз|\bмо(?:жет|гут)\b|\bесли\b|вероятн|планиру|аналитик|предлага|предложил|рассматрива|"
                   r"не исключ|по мнению")
# Обороты, где те же слова факт прогнозом не делают: «как и ожидалось», «сильнее ожиданий»,
# «вариант снижения не рассматривался», «ухудшил прогноз по инфляции», «по данным аналитиков»
NOT_MODAL = re.compile(
    r"(?:как и|как|чем) ожидал\w*|ожидаемо\b|"
    r"(?:вопреки|выше|ниже|сильнее|слабее|в соответствии с|в линии с|в рамках)\s+(?:ожидани|прогноз)\w*"
    r"(?:\s+(?:аналитик|рынк|экономист)\w*)?|не рассматрива\w*|"
    r"(?:ухудшил|улучшил|понизил|повысил|сохранил|пересмотрел|изменил|снизил|обновил)\w*\s+(?:\w+\s+)?прогноз\w*|"
    r"по (?:данным|подсчетам) аналитик\w*")
# «…, ожидают аналитики», «…, считают экономисты» после числа — тоже прогноз; «…, отметив, что ожидает…» — нет
MODAL_AFTER = re.compile(r",\s*(?:—\s*)?(?:ожида|прогнозир|счита|полага|предполага|допуска|не исключа)\w*\s|по мнению")
SCENARIO = "Это сценарий из новости, а не свершившийся факт"


def _modal(t: str, end: int | None = None) -> bool:
    """Фраза о прогнозе или условии, а не о факте. end — конец числа шока: слова прогноза ищем до него
    («аналитики ожидают, что ЦБ снизит ставку на 50 б.п.») и в оговорке после него («…, ожидают аналитики»),
    но не в продолжении факта («ЦБ снизил ставку на 50 б.п. и может продолжить снижение»)."""
    t = NOT_MODAL.sub(_blank, t)
    if re.search(r"сообщ|подсчит|зафиксир|по данным", t):  # «аналитики "Автостата" подсчитали» — это отчёт о факте
        t = re.sub(r"аналитик\w*", _blank, t)
    end = len(t) if end is None else end
    return bool(MODAL.search(t[:end]) or MODAL_AFTER.search(t[end:]))


def _scenario(c: dict, cond: str) -> dict:
    """Расчёт по прогнозу из новости: «Если ставку снизят на 0,5 п.п.: стоимость долга», без знака, надёжность низкая."""
    t = c["title"].replace(": ", " — ")
    return c | {"title": f"Если {cond}: {t[:1].lower() + t[1:]}", "sign": "neutral", "strength": "low",
                "assumptions": c["assumptions"] + [SCENARIO]}


# Период изменения. Изменение за месяц или неделю не переносим на годовую выручку
MONTHS_IN = r"январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|ноябре|декабре"
MONTHS_TO = r"январю|февралю|марту|апрелю|маю|июню|июлю|августу|сентябрю|октябрю|ноябрю|декабрю"
SHORT = re.compile(r"\bм/м\b|месяц к месяцу|к предыдущему месяцу|\bза (?:последн\w+ )?(?:месяц|неделю|сутки|день|квартал)\b|"
                   rf"\bв (?:{MONTHS_IN})\b|\bк (?:{MONTHS_TO})\b|с начала (?:месяца|недели)|"
                   r"в (?:прошлом|минувшем|этом|текущем) месяце|\bкв/кв\b|"
                   r"\bв (?:i|ii|iii|iv|первом|втором|третьем|четвертом) квартале\b")
ANNUAL = re.compile(r"\bг/г\b|год к году|в годовом (?:выражении|исчислении)|годом ранее|\bза (?:весь |последний )?год\b|"
                    r"с начала года|прошлого года|\bза \d+ месяц|\b\d{1,2}м\d{2}\b|по итогам года|к аналогичному периоду")
MONTH_YOY = "Изменение за месяц к тому же месяцу прошлого года: оценка в год — если так будет весь год"


def _period(quote: str, text: str, raw: str) -> str:
    """'short' — изменение за месяц или неделю; 'month_yoy' — месяц к тому же месяцу прошлого года; '' — за год или не указано.

    Заголовок часто пересказывает фразу текста короче («…в августе упали на 7%»), поэтому смотрим все
    предложения события с тем же числом."""
    same = re.compile(r"(?<![\d.,])" + re.escape(raw) + r"\s*%")
    parts = [_low(quote)] + [_low(s) for s in _sentences(text) if same.search(s)]
    short = any(SHORT.search(t) for t in parts)
    annual = any(ANNUAL.search(t) for t in parts)
    return "month_yoy" if short and annual else "short" if short else ""


# ---------- извлечение шоков из текста ----------
def _rate_in(t: str) -> dict:
    out: dict = {}
    m = re.search(r"\bс\s+" + PCT + r"(?:\s*годовых)?\s+до\s+" + PCT, t)
    if m:
        a, b = _f(m.group(1)), _f(m.group(2))
        return {"level": b, "delta": b - a, "end": m.end()}
    sign = -1 if re.search(DOWN, t) else 1 if re.search(UP, t) else 0
    bp = re.search(r"на\s+(\d+)\s*(?:б\.\s?п|базисн)", t)
    ppm = re.search(r"на\s+" + NUM + r"\s*(?:п\.\s?п|процентн\w* пункт)", t)
    hold = re.search(HOLD, t)
    if bp and sign:
        out |= {"delta": sign * int(bp.group(1)) / 100, "end": bp.end()}
    elif ppm and sign:
        out |= {"delta": sign * _f(ppm.group(1)), "end": ppm.end()}
    elif hold:
        out |= {"delta": 0.0, "end": hold.end()}
    lv = re.search(r"(?:\bдо|на уровне|уровне|составляет|составит|ставк\w*(?:\s+в)?)\s+" + PCT, t)
    if lv:
        out["level"] = _f(lv.group(1))
    return out


# Ставка Банка России, а не «ставка НДС», «ставки аренды», «ставка купона», «ставки по ипотеке» и не ставка ФРС
RATE_CTX = re.compile(r"ключев|\bцб\b|банк\w* росси|регулятор|центробанк")
FOREIGN_CB = re.compile(r"\bфрс\b|\bецб\b|федеральн\w* резерв|банк\w* (?:англии|японии|китая|кореи|канады)|народн\w* банк")
RUS_CB = re.compile(r"\bцб\b|банк\w* росси")
NOT_KEY = re.compile(r"ндс|налог|пошлин|утильсбор|акциз|аренд|купон|ипотек|вклад|депозит")


def _key_rate_sentence(t: str) -> bool:
    if "ставк" not in t or not RATE_CTX.search(t):
        return False
    if FOREIGN_CB.search(t) and not RUS_CB.search(t):
        return False
    for m in re.finditer(r"(\w+\s+)?ставк\w*((?:\s+\S+){0,3})", t):
        before = m.group(1) or ""
        if before.startswith("ключев") or not NOT_KEY.search(before + m.group(2)):
            return True
    return False


def rate_shock(text: str) -> dict | None:
    """Ключевая ставка: уровень, изменение в п.п. и фразы-основания (дословно из новости).

    Уровень и изменение могут стоять в разных предложениях («ЦБ сохранил ставку» + «…на уровне 14%»).
    modal — изменение взято из прогноза или условия; факт из другого предложения важнее прогноза."""
    found: dict = {}
    src: dict = {}  # из какого предложения взят уровень и изменение
    modal = False
    for s in _sentences(text):
        t = _low(s)
        if not _key_rate_sentence(t):
            continue
        r = _rate_in(t)
        here = _modal(t, r.pop("end", None))
        if "level" in r and not 2 <= r["level"] <= 40:
            r.pop("level")
        if "delta" in r and abs(r["delta"]) > 10:
            r.pop("delta")
        if "level" in r and "level" not in found:
            found["level"], src["level"] = r["level"], s
        # явное изменение («с 16% до 14%», «на 200 б.п.») важнее общего «сохранил» из заголовка, факт — важнее прогноза
        if "delta" in r and ("delta" not in found or (found["delta"] == 0 and r["delta"] and modal <= here)
                             or (modal and not here)):
            found["delta"], src["delta"], modal = r["delta"], s, here
        if "level" in found and found.get("delta") and not modal:
            break
    if not found:
        return None
    used = [s for s in _sentences(text) if s in src.values()]
    return found | {"quote": " ".join(s[:300] for s in used), "modal": modal, "used": used}


CURRENCY = r"доллар|юан|евро|дирхам|лир[аыу]\b|рупи|рубл"
# валюта названа как подлежащее (именительный или родительный: «рубль ослаб», «ослабление рубля»), а не «в рублях»
CUR_NAMED = r"\b(?:рубл[ья]|доллара?|юан[ья]|евро|дирхама?|лир[аы])\b"
CLAUSES = re.compile(r",|;|\s—\s|\s(?:и|а|но|после|на фоне|из-за|вслед за|благодаря|тогда как|при этом)\s")


def _fx_clause(c: str) -> dict | None:
    """Часть фразы о движении одной валюты: подлежащее, ослаб ли рубль, на сколько процентов."""
    if not ("курс" in c or re.search(CUR_NAMED, c)):
        return None
    # подлежащее — первая валюта, которая не сумма («95 рублей», «1,2 млрд рублей»)
    cur = next((m.group(1) for m in re.finditer(r"\b(" + CURRENCY + ")", c)
                if not re.search(r"(?:\d|млрд|млн|трлн|тыс\.?)\s*$", c[:m.start()])), None)
    if not cur:
        return None
    down, up = re.search(DOWN + r"|ослаб|обесцен|девальв", c), re.search(UP + r"|укрепи", c)
    if bool(down) == bool(up):
        return None
    ruble = cur == "рубл"
    if not ruble and re.search(r"\bк (?:доллар|евро|юан)", c):
        return None  # кросс-курс («евро подешевел к доллару») — не про рубль
    m = re.search(r"на\s+" + PCT, c)
    return {"ruble": ruble, "weaker": bool(down) if ruble else bool(up),
            "size": _f(m.group(1)) if m else None, "raw": m.group(1) if m else None, "end": m.end() if m else 0}


def fx_shock(text: str) -> dict | None:
    """Курс: на сколько процентов ослаб (+) или укрепился (−) рубль.

    Фраза делится на части по запятым и союзам: «Курс доллара вырос до 95 рублей, рубль ослаб на 10%» — обе части
    говорят об ослаблении рубля, процент берётся из части про рубль. Части с разным направлением — расчёта нет."""
    for s in _sentences(text):
        t = _low(s)
        if "ставк" in t:
            continue
        moves = [mv | {"at": t.find(c)} for c in CLAUSES.split(t) if (mv := _fx_clause(c))]
        if not moves or len({mv["weaker"] for mv in moves}) > 1:
            continue
        sized = [mv for mv in moves if mv["size"]]
        mv = next((x for x in sized if x["ruble"]), sized[0] if sized else None)
        if not mv or not 0 < mv["size"] <= 50:
            continue
        return {"change_pct": mv["size"] if mv["weaker"] else -mv["size"], "quote": s[:300], "raw": mv["raw"],
                "modal": _modal(t, mv["at"] + mv["end"]), "used": [s]}
    return None


DUTY_KINDS = (("налог на прибыль", r"налог\w* на прибыль"), ("утильсбор", r"утильсбор|утилизационн"), ("пошлина", r"пошлин"))


def duty_shock(text: str) -> dict | None:
    """Пошлина (экспорт и импорт — раздельно), утильсбор, налог на прибыль: новая ставка или изменение.

    - «с 25% до 27%» и «на 2 п.п.» — изменение ставки в процентных пунктах базы (delta);
    - «повышает утильсбор на 20%» — сбор растёт на 20% от текущего (rel), а не на 20% стоимости ввоза;
    - «введёт пошлину 15%» — новая пошлина: её ставка целиком — новый расход (delta);
    - «повысит пошлину до 15%» — известен только уровень, прежняя ставка неизвестна: расчёта нет."""
    for s in _sentences(text):
        t = _low(s)
        kind, km = next(((k, m) for k, rx in DUTY_KINDS if (m := re.search(rx, t))), (None, None))
        if not kind:
            continue
        if kind == "пошлина":
            kind = ("экспортная пошлина" if re.search(r"экспортн\w* пошлин|пошлин\w* на (?:экспорт|вывоз)|вывозн", t) else
                    "импортная пошлина" if re.search(r"импортн\w* пошлин|пошлин\w* на (?:импорт|ввоз)|ввозн", t) else kind)
        out = {"kind": kind, "quote": s[:300], "used": [s]}
        down = re.search(DOWN + r"|отмен|обнул", t)
        up = re.search(UP + r"|индексир", t)
        new = re.search(r"введ|ввод|установ", t)
        sign = 1 if (up or new) and not down else -1 if down and not (up or new) else 0
        m = re.search(r"\bс\s+" + PCT + r"\s+до\s+" + PCT, t)
        if m:
            return out | {"from": _f(m.group(1)), "to": _f(m.group(2)), "modal": _modal(t, m.end())}
        tail, at = t[km.start():], km.start()
        m = re.search(r"\bна\s+" + NUM + r"\s*(?:п\.\s?п|процентн\w* пункт)", tail)
        if m and sign:
            return out | {"delta": sign * _f(m.group(1)), "modal": _modal(t, at + m.end())}
        if kind == "налог на прибыль":
            continue  # «налог повысят до 25%» — только уровень
        # «повышает утильсбор на легковые автомобили на 20%»: процент — рядом со словом сбора, не про продажи или цены
        m = re.search(r"^[^.;]{0,80}?\bна\s+" + PCT, tail)
        if m and sign and not new and not re.search(r"продаж|спрос|цен|выпуск|импорт\b|ввоз\b", m.group()[len(km.group()):]):
            v = _f(m.group(1))
            if 0 < v <= 100:
                return out | {"rel": sign * v, "modal": _modal(t, at + m.end())}
        # новая пошлина: «введёт пошлину 15%», «установит пошлину в размере 15%»
        m = re.search(r"(?:в размере|ставк\w*|на уровне|\bдо)\s+" + PCT, tail) or re.search(r"пошлин\w*\s+" + PCT, tail)
        if m and new and kind != "утильсбор":
            v = _f(m.group(1))
            if 0 < v <= 100:
                return out | {"delta": v, "new": True, "modal": _modal(t, at + m.end())}
    return None


# Рынок за рубежом: «продажи недвижимости в Дубае упали» не переносим на российскую компанию,
# если страны нет в её ВЭД. Ключ — название страны, как в профиле (foreign_trade.countries).
FOREIGN = {"ОАЭ": r"оаэ|дуба|абу-даби|эмират", "КИТАЙ": r"кита|кнр|пекин|шанха", "США": r"\bсша\b|американ",
           "ТУРЦИЯ": r"турци|турецк|стамбул", "ИНДИЯ": r"инди[ия]|индийск", "ГЕРМАНИЯ": r"германи|немецк",
           "ЕВРОСОЮЗ": r"\bес\b|евросоюз|европ", "ЯПОНИЯ": r"япони|японск", "КАЗАХСТАН": r"казахстан",
           "БЕЛАРУСЬ": r"беларус|белорус", "ВЕЛИКОБРИТАНИЯ": r"британи|великобритан|лондон", "УКРАИНА": r"украин",
           "ФРАНЦИЯ": r"франци|французск|париж", "ИТАЛИЯ": r"итали|итальянск", "ШВЕЙЦАРИЯ": r"швейцари",
           "КАНАДА": r"канад", "ИРАН": r"\bиран", "ИЗРАИЛЬ": r"израил", "КОРЕЯ": r"коре[ия]|корейск"}

# Показатели, которые не спрос и не цены отрасли: запасы, ВВП, долги, прибыль, капитализация
NOT_MARKET = r"запас|резерв|ввп|задолженност|просроч|прибыл|капитализац|убыт|инфляц"
PRICE = r"цен|подорож|подешев|стоимост|тариф|котировк"
# «импортные автомобили» и «экспортная пошлина» — не объём импорта или экспорта
VOLUME = r"продаж|спрос|выпуск|производств|отгрузк|потреблен|ввод|выдач|импорт(?!н)|экспорт(?!н)|перевоз|оборот"
# Биржевая цена товара — мировой ориентир для всех, а не рынок другой страны: «медь на LME», «бензин на бирже»
BENCH = r"\blme\b|\bлбм\b|бирж|\bcomex\b"
# Розничная цена — цена продажи для АЗС и магазина; биржевая и оптовая — цена закупки
RETAIL = r"розничн|на азс|на заправк|мелкоопт"
WHOLESALE = r"бирж|оптов|закупочн|\bнпз\b|\blme\b"
# Слова отрасли, которых нет в общем справочнике, но которые нужны для закупок: «арматура» — металл
EXTRA_STEMS = {"24": r"арматур"}


def sector_shocks(text: str, skip: list[str] | tuple = ()) -> list[dict]:
    """«Продажи легковых автомобилей упали на 12%» → {sector: 29/45, kind: volume, change: −12}.

    skip — предложения, уже давшие шок ставки, курса или пошлины: «утильсбор на импортные автомобили
    повысили на 30%» — это расход на сбор, а не рост продаж автомобилей на 30%."""
    out = []
    for s in _sentences(text):
        if s in skip:
            continue
        t = _low(s)
        if ("ключев" in t and "ставк" in t) or _key_rate_sentence(t) or "курс" in t or re.search(NOT_MARKET, t):
            continue
        m = re.search(r"(" + DOWN + "|" + UP + r")\w*[^.%;]{0,80}?на\s+" + PCT, t) or \
            re.search(r"на\s+" + PCT + r"[^.%;]{0,40}?(" + DOWN + "|" + UP + r")", t)
        if not m:
            continue
        verb, val = (m.group(1), m.group(2)) if not m.group(1)[0].isdigit() else (m.group(2), m.group(1))
        sectors = sorted(set(sectors_in(s)) | {c for c, rx in EXTRA_STEMS.items() if re.search(rx, t)})
        bench = bool(re.search(BENCH, t)) and "индекс" not in t and bool(set(sectors) - {"66"})
        if bench:
            sectors = [x for x in sectors if x != "66"]  # биржа здесь — площадка цены, а не отрасль брокеров
        if not sectors:
            continue
        v = _f(val)
        if not 0 < v <= 80:
            continue
        change = -v if re.match(DOWN, verb) else v
        kind = "price" if re.search(PRICE, t) or bench else "volume" if re.search(VOLUME, t) else None
        if kind is None:  # «просроченная задолженность выросла на 43%» — не спрос и не цены
            continue
        if any(x["quote"] == s[:300] for x in out):
            continue
        out.append({"sectors": sectors, "kind": kind, "change": change, "raw": val, "quote": s[:300],
                    "abroad": [] if bench else [k for k, v in FOREIGN.items() if re.search(v, t)],
                    "modal": _modal(t, m.end()), "sentence": s})
    return out


# Что компания закупает, а не продаёт: цена такого товара — её расход. Ключ — префикс полного кода ОКВЭД.
# (отрасль товара, основа слова — если отрасль товара совпадает с отраслью компании, название, оно же в
# родительном падеже, перепродаётся ли товар как есть — тогда розничная цена этого товара — цена продажи)
INPUTS = {
    "47.30": [("19", None, "топливо", "топлива", True)],
    "46.71": [("19", None, "топливо", "топлива", True)],
    "17.2": [("17", r"целлюлоз", "целлюлоза", "целлюлозы", False)],
    "25": [("24", None, "металл", "металла", False)],
    "41": [("24", None, "металл", "металла", False)],
}
# Продажи машин (29) — спрос и для автодилеров и автосервисов (45): «продажи авто упали» касается их покупателей
AUTO_CHAIN = {"29": ["45"]}


def _chain(sectors) -> list[str]:
    return list(dict.fromkeys([*sectors, *(x for s in sectors for x in AUTO_CHAIN.get(s, []))]))


def _input_goods(p: dict, sh: dict) -> dict | None:
    t = _low(sh["quote"])
    for prefix, items in INPUTS.items():
        if p["okved"].startswith(prefix):
            for sector, kw, name, gen, resale in items:
                if sector in sh["sectors"] and (kw is None or re.search(kw, t)):
                    return {"name": name, "gen": gen, "resale": resale}
    return None


def _price_moves(sentence: str) -> list[tuple[str, float, str]]:
    """«Биржевые цены на бензин выросли на 8%, розничные — на 1,2%» → [('in', 8, '8'), ('out', 1.2, '1,2')]."""
    out, verb = [], None
    for c in re.split(r",|;|\s(?:а|но|тогда как|при этом|в то время как)\s", _low(sentence)):
        v = re.search("(" + DOWN + "|" + UP + ")", c)
        verb = v.group(1) if v else verb  # «розничные — на 1,2%»: глагол из предыдущей части
        m = re.search(r"на\s+" + PCT, c)
        if m and verb:
            out.append(("out" if re.search(RETAIL, c) else "in", (-1 if re.match(DOWN, verb) else 1) * _f(m.group(1)), m.group(1)))
    return out


def _flows(p: dict, side: str) -> dict[str, float]:
    """Доли отраслей в платежах через банк; при малой выборке (меньше 10 плательщиков) — пусто: доли случайны."""
    s = p[side]
    return {b["okved2"]: b["share"] for b in s["by_industry"]} if s["n"] >= FLOW_N_MIN else {}


def _count(n: int, one: str, many: str) -> str:
    return one if n % 10 == 1 and n % 100 != 11 else many


# ---------- расчёты ----------
def profile_text(p: dict) -> str:
    """Текст «источника» с данными профиля: по нему валидатор сверяет числа входных данных, его же читает модель.

    Подписи — чтобы модель не путала смысл чисел: доли группы — это доли юрлиц, а не платежей; n — плательщики
    в выборке банка, а не «ключевые клиенты»; при малой выборке прямо сказано, что оценки по долям нет."""
    goods = [p["okved2"], *related_sectors(p["okved"])]
    lines = [f"ОКВЭД компании: {p['okved']}; отрасль компании и её товара: " + ", ".join(label(s) for s in dict.fromkeys(goods)),
             f"Вид деятельности: {p['industry']}",
             f"Выручка компании за год: {num(p['revenue_rub'] / BN)} млрд ₽",
             f"Продукты банка: {', '.join(p['bank_products']) or 'нет'}",
             "ВЭД: " + (", ".join(country_name(c) for c in p["foreign_trade"]["countries"]) or "нет")]
    g = p["group"]
    big = [x for x in g["industries"] if x["share"] >= 0.1]
    if g["size"] > 1 and big:
        lines.append(f"Группа: {g['size']} юрлиц; доля юрлиц группы по отраслям (не доля платежей): "
                     + ", ".join(f"{label(x['okved2'])} — {pct(x['share'])}" for x in big))
    for side, name, one, many, flow in (("customers", "Покупатели", "плательщика", "плательщиков", "поступлений"),
                                        ("suppliers", "Поставщики", "получателя платежей", "получателей платежей", "платежей")):
        s = p[side]
        if s["n"] < FLOW_N_MIN:
            lines.append(f"{name}: " + (f"по платежам через банк видно лишь {s['n']} {_count(s['n'], one, many)} — оценку не даём"
                                        if s["n"] else "в платёжной выборке банка их нет — оценку не даём"))
            continue
        lines.append(f"{name} — {many} в выборке банка: {s['n']}; отрасль известна для {pct(s['known_industry_share'])} {flow}")
        for b in s["by_industry"]:
            if b["share"] >= FLOW_MIN:
                lines.append(f"{label(b['okved2'])}: {round(b['share'] * 100)}% {flow} через банк")
    return "\n".join(lines)


def _calc(title, value, formula, inputs, assumptions, sign, scale=None, strength="medium") -> dict:
    c = {"title": title, "value": value, "formula": formula, "inputs": inputs,
         "assumptions": assumptions, "sign": sign, "strength": strength}
    if scale:
        c["scale"] = scale
    return c


# Расходы: рост — риск, снижение — возможность. Остальное (выручка, доход, маржа): рост — возможность, падение — риск.
# «Валютные расчёты» считаются как закупки: направление ВЭД в данных не указано, оговорка — в допущениях.
COST_TITLES = ("Стоимость долга", "Закупки", "Утильсбор", "Пошлина", "Налог", "Валютные расчёты")


def _sign(title: str, eff: float) -> str:
    if not eff:
        return "neutral"
    return "risk" if (eff > 0) == title.startswith(COST_TITLES) else "opportunity"


def compute(event: dict, p: dict, news_src: str = "news", profile_src: str = "profile") -> list[dict]:
    """Расчёты эффекта. src во входных данных — метки news_src / profile_src (их подставляет вызывающий)."""
    text = (event.get("title", "") + ". " + event.get("text", "")).strip()
    products = set(p["bank_products"])
    has_debt = bool({"кредит", "лизинг"} & products)
    ved = p["foreign_trade"]["active"]
    countries = ", ".join(country_name(c) for c in p["foreign_trade"]["countries"][:4])
    R = p["revenue_rub"]
    fact = lambda t, src=news_src: {"t": t, "p": "fact", "src": src}  # noqa: E731
    calcs: list[dict] = []
    used: list[str] = []  # предложения, уже давшие шок ставки, курса или пошлины

    rs = rate_shock(text)
    if rs:
        used += rs["used"]
        news_in = fact(rs["quote"])
        # расчёт — только при изменении ставки: «расходы при текущей ставке» руководителю ничего нового не говорят
        d = rs.get("delta")
        cond = f"ставку {'снизят' if d and d < 0 else 'повысят'} на {num(abs(d or 0), 2)} п.п."
        rate_calcs = []
        if has_debt and d:
            prods = ", ".join(x for x in ("кредит", "лизинг") if x in products)
            eff = d / 100 * BN
            rate_calcs.append(_calc(
                "Стоимость долга", f"{_plus(eff)}{rub(eff)} в год на каждый 1 млрд ₽ долга с плавающей ставкой",
                f"1 млрд ₽ × {pp(d)}", [news_in, fact(f"Продукты банка: {prods}", profile_src)],
                ["Ставка по кредиту привязана к ключевой и меняется на ту же величину",
                 "Для долга с фиксированной ставкой эффект появится при рефинансировании"],
                _sign("Стоимость долга", eff),
                {"label": "Ваш долг с плавающей ставкой, млрд ₽", "per_bn": eff, "period": "в год"}))
        if "депозиты" in products and d:
            eff = d / 100 * BN
            rate_calcs.append(_calc(
                "Доход по депозитам", f"{_plus(eff)}{rub(eff)} в год на каждый 1 млрд ₽ размещённых средств",
                f"1 млрд ₽ × {pp(d)}", [news_in, fact("Продукты банка: депозиты", profile_src)],
                ["Ставки по новым депозитам следуют за ключевой", "Действующие срочные депозиты доработают по старой ставке"],
                _sign("Доход по депозитам", eff),
                {"label": "Свободные средства на депозитах, млрд ₽", "per_bn": eff, "period": "в год"}))
        calcs += [_scenario(c, cond) for c in rate_calcs] if rs["modal"] else rate_calcs

    fx = fx_shock(text)
    if fx:
        used += fx["used"]
    if fx and ved:
        ch = fx["change_pct"]
        eff = ch / 100 * BN
        c = _calc(
            "Валютные расчёты", f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ валютных закупок "
            f"({'дороже' if ch > 0 else 'дешевле'} в рублях)",
            f"1 млрд ₽ × {_pc(ch)}", [fact(fx["quote"]), fact(f"ВЭД: {countries}", profile_src)],
            ["Направление ВЭД (импорт или экспорт) в данных не указано: для импортёра это расход, "
             "для экспортёра — такая же прибавка к рублёвой выручке",
             "Без учёта хеджирования и валютной оговорки в контрактах"],
            _sign("Валютные расчёты", eff),
            {"label": "Ваши валютные закупки за год, млрд ₽", "per_bn": eff, "period": "за год"})
        calcs.append(_scenario(c, f"рубль {'ослабнет' if ch > 0 else 'укрепится'} на {num(abs(ch), 2)}%") if fx["modal"] else c)

    goods = {p["okved2"], *related_sectors(p["okved"])}  # что компания продаёт: своя отрасль и отрасль её товара
    du = duty_shock(text)
    if du:
        used += du["used"]
        kind = du["kind"]
        relevant = (kind == "налог на прибыль" or ved or
                    (kind in ("утильсбор", "пошлина", "импортная пошлина") and {"29", "45"} & goods))
        rel = du.get("rel")
        delta = du.get("delta", du.get("to", 0) - du.get("from", 0))
        if relevant and (rel or delta):
            title = {"налог на прибыль": "Налог на прибыль", "утильсбор": "Утильсбор", "пошлина": "Пошлина",
                     "экспортная пошлина": "Пошлина на экспорт", "импортная пошлина": "Пошлина на импорт"}[kind]
            if rel:  # сбор или пошлина растут на rel% от текущих платежей
                base = "утильсбора, который вы платите сейчас" if kind == "утильсбор" else "пошлин, которые вы платите сейчас"
                label_ = "Ваши платежи утильсбора за год, млрд ₽" if kind == "утильсбор" else "Ваши платежи пошлин за год, млрд ₽"
                eff, formula = rel / 100 * BN, f"1 млрд ₽ × {_pc(rel)}"
                notes = [f"Сбор растёт на {num(abs(rel), 2)}% от нынешнего, а не на {num(abs(rel), 2)}% стоимости товара",
                         "Если технику ввозит поставщик, рост сбора придёт к вам через его цены" if kind == "утильсбор"
                         else "Переложение в цены покупателям не учтено"]
            else:
                base = {"налог на прибыль": "прибыли до налогообложения",
                        "экспортная пошлина": "таможенной стоимости вывоза"}.get(kind, "таможенной стоимости ввоза")
                label_ = f"Ваша база ({base}), млрд ₽"
                eff = delta / 100 * BN
                formula = f"1 млрд ₽ × {pp(delta) if 'from' in du or not du.get('new') else num(delta, 2) + '%'}"
                notes = ["Ставка применяется ко всей базе; льготы и переходные периоды не учтены",
                         "Переложение в цены покупателям не учтено"]
                if kind == "экспортная пошлина":
                    notes.insert(0, "Направление ВЭД в данных не указано: это расход, только если вы вывозите этот товар")
                if du.get("new"):
                    notes.insert(0, "Пошлина вводится впервые: её ставка целиком — новый расход")
            c = _calc(title, f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ {base}", formula, [fact(du["quote"])], notes,
                      _sign(title, eff), {"label": label_, "per_bn": eff, "period": "за год"})
            calcs.append(_scenario(c, "решение примут") if du.get("modal") else c)

    cust, supp = _flows(p, "customers"), _flows(p, "suppliers")
    trade = p["okved2"] in ("46", "47")
    shocks = [sh for sh in sector_shocks(text, used)
              # рынок другой страны, с которой компания не работает
              if not any(c not in p["foreign_trade"]["countries"] and not (c == "ЕВРОСОЮЗ" and ved) for c in sh["abroad"])]
    flow_done = False   # уже есть оценка выручки или закупок — «выручка под влиянием» не нужна
    blocked = False     # шок был, но за месяц или неделю — на год не переносим и масштаб не показываем

    # 1. Цена того, что компания закупает: закупки, а если двигаются и закупочные, и розничные цены — маржа
    moves: dict[str, dict] = {}
    rest = []
    for sh in shocks:
        g = _input_goods(p, sh)
        if not g:
            rest.append(sh)
            continue
        t = _low(sh["sentence"])
        out_side = g["resale"] and re.search(RETAIL, t) and not re.search(WHOLESALE, t)
        if out_side:
            rest.append(sh | {"resale": True})  # розничная цена товара — цена продажи, считается как выручка
            continue
        if sh["kind"] != "price":
            continue  # объём производства сырья — не спрос на продукцию компании: расчёта нет
        mv = moves.setdefault(g["name"], {"g": g})
        pair = _price_moves(sh["sentence"]) if g["resale"] else []
        if {"in", "out"} <= {x[0] for x in pair}:  # обе цены в одной фразе
            mv.setdefault("in", (next(x for x in pair if x[0] == "in")[1], sh))
            mv.setdefault("out", (next(x for x in pair if x[0] == "out")[1], sh))
        else:
            mv.setdefault("in", (sh["change"], sh))
    for sh in [x for x in rest if x.get("resale")]:
        g = _input_goods(p, sh)
        if g and g["name"] in moves and "out" not in moves[g["name"]] and sh["kind"] == "price":
            moves[g["name"]]["out"] = (sh["change"], sh)
            rest.remove(sh)
    for name, mv in moves.items():
        g, (ch_in, sh_in) = mv["g"], mv["in"]
        quotes = list(dict.fromkeys([sh_in["quote"], *([mv["out"][1]["quote"]] if "out" in mv else [])]))
        modal = sh_in["modal"] or ("out" in mv and mv["out"][1]["modal"])
        if "out" in mv:
            ch_out = mv["out"][0]
            d = ch_out - ch_in
            eff = d / 100 * BN
            c = _calc(
                f"Маржа: {name}", f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ закупок {g['gen']}",
                f"1 млрд ₽ × ({_pc(ch_out)} − {_pc(ch_in)})", [fact(q) for q in quotes],
                [f"Закупочные цены меняются на {_pc(ch_in)}, цены продажи — на {_pc(ch_out)}: разница ложится на маржу",
                 "Объём продаж не меняется; наценка не учтена — выручка с того же объёма принята равной закупкам"],
                _sign("Маржа", eff), {"label": f"Ваши закупки {g['gen']} за год, млрд ₽", "per_bn": eff, "period": "за год"})
        else:
            eff = ch_in / 100 * BN
            c = _calc(
                f"Закупки: {name}", f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ закупок {g['gen']}",
                f"1 млрд ₽ × {_pc(ch_in)}", [fact(quotes[0]), fact(f"Вид деятельности: {p['industry']}", profile_src)],
                [f"Для вас {name} — закупка: её цена меняется так же, как в новости",
                 "Условия действующих контрактов и переложение в цены продажи не учтены"],
                _sign("Закупки", eff), {"label": f"Ваши закупки {g['gen']} за год, млрд ₽", "per_bn": eff, "period": "за год"})
        calcs.append(_scenario(c, "так и будет") if modal else c)
        flow_done = True

    # 2. Спрос и цены в отраслях покупателей, поставщиков и своей. Одна оценка выручки на событие:
    #    по доле покупателей (точнее), вторым — «если ваши продажи изменятся как у рынка» (если рынок — ваш)
    seen: set[str] = set()
    revenue_done = False
    for sh in rest:
        ch, kind = sh["change"], sh["kind"]
        sectors = [s for s in _chain(sh["sectors"]) if s not in seen]
        what = "цены" if kind == "price" else "спрос/объёмы"
        period = _period(sh["quote"], text, sh["raw"])
        can_revenue = not revenue_done and period != "short"
        blocked = blocked or (period == "short" and not revenue_done)
        cond = (f"цены {'вырастут' if ch > 0 else 'снизятся'}" if kind == "price" else
                f"рынок {'вырастет' if ch > 0 else 'сократится'}") + f" на {num(abs(ch), 2)}%"
        out: list[dict] = []
        sh_news = fact(sh["quote"])
        own = [s for s in sectors if s in goods]
        # цена своего товара — выручка, а не закупки у поставщиков той же отрасли («цены на новостройки» для застройщика)
        own_price = kind == "price" and (p["okved2"] in own or sh.get("resale"))
        # закупки у поставщиков отрасли: меняется уровень цены — период не важен
        if kind == "price" and not own_price:
            for s in sectors:
                share = supp.get(s, 0)
                if share >= FLOW_MIN and s not in goods | {p["okved2"]} or share >= FLOW_MIN and not own:
                    seen.add(s)
                    eff = ch / 100 * BN
                    out.append(_calc(
                        f"Закупки у поставщиков — {who(s)}", f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ закупок у отрасли",
                        f"1 млрд ₽ × {_pc(ch)}",
                        [sh_news, fact(f"{round(share * 100)}% платежей через банк — {who(s)}", profile_src)],
                        ["Цены поставщиков меняются вслед за ценами отрасли", "Условия действующих контрактов не учтены"],
                        _sign("Закупки", eff),
                        {"label": f"Ваши закупки за год ({who(s)}), млрд ₽", "per_bn": eff, "period": "за год"}))
        if can_revenue and kind == "volume":
            best = max(sectors, key=lambda s: cust.get(s, 0), default=None)
            share = cust.get(best, 0) if best else 0
            if share >= FLOW_MIN:
                seen.add(best)
                revenue_done = True
                eff = R * share * ch / 100
                out.append(_calc(
                    f"Выручка от покупателей — {who(best)}", f"{_plus(eff)}{rub(eff)} в год",
                    f"{num(R / BN)} млрд ₽ × {round(share * 100)}% × {_pc(ch)}",
                    [sh_news, fact(f"Выручка компании за год: {num(R / BN)} млрд ₽", profile_src),
                     fact(f"{round(share * 100)}% поступлений через банк — {who(best)}", profile_src)],
                    [f"Выручка от этих покупателей меняется так же, как {what} в их отрасли (эластичность 1)",
                     f"Доля — по платежам через банк; отрасль плательщика известна для {pct(p['customers']['known_industry_share'])} "
                     "поступлений, поэтому это оценка по известной части"] + ([MONTH_YOY] if period == "month_yoy" else []),
                    _sign("Выручка", eff), strength="medium"))
        has_customer = any(c["title"].startswith("Выручка от покупателей") for c in out)
        if (can_revenue or has_customer) and own and (has_customer or not out or own_price):
            for s in own:
                seen.add(s)
            revenue_done = True
            eff = R * ch / 100
            notes = [f"{what.capitalize()} компании меняются так же, как в отрасли в целом",
                     "Компания может расти или падать быстрее рынка; это ориентир масштаба, а не прогноз"]
            if has_customer:
                notes.insert(0, "Вариант, если от рынка зависит вся выручка, а не только поступления от покупателей этой отрасли")
            if kind == "price" and trade:
                notes.append("В торговле закупочные цены обычно меняются вместе с ценами продажи: прибыль меняется меньше выручки")
            if period == "month_yoy":
                notes.append(MONTH_YOY)
            out.append(_calc(
                f"Выручка, если ваши {'цены' if kind == 'price' else 'продажи'} изменятся как у рынка",
                f"{_plus(eff)}{rub(eff)} в год", f"{num(R / BN)} млрд ₽ × {_pc(ch)}",
                [sh_news, fact(f"Выручка компании за год: {num(R / BN)} млрд ₽", profile_src)],
                notes, _sign("Выручка", eff), strength="low"))
        if out:
            flow_done = True
            blocked = False
        calcs += [_scenario(c, cond) for c in out] if sh["modal"] else out

    # Без процента в новости: сколько выручки связано с отраслью события (масштаб того, что под влиянием)
    head = event.get("title", "") + " " + event.get("text", "")[:600]
    abroad = [c for c, rx in FOREIGN.items() if re.search(rx, _low(head))]
    if not flow_done and not blocked and not (abroad and not set(abroad) & set(p["foreign_trade"]["countries"])):
        best = max(_chain(sectors_in(head)), key=lambda s: cust.get(s, 0), default=None)
        share = cust.get(best, 0) if best else 0
        if share >= EXPOSURE_MIN:
            calcs.append(_calc(
                f"Выручка под влиянием: покупатели — {who(best)}", f"≈ {rub(R * share)} в год",
                f"{num(R / BN)} млрд ₽ × {round(share * 100)}%",
                [fact(f"Выручка компании за год: {num(R / BN)} млрд ₽", profile_src),
                 fact(f"{round(share * 100)}% поступлений через банк — {who(best)}", profile_src)],
                ["Это масштаб выручки, связанной с отраслью события, а не прогноз потерь: размер эффекта в новости не указан",
                 f"Доля — по платежам через банк; отрасль плательщика известна для {pct(p['customers']['known_industry_share'])} поступлений"],
                "neutral", strength="medium"))
    return calcs[:4]


PROFILE_SOURCE = "Профиль компании: данные банка (выручка, продукты, платёжный поток)"


def attach(card: dict, p: dict, today: str) -> dict:
    """Добавляет расчёт в готовую карточку (примеры команды, карточка без модели).

    Текст шока берётся из фактов «что произошло»; входные данные ссылаются на источник того факта,
    из которого взята фраза, а данные профиля — на добавленный источник вида stat."""
    what = [s for s in card["blocks"]["what"] if s.get("p") == "fact"]
    calcs = compute({"title": card["title"], "text": " ".join(s["t"] for s in what)}, p)
    if not calcs or card["blocks"].get("calc"):
        return card
    card = {**card, "sources": list(card["sources"]), "blocks": dict(card["blocks"])}
    if any(i["src"] == "profile" for c in calcs for i in c["inputs"]):
        card["sources"].append({"name": PROFILE_SOURCE, "kind": "stat", "date": today, "url": ""})
    prof = len(card["sources"]) - 1

    from .contract.validator import numbers_in

    def src_of(quote: str) -> int | None:
        for s in what:
            if any(q.strip(" .") and q.strip(" .") in s["t"] for q in _sentences(quote)):
                return s.get("src")
        # фраза из заголовка: ссылаемся на факт, где есть те же числа (заголовок пересказывает источник)
        nums = numbers_in(quote)
        return next((s.get("src") for s in what if nums and nums <= numbers_in(s["t"])), None)

    out = []
    for c in calcs:
        inputs = []
        for i in c["inputs"]:
            src = prof if i["src"] == "profile" else src_of(i["t"])
            if src is None:
                break  # шок не подтверждается фактом с источником — такой расчёт не показываем
            inputs.append({**i, "src": src})
        else:
            out.append(c | {"inputs": inputs})
    if out:
        card["blocks"]["calc"] = out
    return card
