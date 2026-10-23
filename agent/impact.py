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


DOWN = r"сниз|понизи|снижен|смягч|уменьш|сократ|сокращ|упал|упад|упаст|подешев|обвал|рухн|падени"
UP = r"повыс|повыш|поднял|подня|повышен|ужесточ|увелич|вырос|вырас|рост|подорож|ускор|подскоч"
HOLD = r"сохрани|оставил|без изменени|не стал менять|паузу"

# Прогноз, ожидание, предложение или условие — не свершившийся факт: расчёт становится сценарием «Если …»
MODAL = re.compile(r"(?<![а-яё])ожида|прогноз|\bмо(?:жет|гут)\b|\bесли\b|вероятн|планиру|аналитик|предлага|предложил|рассматрива|"
                   r"не исключ|по мнению|\bжд[её]т\b|\bждут\b|закладыва|допуска|консенсус|по оценк")
# «ожида» — с начала слова: «неожиданный рост» — факт.
# Обороты, где те же слова факт прогнозом не делают: «как и ожидалось», «сильнее ожиданий»,
# «вариант снижения не рассматривался», «ухудшил прогноз по инфляции», «по данным аналитиков»
NOT_MODAL = re.compile(
    r"(?:как и|как|чем) ожидал\w*|ожидаемо\b|"
    r"(?:вопреки|выше|ниже|сильнее|слабее|в соответствии с|в линии с|в рамках)\s+(?:ожидани|прогноз)\w*"
    r"(?:\s+(?:аналитик|рынк|экономист)\w*)?|не рассматрива\w*|"
    r"(?:ухудшил|улучшил|понизил|повысил|сохранил|пересмотрел|изменил|снизил|обновил)\w*\s+(?:\w+\s+)?прогноз\w*|"
    r"по (?:данным|подсчетам) аналитик\w*")
# «…, ожидают аналитики», «…, считают экономисты» после числа — тоже прогноз; «…, отметив, что ожидает…» — нет
MODAL_AFTER = re.compile(r",\s*(?:—\s*)?(?:ожида|прогнозир|счита|полага|предполага|допуска|не исключа)\w*\s|по мнению|по оценк|по прогноз")
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
    """Расчёт по прогнозу из новости: «Если ставку снизят на 0,5 п.п., стоимость долга», без знака, надёжность низкая.

    Условие отделено запятой, а не двоеточием: в главном выводе за названием расчёта идёт «: сумма»."""
    t = c["title"].replace(": ", " — ")
    # без двойного условия: «Если рынок вырастет на 3%, выручка при продажах как у рынка»
    t = re.sub(r"^Выручка, если ваши (цены|продажи) изменятся как у рынка$",
               lambda m: f"выручка при {'ценах' if m.group(1) == 'цены' else 'продажах'} как у рынка", t)
    return c | {"title": f"Если {cond}, {t[:1].lower() + t[1:]}", "sign": "neutral", "strength": "low",
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
NOT_KEY = re.compile(r"ндс|налог|пошлин|утильсбор|акциз|аренд|купон|ипотек|вклад|депозит|доходност|офз|облигац|кредит|резерв")
DECISION = re.compile(r"сниз|пониз|повыс|подня")  # решение по ставке, а не упоминание или ожидание


def _key_mention(t: str) -> bool:
    """Есть «ставка», не уточнённая как налог, аренда, ипотека, купон: «ключевую ставку», «снизил ставку»."""
    for m in re.finditer(r"(\w+\s+)?ставк\w*((?:\s+\S+){0,3})", t):
        before = m.group(1) or ""
        if before.startswith("ключев") or not NOT_KEY.search(before + m.group(2)):
            return True
    return False


def _key_rate_sentence(t: str) -> bool:
    if "ставк" not in t or not RATE_CTX.search(t):
        return False
    if FOREIGN_CB.search(t) and not RUS_CB.search(t):
        return False
    return _key_mention(t)


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
        # части фразы о других ставках не дают изменения ключевой: «…снизил ключевую ставку до 15%, ставки по
        # ипотеке снизятся на 2 п.п.» — 2 п.п. про ипотеку
        parts = CLAUSES.split(t)
        if any("ставк" in c and not _key_mention(c) for c in parts):
            t = ", ".join(c.strip() for c in parts if "ставк" not in c or _key_mention(c))
        r = _rate_in(t)
        here = _modal(t, r.pop("end", None))
        if "level" in r and not 2 <= r["level"] <= 40:
            r.pop("level")
        if "delta" in r and abs(r["delta"]) > 10:
            r.pop("delta")
        if "level" in r and "level" not in found:
            found["level"], src["level"] = r["level"], s
        # явное изменение («с 16% до 14%», «на 200 б.п.») важнее общего «сохранил» из заголовка, факт — важнее прогноза
        if "delta" in r and ("delta" not in found
                             or (found["delta"] == 0 and r["delta"] and not here and DECISION.search(t))
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
# запятая внутри числа («на 4,5%») — не граница части фразы
CLAUSES = re.compile(r",(?!\d)|;|\s—\s|\s(?:и|а|но|после|на фоне|из-за|вслед за|благодаря|тогда как|при этом)\s")


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
    говорят об ослаблении рубля, процент берётся из части про рубль. Части расходятся по направлению — верим только
    части про рубль; нет её или и она противоречива — расчёта нет."""
    for s in _sentences(text):
        t = _low(s)
        if "ставк" in t:
            continue
        moves = [mv | {"at": t.find(c)} for c in CLAUSES.split(t) if (mv := _fx_clause(c))]
        if len({mv["weaker"] for mv in moves}) > 1:
            moves = [mv for mv in moves if mv["ruble"]]
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
    - «отменит пошлину на автозапчасти 5%» — ставка целиком перестаёт быть расходом (delta < 0);
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
        cancel = re.search(r"отмен|обнул", t)
        down = re.search(DOWN, t) or cancel
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
        # новая или отменённая пошлина: «введёт пошлину 15%», «установит пошлину в размере 15%»,
        # «отменит пошлину на автозапчасти 5%» (но не «пошлина на ввоз, из-за которой цены вырастут на 15%»)
        m = (re.search(r"(?:в размере|ставк\w*|на уровне|\bдо)\s+" + PCT, tail) or
             re.search(r"пошлин\w*(?:\s+на\s+[^.;%\d,]{1,40}?)?(?<!на)\s+" + PCT, tail))
        if m and kind != "утильсбор" and (new or cancel) and not (new and cancel):
            v = _f(m.group(1))
            if 0 < v <= 100:
                return out | {"delta": v if new else -v, "new" if new else "cancel": True, "modal": _modal(t, at + m.end())}
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
PRICE = r"(?<![а-яё])цен(?!тр)|подорож|подешев|стоимост|тариф|котировк"  # не «оценка», «сценарий», «Центральная Азия»
# «импортные автомобили» и «экспортная пошлина» — не объём импорта или экспорта
VOLUME = r"продаж|спрос|выпуск|производств|отгрузк|потреблен|ввод|выдач|импорт(?!н)|экспорт(?!н)|перевоз|оборот"
# Биржевая цена товара — мировой ориентир для всех, а не рынок другой страны: «медь на LME», «бензин на бирже»
BENCH = r"\blme\b|\bлбм\b|\bcomex\b|биржев\w* (?:цен|котировк)|цен\w*[^.]{0,30}на бирж|товарно-сырьев|спбмтсб"
# акции, облигации, индексы — новости фондового рынка, а не цены товаров («Акции ММК на Мосбирже упали на 5%»)
SECURITIES = r"\bакци(?:и|й|ям|ями|ях|я|ю)\b|бумаг|облигац|доходност|индекс|капитализац"
# премия к биржевой цене (надбавка для покупателей одной страны) — не цена товара: «премия на алюминий упала на 35%»
PREMIUM = r"\bпреми(?:я|и|ю|ей|ям|ях)\b"
# Розничная цена — цена продажи для АЗС и магазина; биржевая и оптовая — цена закупки
RETAIL = r"розничн|на азс|на заправк|мелкоопт"
WHOLESALE = r"бирж|оптов|закупочн|\bнпз\b|\blme\b"
# Слова отрасли, которых нет в общем справочнике, но которые нужны для закупок: «арматура» — металл
EXTRA_STEMS = {"24": r"арматур"}


CROSS = r"экспорт|импорт|ввоз|вывоз"  # торговля через границу, а не продажи внутри страны
PARTS = re.compile(r",(?!\d)|;|\s(?:а|но|тогда как|при этом|в то время как)\s")  # части фразы с отдельными числами


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
        if re.search(SECURITIES, t) or re.search(PREMIUM, t):
            continue
        abroad = [k for k, v in FOREIGN.items() if re.search(v, t)]
        # экспорт, импорт, рынок другой страны — не спрос внутри страны: продажи машин не переносим на автодилеров
        cross = bool(abroad) or bool(re.search(CROSS, t))
        # два процента в одной фразе — у каждой части свои отрасль и число («Цены на новостройки выросли на 3%,
        # а стоимость арматуры — на 5%»); отрасль, глагол, вид шока и оговорка прогноза без повтора — из предыдущей
        # части («спрос на жильё подскочил на 5%, а число сделок выросло на 3%» — обе части про жильё)
        parts = PARTS.split(s) if len(re.findall(PCT, t)) > 1 else [s]
        verb = kind_prev = None
        modal, ctx = False, []
        for part in parts:
            tp = _low(part)
            ctx = sorted(set(sectors_in(part)) | {c for c, rx in EXTRA_STEMS.items() if re.search(rx, tp)}) or ctx
            m = re.search(r"(" + DOWN + "|" + UP + r")\w*[^.%;]{0,80}?на\s+" + PCT, tp) or \
                re.search(r"на\s+" + PCT + r"[^.%;]{0,40}?(" + DOWN + "|" + UP + r")", tp)
            if m:
                verb, val = (m.group(1), m.group(2)) if not m.group(1)[0].isdigit() else (m.group(2), m.group(1))
            elif verb and (m := re.search(r"на\s+" + PCT, tp)):
                val = m.group(1)
            else:
                continue
            modal = modal or _modal(tp, m.end())
            sectors = list(ctx)
            bench = bool(re.search(BENCH, t)) and "индекс" not in t and bool(set(sectors) - {"66"})
            if bench:
                sectors = [x for x in sectors if x != "66"]  # биржа здесь — площадка цены, а не отрасль брокеров
            kind = "price" if re.search(PRICE, tp) or bench else "volume" if re.search(VOLUME, tp) else kind_prev
            kind_prev = kind
            v = _f(val)
            if not sectors or not 0 < v <= 80:
                continue
            if kind is None:  # «просроченная задолженность выросла на 43%» — не спрос и не цены
                continue
            if any(x["quote"] == s[:300] and x["sectors"] == sectors for x in out):
                continue
            out.append({"sectors": sectors, "kind": kind, "change": -v if re.match(DOWN, verb) else v, "raw": val,
                        "quote": s[:300], "abroad": [] if bench else abroad, "bench": bench, "cross": cross,
                        "modal": modal, "sentence": s})
    return out


# Что компания закупает, а не продаёт: цена такого товара — её расход. Ключ — префикс полного кода ОКВЭД.
# (отрасль товара, основа слова — если отрасль товара совпадает с отраслью компании, название, оно же в
# родительном и предложном падеже, перепродаётся ли товар как есть — тогда розничная цена этого товара — цена продажи)
INPUTS = {
    "47.30": [("19", None, "топливо", "топлива", "топливе", True)],
    "46.71": [("19", None, "топливо", "топлива", "топливе", True)],
    "17.2": [("17", r"целлюлоз", "целлюлоза", "целлюлозы", "целлюлозе", False)],
    "25": [("24", None, "металл", "металла", "металле", False)],
    "41": [("24", None, "металл", "металла", "металле", False)],
}
# Продажи машин (29) — спрос и для автодилеров и автосервисов (45): «продажи авто упали» касается их покупателей
AUTO_CHAIN = {"29": ["45"]}


def _chain(sh: dict) -> list[str]:
    if sh.get("cross"):  # «экспорт автомобилей из Китая вырос» — не продажи российским покупателям автодилеров
        return list(sh["sectors"])
    return list(dict.fromkeys([*sh["sectors"], *(x for s in sh["sectors"] for x in AUTO_CHAIN.get(s, []))]))


def _input_goods(p: dict, sh: dict) -> dict | None:
    t = _low(sh["quote"])
    for prefix, items in INPUTS.items():
        if p["okved"].startswith(prefix):
            for sector, kw, name, gen, loc, resale in items:
                if sector in sh["sectors"] and (kw is None or re.search(kw, t)):
                    return {"sector": sector, "name": name, "gen": gen, "loc": loc, "resale": resale}
    return None


def _price_moves(sentence: str) -> list[tuple[str, float, str]]:
    """«Биржевые цены на бензин выросли на 8%, розничные — на 1,2%» → [('in', 8, '8'), ('out', 1.2, '1,2')]."""
    out, verb = [], None
    for c in PARTS.split(_low(sentence)):
        v = re.search("(" + DOWN + "|" + UP + ")", c)
        verb = v.group(1) if v else verb  # «розничные — на 1,2%»: глагол из предыдущей части
        m = re.search(r"на\s+" + PCT, c)
        if re.search(VOLUME, c) and not re.search(PRICE, c):
            continue  # «продажи топлива на АЗС упали на 3%» — объём продаж, а не цена
        if m and verb:
            out.append(("out" if re.search(RETAIL, c) else "in", (-1 if re.match(DOWN, verb) else 1) * _f(m.group(1)), m.group(1)))
    return out


def _flows(p: dict, side: str) -> dict[str, float]:
    """Доли отраслей в платежах через банк; при малой выборке (меньше 10 плательщиков) — пусто: доли случайны."""
    s = p[side]
    return {b["okved2"]: b["share"] for b in s["by_industry"]} if s["n"] >= FLOW_N_MIN else {}


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова после числа: 1 юрлицо, 3 юрлица, 5 юрлиц."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


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
        lines.append(f"Группа: {g['size']} {_plural(g['size'], 'юрлицо', 'юрлица', 'юрлиц')}; "
                     "доля юрлиц группы по отраслям (не доля платежей): "
                     + ", ".join(f"{label(x['okved2'])} — {pct(x['share'])}" for x in big))
    # «видно лишь 1 плательщика / 3 плательщиков» — винительный падеж одушевлённого: одного плательщика, трёх плательщиков
    for side, name, one, many, flow in (("customers", "Покупатели", "плательщика", "плательщиков", "поступлений"),
                                        ("suppliers", "Поставщики", "получателя платежей", "получателей платежей", "платежей")):
        s = p[side]
        if s["n"] < FLOW_N_MIN:
            lines.append(f"{name}: " + (f"по платежам через банк видно лишь {s['n']} {_plural(s['n'], one, many, many)} — оценку не даём"
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
            f"1 млрд ₽ × {_pc(ch)}", [fact(fx["quote"]), fact(f"Внешняя торговля: {countries}", profile_src)],
            ["Направление внешней торговли (импорт или экспорт) в данных не указано: для импортёра это расход, "
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
                whole = du.get("new") or du.get("cancel")  # вся ставка, а не изменение в п.п.
                formula = f"1 млрд ₽ × {_pc(delta) if whole else pp(delta)}"
                notes = ["Ставка применяется ко всей базе; льготы и переходные периоды не учтены",
                         "Переложение в цены покупателям не учтено"]
                if kind == "экспортная пошлина":
                    notes.insert(0, "Направление внешней торговли в данных не указано: это расход, только если вы вывозите этот товар")
                if du.get("new"):
                    notes.insert(0, "Пошлина вводится впервые: её ставка целиком — новый расход")
                if du.get("cancel"):
                    notes.insert(0, "Пошлина отменяется: её ставка целиком перестаёт быть расходом")
            c = _calc(title, f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ {base}", formula, [fact(du["quote"])], notes,
                      _sign(title, eff), {"label": label_, "per_bn": eff, "period": "за год"})
            calcs.append(_scenario(c, "решение примут") if du.get("modal") else c)

    cust, supp = _flows(p, "customers"), _flows(p, "suppliers")
    trade = p["okved2"] in ("46", "47")
    # рынок другой страны, с которой компания не работает: страна названа во фразе шока или в заголовке и начале
    # новости («В Германии бензин и дизель подорожали… Цены выросли на 37%»); мировые биржевые цены — для всех
    lead = [c for c, rx in FOREIGN.items() if re.search(rx, _low(event.get("title", "") + " " + event.get("text", "")[:300]))]
    alien = lambda cs: any(c not in p["foreign_trade"]["countries"] and not (c == "ЕВРОСОЮЗ" and ved) for c in cs)  # noqa: E731
    shocks = [sh for sh in sector_shocks(text, used) if not alien(sh["abroad"]) and (sh["bench"] or not alien(lead))]
    blocked = False     # шок был, но за месяц или неделю — на год не переносим и масштаб не показываем

    # 1. Цена того, что компания закупает: закупки, а если двигаются и закупочные, и розничные цены — маржа
    moves: dict[str, dict] = {}
    rest = []
    paired: set[str] = set()  # фразы, давшие и закупочную, и розничную цену: другие их части — уже в марже
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
            blocked = True  # объём производства сырья — не спрос на продукцию компании: ни расчёта, ни масштаба
            continue
        mv = moves.setdefault(g["name"], {"g": g})
        pair = _price_moves(sh["sentence"]) if g["resale"] else []
        if {"in", "out"} <= {x[0] for x in pair}:  # обе цены в одной фразе
            mv.setdefault("in", (next(x for x in pair if x[0] == "in")[1], sh))
            mv.setdefault("out", (next(x for x in pair if x[0] == "out")[1], sh))
            paired.add(sh["sentence"])
        else:
            mv.setdefault("in", (sh["change"], sh))
    rest = [sh for sh in rest if sh["sentence"] not in paired]
    # розничная цена того же товара в другой фразе («Розничные цены на АЗС выросли на 1,2%») — цена продажи: пара для маржи
    for mv in moves.values():
        g = mv["g"]
        retail = [sh for sh in rest if g["resale"] and sh["kind"] == "price" and {g["sector"], p["okved2"]} & set(sh["sectors"])
                  and re.search(RETAIL, _low(sh["sentence"])) and not re.search(WHOLESALE, _low(sh["sentence"]))]
        if retail and "out" not in mv:
            mv["out"] = (retail[0]["change"], retail[0])
            rest.remove(retail[0])
    for name, mv in moves.items():
        g, (ch_in, sh_in) = mv["g"], mv["in"]
        quotes = list(dict.fromkeys([sh_in["quote"], *([mv["out"][1]["quote"]] if "out" in mv else [])]))
        modal = sh_in["modal"] or ("out" in mv and mv["out"][1]["modal"])
        if "out" in mv:
            ch_out = mv["out"][0]
            d = ch_out - ch_in
            eff = d / 100 * BN
            c = _calc(
                f"Маржа на {g['loc']}", f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ закупок {g['gen']}",
                f"1 млрд ₽ × ({_pc(ch_out)} {'−' if ch_in > 0 else '+'} {num(abs(ch_in), 2)}%)", [fact(q) for q in quotes],
                [f"Закупочные цены меняются на {_pc(ch_in)}, цены продажи — на {_pc(ch_out)}: разница ложится на маржу",
                 "Объём продаж не меняется; наценка не учтена — выручка с того же объёма принята равной закупкам"],
                _sign("Маржа", eff), {"label": f"Ваши закупки {g['gen']} за год, млрд ₽", "per_bn": eff, "period": "за год"})
        else:
            eff = ch_in / 100 * BN
            c = _calc(
                f"Закупки {g['gen']}", f"{_plus(eff)}{rub(eff)} на каждый 1 млрд ₽ закупок {g['gen']}",
                f"1 млрд ₽ × {_pc(ch_in)}", [fact(quotes[0]), fact(f"Вид деятельности: {p['industry']}", profile_src)],
                [f"Для вас {name} — закупка: её цена меняется так же, как в новости",
                 "Условия действующих контрактов и переложение в цены продажи не учтены"],
                _sign("Закупки", eff), {"label": f"Ваши закупки {g['gen']} за год, млрд ₽", "per_bn": eff, "period": "за год"})
        calcs.append(_scenario(c, "так и будет" if "out" in mv else
                               f"цены {'вырастут' if ch_in > 0 else 'снизятся'} на {num(abs(ch_in), 2)}%") if modal else c)

    # 2. Спрос и цены в отраслях покупателей, поставщиков и своей. Одна оценка выручки на событие: по доле
    #    покупателей, если отрасль покупателей видна в платежах, иначе — «если ваши продажи изменятся как у рынка»
    #    (если рынок — ваш, надёжность низкая). Две разные суммы выручки на одно событие только путают
    seen: set[str] = set()
    revenue_done = False
    for sh in rest:
        ch, kind = sh["change"], sh["kind"]
        sectors = [s for s in _chain(sh) if s not in seen]
        what = "цены" if kind == "price" else "спрос/объёмы"
        period = _period(sh["quote"], text, sh["raw"])
        # рынок другой страны («экспорт автомобилей из Китая вырос на 20%») — не ваши продажи: выручку не считаем,
        # и «размер эффекта в новости не указан» тоже было бы неправдой
        can_revenue = not revenue_done and period != "short" and not sh["abroad"]
        blocked = blocked or period == "short" or bool(sh["abroad"])
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
                if share >= FLOW_MIN and s not in goods:
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
                    [f"Выручка от этих покупателей меняется так же, как {what} в их отрасли",
                     f"Доля — по платежам через банк; отрасль плательщика известна для {pct(p['customers']['known_industry_share'])} "
                     "поступлений, поэтому это оценка по известной части"] + ([MONTH_YOY] if period == "month_yoy" else []),
                    _sign("Выручка", eff), strength="medium"))
        # своя отрасль — только если нет ни оценки по покупателям, ни закупок у поставщиков по этой фразе
        if can_revenue and own and not out:
            for s in own:
                seen.add(s)
            revenue_done = True
            eff = R * ch / 100
            notes = [f"{what.capitalize()} компании меняются так же, как в отрасли в целом",
                     "Компания может расти или падать быстрее рынка; это ориентир масштаба, а не прогноз"]
            if kind == "price" and trade:
                notes.append("В торговле закупочные цены обычно меняются вместе с ценами продажи: прибыль меняется меньше выручки")
            if period == "month_yoy":
                notes.append(MONTH_YOY)
            out.append(_calc(
                f"Выручка, если ваши {'цены' if kind == 'price' else 'продажи'} изменятся как у рынка",
                f"{_plus(eff)}{rub(eff)} в год", f"{num(R / BN)} млрд ₽ × {_pc(ch)}",
                [sh_news, fact(f"Выручка компании за год: {num(R / BN)} млрд ₽", profile_src)],
                notes, _sign("Выручка", eff), strength="low"))
        calcs += [_scenario(c, cond) for c in out] if sh["modal"] else out

    # Без процента в новости: сколько выручки связано с отраслью события (масштаб того, что под влиянием).
    # Только когда другого расчёта по событию нет: рядом с расчётом пошлины или ставки «размер эффекта
    # в новости не указан» было бы неправдой
    head = event.get("title", "") + " " + event.get("text", "")[:600]
    abroad = [c for c, rx in FOREIGN.items() if re.search(rx, _low(head))]
    if not calcs and not blocked and not (abroad and not set(abroad) & set(p["foreign_trade"]["countries"])):
        best = max(_chain({"sectors": sectors_in(head)}), key=lambda s: cust.get(s, 0), default=None)
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
