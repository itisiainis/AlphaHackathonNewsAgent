"""Калькулятор эффекта события для компании: формулы в коде, без LLM.

Зачем: «как событие повлияет на компанию» — в рублях, а не словами. Модель хорошо объясняет,
но плохо считает, поэтому цифры эффекта считает код по явным формулам, а модель только
ссылается на результат. Каждый расчёт показывает формулу, входные данные с происхождением
и допущения — чтобы руководитель мог проверить и пересчитать под себя.

Что умеет (шок берётся из текста новости правилами; нет шока — нет расчёта):
- ключевая ставка → стоимость плавающего долга и доход по депозитам на каждый 1 млрд ₽;
- курс рубля → стоимость импортных закупок (или валютной выручки) на каждый 1 млрд ₽;
- пошлина / утильсбор / налог на прибыль → доп. расходы на 1 млрд ₽ базы;
- изменение спроса или цен в отрасли → выручка от покупателей этой отрасли по доле входящего потока
  и расходы на закупки у поставщиков этой отрасли.

Честность (ограничение кейса 06): суммы долга, депозитов и импорта в данных кейса нет — есть только
факт наличия продукта и ВЭД. Поэтому эффект даётся на единицу базы с полем «введите свою сумму».
Доли отраслей — по платёжной выборке банка, где отрасль известна лишь для части потока:
это оценка по известной части, а не полный эффект.
"""
from __future__ import annotations

import re

from .okved import country_name, label, related_sectors, sectors_in

BN = 1e9
FLOW_MIN = 0.02     # отрасль контрагентов меньше 2% потока не считаем
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


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+(?=[А-ЯA-Z«\"(\d])", text) if s.strip()]


def _low(s: str) -> str:
    return s.lower().replace("ё", "е")


DOWN = r"сниз|понизи|снижен|смягч|уменьш|сократ|упал|упад|подешев|обвал|рухн|падени"
UP = r"повыс|повыш|поднял|подня|повышен|ужесточ|увелич|вырос|рост|подорож|ускор"
HOLD = r"сохрани|оставил|без изменени|не стал менять|паузу"


# ---------- извлечение шоков из текста ----------
def _rate_in(t: str) -> dict:
    out: dict = {}
    m = re.search(r"\bс\s+" + PCT + r"(?:\s*годовых)?\s+до\s+" + PCT, t)
    if m:
        a, b = _f(m.group(1)), _f(m.group(2))
        return {"level": b, "delta": b - a}
    sign = -1 if re.search(DOWN, t) else 1 if re.search(UP, t) else 0
    bp = re.search(r"на\s+(\d+)\s*(?:б\.\s?п|базисн)", t)
    ppm = re.search(r"на\s+" + NUM + r"\s*(?:п\.\s?п|процентн\w* пункт)", t)
    if bp and sign:
        out["delta"] = sign * int(bp.group(1)) / 100
    elif ppm and sign:
        out["delta"] = sign * _f(ppm.group(1))
    elif re.search(HOLD, t):
        out["delta"] = 0.0
    lv = re.search(r"(?:\bдо|на уровне|уровне|составляет|составит|ставк\w*(?:\s+в)?)\s+" + PCT, t)
    if lv:
        out["level"] = _f(lv.group(1))
    return out


def rate_shock(text: str) -> dict | None:
    """Ключевая ставка: уровень, изменение в п.п. и фразы-основания (дословно из новости).

    Уровень и изменение могут стоять в разных предложениях («ЦБ сохранил ставку» + «…на уровне 14%»)."""
    found: dict = {}
    quotes: list[str] = []
    for s in _sentences(text):
        t = _low(s)
        if not ("ключев" in t and "ставк" in t):
            continue
        r = _rate_in(t)
        if "level" in r and not 2 <= r["level"] <= 40:
            r.pop("level")
        if "delta" in r and abs(r["delta"]) > 10:
            r.pop("delta")
        new = {k: v for k, v in r.items() if k not in found}
        if new:
            found |= new
            quotes.append(s[:300])
        if "level" in found and "delta" in found:
            break
    return found | {"quote": " ".join(quotes)} if found else None


CURRENCY = r"доллар|юан|евро|дирхам|лир[аыу]\b|рупи|рубл"


def fx_shock(text: str) -> dict | None:
    """Курс: на сколько процентов ослаб (+) или укрепился (−) рубль. Подлежащее — валюта, названная первой."""
    for s in _sentences(text):
        t = _low(s)
        # валюта как подлежащее («рубль ослаб», «курс юаня вырос»), а не сумма («1,2 млрд рублей»)
        cur = re.search(r"(?<!млрд )(?<!млн )(?<!трлн )(?<!тыс\. )\b(?:курс\w*\s+)?(" + CURRENCY + r")", t)
        m = re.search(r"на\s+" + PCT, t)
        if not cur or not m or "ставк" in t or not ("курс" in t or re.search(r"\b(рубль|доллар|юань|евро)\b", t)):
            continue
        down, up = re.search(DOWN + r"|ослаб|обесцен|девальв", t), re.search(UP + r"|укрепи", t)
        if not (down or up) or down and up:
            continue
        ruble_first = cur.group(1) == "рубл"
        weaker = (down is not None) if ruble_first else (up is not None)
        if "укрепи" in t and ruble_first:
            weaker = False
        size = _f(m.group(1))
        if not 0 < size <= 50:
            continue
        return {"change_pct": size if weaker else -size, "quote": s[:300], "raw": m.group(1)}
    return None


def duty_shock(text: str) -> dict | None:
    """Пошлина, утильсбор, налог на прибыль: новая ставка или изменение."""
    for s in _sentences(text):
        t = _low(s)
        kind = ("налог на прибыль" if "налог на прибыль" in t or "налога на прибыль" in t else
                "утильсбор" if "утильсбор" in t or "утилизационн" in t else
                "пошлина" if "пошлин" in t else None)
        if not kind:
            continue
        m = re.search(r"\bс\s+" + PCT + r"\s+до\s+" + PCT, t)
        if m:
            return {"kind": kind, "from": _f(m.group(1)), "to": _f(m.group(2)), "quote": s[:300]}
        m = re.search(r"(?:на|в размере|ставк\w*)\s+" + PCT, t) or re.search(r"\bдо\s+" + PCT, t)
        if m and kind != "налог на прибыль":
            v = _f(m.group(1))
            up = re.search(UP + r"|введ|ввод|установ", t)
            if 0 < v <= 100 and up:
                return {"kind": kind, "delta": v, "quote": s[:300]}
    return None


# Рынок за рубежом: «продажи недвижимости в Дубае упали» не переносим на российскую компанию,
# если страны нет в её ВЭД. Ключ — название страны, как в профиле (foreign_trade.countries).
FOREIGN = {"ОАЭ": r"оаэ|дуба|абу-даби|эмират", "КИТАЙ": r"кита|кнр|пекин|шанха", "США": r"\bсша\b|американ",
           "ТУРЦИЯ": r"турци|турецк|стамбул", "ИНДИЯ": r"инди[ия]|индийск", "ГЕРМАНИЯ": r"германи|немецк",
           "ЕВРОСОЮЗ": r"\bес\b|евросоюз|европ", "ЯПОНИЯ": r"япони|японск", "КАЗАХСТАН": r"казахстан",
           "БЕЛАРУСЬ": r"беларус|белорус", "ВЕЛИКОБРИТАНИЯ": r"британи|великобритан|лондон"}

# Показатели, которые не спрос и не цены отрасли: запасы, ВВП, долги, прибыль, капитализация
NOT_MARKET = r"запас|резерв|ввп|задолженност|просроч|прибыл|капитализац|убыт|инфляц"
PRICE = r"цен|подорож|подешев|стоимост|тариф|котировк"
VOLUME = r"продаж|спрос|выпуск|производств|отгрузк|потреблен|ввод|выдач|импорт|экспорт|перевоз|оборот"


def sector_shocks(text: str) -> list[dict]:
    """«Продажи легковых автомобилей упали на 12%» → {sector: 29/45, kind: volume, change: −12}."""
    out = []
    for s in _sentences(text):
        t = _low(s)
        if ("ключев" in t and "ставк" in t) or "курс" in t or re.search(NOT_MARKET, t):
            continue
        m = re.search(r"(" + DOWN + "|" + UP + r")\w*[^.%;]{0,80}?на\s+" + PCT, t) or \
            re.search(r"на\s+" + PCT + r"[^.%;]{0,40}?(" + DOWN + "|" + UP + r")", t)
        if not m:
            continue
        verb, val = (m.group(1), m.group(2)) if not m.group(1)[0].isdigit() else (m.group(2), m.group(1))
        sectors = sectors_in(s)
        if not sectors:
            continue
        v = _f(val)
        if not 0 < v <= 80:
            continue
        change = -v if re.match(DOWN, verb) else v
        kind = "price" if re.search(PRICE, t) else "volume" if re.search(VOLUME, t) else None
        if kind is None:  # «просроченная задолженность выросла на 43%» — не спрос и не цены
            continue
        if any(x["quote"] == s[:300] for x in out):
            continue
        out.append({"sectors": sectors, "kind": kind, "change": change, "raw": val, "quote": s[:300],
                    "abroad": [k for k, v in FOREIGN.items() if re.search(v, t)]})
    return out


# ---------- расчёты ----------
def profile_text(p: dict) -> str:
    """Текст «источника» с данными профиля: по нему валидатор сверяет числа входных данных."""
    own = [p["okved2"], *related_sectors(p["okved"]), *(g["okved2"] for g in p["group"]["industries"] if g["share"] >= 0.1)]
    lines = [f"ОКВЭД компании: {p['okved']}; отрасли компании, её товара и группы: " + ", ".join(label(s) for s in dict.fromkeys(own)),
             f"Выручка компании за год: {num(p['revenue_rub'] / BN)} млрд ₽",
             f"Продукты банка: {', '.join(p['bank_products']) or 'нет'}",
             "ВЭД: " + (", ".join(country_name(c) for c in p["foreign_trade"]["countries"]) or "нет")]
    for side, name in (("customers", "входящего"), ("suppliers", "исходящего")):
        s = p[side]
        if s["n"]:
            lines.append(f"Отрасль известна для {round(s['known_industry_share'] * 100)}% {name} потока")
            for b in s["by_industry"]:
                if b["share"] >= FLOW_MIN:
                    lines.append(f"{label(b['okved2'])}: {round(b['share'] * 100)}% {name} потока")
    return "\n".join(lines)


def _calc(title, value, formula, inputs, assumptions, sign, scale=None, strength="medium") -> dict:
    c = {"title": title, "value": value, "formula": formula, "inputs": inputs,
         "assumptions": assumptions, "sign": sign, "strength": strength}
    if scale:
        c["scale"] = scale
    return c


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

    rs = rate_shock(text)
    if rs:
        news_in = fact(rs["quote"])
        d, lvl = rs.get("delta"), rs.get("level")
        if has_debt:
            prods = ", ".join(x for x in ("кредит", "лизинг") if x in products)
            if d:
                eff = d / 100 * BN
                calcs.append(_calc(
                    "Стоимость долга", f"{'+' if eff > 0 else ''}{rub(eff)} в год на каждый 1 млрд ₽ долга с плавающей ставкой",
                    f"1 млрд ₽ × {pp(d)}", [news_in, fact(f"Продукты банка: {prods}", profile_src)],
                    ["Ставка по кредиту привязана к ключевой и меняется на ту же величину",
                     "Для долга с фиксированной ставкой эффект появится при рефинансировании"],
                    "risk" if d > 0 else "opportunity",
                    {"label": "Ваш долг с плавающей ставкой, млрд ₽", "per_bn": eff, "period": "в год"}))
            elif lvl:
                calcs.append(_calc(
                    "Процентные расходы", f"{rub(lvl / 100 * BN)} в год на каждый 1 млрд ₽ — только ключевая часть ставки",
                    f"1 млрд ₽ × {num(lvl, 2)}%", [news_in, fact(f"Продукты банка: {prods}", profile_src)],
                    ["Без спреда банка: полная ставка = ключевая + спред по договору",
                     "Ставка не изменилась — процентные расходы остаются прежними, удешевления долга пока нет" if d == 0
                     else "Размер изменения ставки в тексте новости не указан"], "neutral",
                    {"label": "Ваш долг с плавающей ставкой, млрд ₽", "per_bn": lvl / 100 * BN, "period": "в год"}, "low"))
        if "депозиты" in products and d:
            eff = d / 100 * BN
            calcs.append(_calc(
                "Доход по депозитам", f"{'+' if eff > 0 else ''}{rub(eff)} в год на каждый 1 млрд ₽ размещённых средств",
                f"1 млрд ₽ × {pp(d)}", [news_in, fact("Продукты банка: депозиты", profile_src)],
                ["Ставки по новым депозитам следуют за ключевой", "Действующие срочные депозиты доработают по старой ставке"],
                "opportunity" if d > 0 else "risk",
                {"label": "Свободные средства на депозитах, млрд ₽", "per_bn": eff, "period": "в год"}))

    fx = fx_shock(text)
    if fx and ved:
        ch = fx["change_pct"]
        eff = ch / 100 * BN
        calcs.append(_calc(
            "Валютные расчёты", f"{'+' if eff > 0 else ''}{rub(eff)} на каждый 1 млрд ₽ валютных закупок "
            f"({'дороже' if ch > 0 else 'дешевле'} в рублях)",
            f"1 млрд ₽ × {'+' if ch > 0 else '−'}{num(abs(ch), 2)}%", [fact(fx["quote"]), fact(f"ВЭД: {countries}", profile_src)],
            ["Направление ВЭД (импорт или экспорт) в данных не указано: для импортёра это расход, "
             "для экспортёра — такая же прибавка к рублёвой выручке",
             "Без учёта хеджирования и валютной оговорки в контрактах"],
            "risk" if ch > 0 else "opportunity",
            {"label": "Ваши валютные закупки за год, млрд ₽", "per_bn": eff, "period": "за год"}))

    du = duty_shock(text)
    if du:
        delta = du.get("delta", du.get("to", 0) - du.get("from", 0))
        if delta and (du["kind"] == "налог на прибыль" or ved or {"29", "45"} & {p["okved2"], *related_sectors(p["okved"])}):
            base = "прибыли до налогообложения" if du["kind"] == "налог на прибыль" else "таможенной стоимости ввоза"
            eff = delta / 100 * BN
            calcs.append(_calc(
                {"налог на прибыль": "Налог на прибыль", "утильсбор": "Утильсбор", "пошлина": "Пошлина"}[du["kind"]],
                f"{'+' if eff > 0 else ''}{rub(eff)} на каждый 1 млрд ₽ {base}",
                f"1 млрд ₽ × {pp(delta) if 'from' in du else num(delta, 2) + '%'}", [fact(du["quote"])],
                ["Ставка применяется ко всей базе; льготы и переходные периоды не учтены",
                 "Переложение в цены покупателям не учтено"],
                "risk" if eff > 0 else "opportunity",
                {"label": f"Ваша база ({base}), млрд ₽", "per_bn": eff, "period": "за год"}))

    cust = {b["okved2"]: b["share"] for b in p["customers"]["by_industry"]}
    supp = {b["okved2"]: b["share"] for b in p["suppliers"]["by_industry"]}
    own = {p["okved2"], *related_sectors(p["okved"]), *(g["okved2"] for g in p["group"]["industries"] if g["share"] >= 0.1)}
    seen: set[str] = set()
    for sh in sector_shocks(text):
        ch, kind = sh["change"], sh["kind"]
        if any(c not in p["foreign_trade"]["countries"] and not (c == "ЕВРОСОЮЗ" and ved) for c in sh["abroad"]):
            continue  # рынок другой страны, с которой компания не работает
        what = "цены" if kind == "price" else "спрос/объёмы"
        revenue_done = False
        # сначала отрасли покупателей (точнее), потом своя отрасль (верхняя оценка) — одна оценка выручки на шок
        for s in sorted(sh["sectors"], key=lambda x: -cust.get(x, 0)):
            if s in seen:
                continue
            share = cust.get(s, 0)
            if kind == "volume" and share >= FLOW_MIN:
                seen.add(s)
                revenue_done = True
                eff = R * share * ch / 100
                calcs.append(_calc(
                    f"Выручка от покупателей: {label(s)}", f"{'+' if eff > 0 else ''}{rub(eff)} в год",
                    f"{num(R / BN)} млрд ₽ × {round(share * 100)}% × {'+' if ch > 0 else '−'}{num(abs(ch), 2)}%",
                    [fact(sh["quote"]), fact(f"Выручка компании за год: {num(R / BN)} млрд ₽", profile_src),
                     fact(f"{label(s)}: {round(share * 100)}% входящего потока", profile_src)],
                    [f"Выручка от покупателей отрасли меняется так же, как {what} в отрасли (эластичность 1)",
                     f"Доля — по платёжной выборке банка; отрасль известна для {round(p['customers']['known_industry_share'] * 100)}% "
                     "входящего потока, поэтому это оценка по известной части"],
                    "risk" if eff < 0 else "opportunity", strength="medium"))
                continue
            share = supp.get(s, 0)
            if kind == "price" and share >= FLOW_MIN:
                seen.add(s)
                eff = ch / 100 * BN
                calcs.append(_calc(
                    f"Закупки у поставщиков: {label(s)}", f"{'+' if eff > 0 else ''}{rub(eff)} на каждый 1 млрд ₽ закупок у отрасли",
                    f"1 млрд ₽ × {'+' if ch > 0 else '−'}{num(abs(ch), 2)}%",
                    [fact(sh["quote"]), fact(f"{label(s)}: {round(share * 100)}% исходящего потока", profile_src)],
                    ["Цены поставщиков меняются вслед за ценами отрасли", "Условия действующих контрактов не учтены"],
                    "risk" if eff > 0 else "opportunity",
                    {"label": f"Ваши закупки у отрасли «{label(s)}» за год, млрд ₽", "per_bn": eff, "period": "за год"}))
                continue
            if s in own and not revenue_done:
                seen.add(s)
                revenue_done = True
                eff = R * ch / 100
                calcs.append(_calc(
                    f"Выручка: {what} в отрасли компании", f"{'+' if eff > 0 else ''}{rub(eff)} в год — верхняя оценка",
                    f"{num(R / BN)} млрд ₽ × {'+' if ch > 0 else '−'}{num(abs(ch), 2)}%",
                    [fact(sh["quote"]), fact(f"Выручка компании за год: {num(R / BN)} млрд ₽", profile_src)],
                    [f"{what.capitalize()} компании меняются так же, как в отрасли в целом",
                     "Компания может расти или падать быстрее рынка; это ориентир масштаба, а не прогноз"],
                    "risk" if eff < 0 else "opportunity", strength="low"))
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
