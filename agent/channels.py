"""Каналы влияния: как событие доходит до денег компании — кодом, без модели.

Каждый канал — причинная цепочка «событие → что меняется → что это значит для вас → сколько → что сделать → до когда»:
- что меняется — из новости: решение ЦБ, ставка, курс, пошлина или налог, цены и спрос в отрасли (agent/impact.py);
- что это значит для вас — из профиля по данным банка: продукты, доли поступлений и платежей, страны внешней торговли
  (признаки отбора agent/matching.py, факторы чувствительности agent/sensitivity.py);
- сколько — расчёт кодом по формуле, если в новости есть число (agent/impact.py); числа нет — так и пишем;
- взгляд банка — действующий прогноз (agent/house_view.py); до когда — следующее заседание ЦБ из материалов банка
  или дата вступления изменения из новости.
Из каналов краткий разбор собирает главный вывод под роль (финансовый директор, генеральный директор, казначей),
риски, возможности и действия. Это вывод по правилам — происхождение «rule», не вывод ИИ. Агенту каналы даются как
каркас: он проверяет их по источникам и углубляет (agent/loop.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from . import house_view, impact
from .matching import Match, cb_decision
from .okved import country_name, plain, who

MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря")
DAY_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s+(" + "|".join(MONTHS_GEN) + r")(?:\s+(20\d\d))?")
MEETING_RE = re.compile(r"заседани", re.I)
CB_RE = re.compile(r"(?<![а-яё])ЦБ|Банк\w* России|Совет\w* директоров|ключев\w* ставк", re.I)
# что ЦБ уже сделал — прошедшее время в заголовке или первой фразе («сохранил», «снизил»)
DONE = {"hold": re.compile(r"(?:сохранил|оставил)[аи]?(?![а-яё])|без изменени|взял[аи]? паузу", re.I),
        "cut": re.compile(r"(?:снизил|понизил)[аи]?(?![а-яё])", re.I),
        "hike": re.compile(r"(?:повысил|поднял)[аи]?(?![а-яё])", re.I)}
# Каналы по важности для роли: главный вывод роли — из первого канала, где для неё есть вывод
ORDER = {"cfo": ("rate", "tax", "suppliers", "fx", "sanctions", "buyers", "deposits", "market", "macro", "group"),
         "ceo": ("buyers", "market", "sanctions", "tax", "suppliers", "rate", "fx", "deposits", "macro", "group"),
         "treasurer": ("fx", "rate", "deposits", "buyers", "sanctions", "tax", "suppliers", "market", "macro", "group"),
         "accountant": ("tax", "fx", "sanctions", "buyers", "rate", "suppliers", "deposits", "market", "macro", "group")}
ROLES = tuple(ORDER)
RISK_MIN_SHARE = 0.2   # доля поступлений от отрасли покупателей, начиная с которой их проблемы — риск неплатежей для вас
NO_FIGURE = "количественной оценки в новости нет"
FLOAT_CHECK = "Определить долю долга с плавающей ставкой и сроки пересмотра ставки по договорам"
VAT_RE = re.compile(r"(?:ндс|налог\w* на добавленную стоимость)[^.]{0,80}?\bс\s+(\d{1,2})\s*%\s+до\s+(\d{1,2})\s*%", re.I)


@dataclass
class Channel:
    id: str                     # rate | deposits | macro | fx | sanctions | tax | buyers | suppliers | market | group
    factor: str | None          # фактор чувствительности бизнеса (agent/sensitivity.py), который затрагивает событие
    sign: str                   # risk | opportunity | neutral
    take: dict[str, str]        # главный вывод под роль; «*» — для всех ролей без своего вывода
    chain: str                  # цепочка одной строкой — в трассу и агенту
    risks: list[dict] = field(default_factory=list)
    opportunities: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    when: str = ""

    def for_role(self, role: str) -> str:
        return self.take.get(role) or self.take.get("*", "")


# ---------- общие куски ----------
def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _num(x: float) -> str:
    return impact.num(x, 2)


def _rule(t: str, cta: dict | None = None, **kw) -> dict:
    """Вывод по правилам: происхождение «rule». Кнопка действия — только если действие обосновано данными."""
    return {"t": t, "p": "rule", **kw, **({"cta": cta} if cta else {})}


def _base_title(title: str) -> str:
    return re.sub(r"^Если [^,]+, ", "", title)


def money(c: dict) -> str:
    """Сумма расчёта словами по смыслу: расход — «экономия / доп. расходы», доход — «потери / доп. доход».
    Сценарий из прогноза — с условием: «если ставку снизят на 0,5 п.п. — экономия 5 млн ₽ в год…»."""
    title, value = c["title"], c["value"]
    val = value.lstrip("+−- ")
    if value.startswith("≈"):
        phrase = val  # масштаб того, что под влиянием, а не изменение
    else:
        down = value.startswith(("−", "-"))
        cost = _base_title(title).startswith(impact.COST_TITLES)
        val = re.sub(r"\s*\([^)]*\)$", "", val)  # пояснение «(дороже в рублях)» есть в расчёте
        phrase = f"{('экономия' if down else 'дополнительные расходы') if cost else ('потери' if down else 'дополнительный доход')} {val}"
    if title.startswith("Если "):
        return f"если {title[5:title.index(',')]} — {phrase}"
    return phrase


def _calc(calcs: list[dict], *prefixes: str) -> dict | None:
    return next((c for c in calcs if _base_title(c["title"]).startswith(prefixes)), None)


def _when_in(text: str, today: date) -> str:
    """Дата вступления изменения из новости: «с 1 января 2027 года» → «с 1 января 2027»."""
    m = re.search(r"\bс\s+(\d{1,2})\s+(" + "|".join(MONTHS_GEN) + r")(?:\s+(20\d\d))?", text, re.I)
    if m:
        return f"с {m.group(1)} {m.group(2).lower()}" + (f" {m.group(3)}" if m.group(3) else "")
    m = re.search(r"\bс\s+(20\d\d)\s+год", text)
    return f"с {m.group(1)} года" if m and int(m.group(1)) >= today.year else ""


def _day(d: date) -> str:
    return f"{d.day} {MONTHS_GEN[d.month - 1]}"


# ---------- календарь ЦБ и взгляд банка: из материалов банка, а не из кода ----------
def cb_meetings(items: list[dict]) -> list[date]:
    """Даты заседаний ЦБ по ставке, названные в материалах банка и новостях: «два заседания ЦБ по ключевой ставке
    (23 октября и 18 декабря)». Год — из даты материала; дата раньше материала без года — следующий год."""
    out: set[date] = set()
    for it in items:
        text = (it.get("title") or "") + ". " + (it.get("text") or "")
        d0 = (it.get("date") or it.get("published") or "")[:10]
        if not d0 or not MEETING_RE.search(text):
            continue
        base = date.fromisoformat(d0)
        for s in re.split(r"(?<=[.!?])\s+|\n+", text):
            if not (MEETING_RE.search(s) and CB_RE.search(s)) or re.search(r"ФРС|ЕЦБ|ЦБ [А-ЯЁ][а-яё]+и\b", s):
                continue
            for m in DAY_MONTH.finditer(s):
                try:
                    d = date(int(m.group(3) or base.year), MONTHS_GEN.index(m.group(2)) + 1, int(m.group(1)))
                except ValueError:
                    continue
                if not m.group(3) and d < base:
                    d = date(d.year + 1, d.month, d.day)
                out.add(d)
    return sorted(out)


_MEETINGS: dict[tuple, list[date]] = {}


def meetings_for(*indexes) -> list[date]:
    """Календарь ЦБ по индексам аналитики банка и новостей (BM25Index.items) — считается один раз на набор индексов."""
    key = tuple((id(ix), len(ix.items)) for ix in indexes)
    if key not in _MEETINGS:
        if len(_MEETINGS) > 32:
            _MEETINGS.clear()
        _MEETINGS[key] = cb_meetings([it for ix in indexes for it in ix.items])
    return _MEETINGS[key]


def next_meeting(meetings: list[date], today: date) -> date | None:
    return next((d for d in meetings if d > today), None)


def bank_rate_view(today: date) -> list[str]:
    """Действующие прогнозы банка по ставке на конец ближайших лет: [«14,0% на конец 2026», «12,0% на конец 2027»]."""
    out = []
    for _, hor in sorted(k for k in house_view.keys() if k[0] == "key_rate" and k[1].startswith("end-")):
        cur = house_view.current("key_rate", hor, today) if int(hor[4:]) >= today.year else None
        if cur:
            out.append(f"{house_view.fmt_value(cur['value'], cur['unit'])} {house_view.horizon_text(hor)}")
    return out[:2]


def rate_decision(event: dict) -> str:
    """Решение ЦБ в событии: hold | cut | hike — только если решение в заголовке или первой фразе и в прошедшем
    времени. Прогноз («аналитики ждут сохранения»), анонс заседания и упоминание прошлого решения — пустая строка."""
    title, text = event.get("title", ""), event.get("text", "")
    if not cb_decision(title, text):
        return ""
    head = title + " " + text[:300]
    hits = [(m.start(), d) for d, rx in DONE.items() if (m := rx.search(head))]
    if hits:
        return min(hits)[1]
    rs = impact.rate_shock(head) or {}
    if "delta" in rs and not rs.get("modal"):
        return "hold" if rs["delta"] == 0 else "cut" if rs["delta"] < 0 else "hike"
    return ""


# ---------- каналы ----------
def _rate(text: str, by: dict, p: dict, calcs: list[dict], today: date, meeting: date | None, decision: str) -> Channel:
    products = set(p["bank_products"])
    debt = [x for x in ("кредит", "лизинг") if x in products]
    rs = impact.rate_shock(text) or {}
    lvl = f"{_num(rs['level'])}%" if rs.get("level") and decision else ""
    view = bank_rate_view(today)
    near = view[0] if view else ""
    bank = f"прогноз Альфа-Банка — {' и '.join(view)}" if view else ""
    meet = f"следующее решение ЦБ — {_day(meeting)}" if meeting else ""
    c_debt, c_dep = _calc(calcs, "Стоимость долга"), _calc(calcs, "Доход по депозитам")
    when = meet
    if debt:
        what = " и ".join(debt)
        dat = " и ".join({"кредит": "кредиту", "лизинг": "лизингу"}[x] for x in debt)
        gen = " и ".join({"кредит": "кредита", "лизинг": "лизинга"}[x] for x in debt)
        not_cheaper = f"стоимость обслуживания {gen} не снизится"
        if decision == "hold":
            take = {"cfo": f"Ставка сохранена{' на уровне ' + lvl if lvl else ''}: {not_cheaper}" + (f", {meet}" if meet else "")
                           + (f"; прогноз Альфа-Банка — {near}" if near else ""),
                    "ceo": "Стоимость заёмных средств остаётся высокой: проекты и закупки в кредит оценивайте по текущей ставке"
                           + (f"; {bank}" if bank else ""),
                    "treasurer": f"Платежи по {dat} с плавающей ставкой не снизятся" + (f": {meet}" if meet else ""),
                    "accountant": f"Проценты по {dat} в расходах останутся прежними" + (f": {meet}" if meet else "")}
            return Channel("rate", "rate", "risk", take, f"ставка без изменений {lvl} → {what} не дешевеют → {meet or bank}",
                           [_rule("Рефинансирование на более выгодных условиях до следующего решения ЦБ маловероятно" if meeting
                                  else "Рефинансирование на более выгодных условиях в ближайшие месяцы маловероятно",
                                  strength="medium")], [],
                           [_rule(FLOAT_CHECK)], when)
        if decision in ("cut", "hike"):
            cut = decision == "cut"
            how = f": {money(c_debt)}" if c_debt else ""
            take = {"cfo": f"Ставка {'снижена' if cut else 'повышена'}{' до ' + lvl if lvl else ''} — "
                           + (money(c_debt) if c_debt else f"стоимость долга с плавающей ставкой {'снижается' if cut else 'растёт'}"),
                    "ceo": ("Стоимость заимствований снижается: отложенные инвестиционные проекты стоит пересчитать по новой ставке"
                            if cut else "Стоимость заимствований растёт: инвестиционные проекты в кредит пересчитайте по новой ставке")
                           + (f"; {bank}" if bank else ""),
                    "treasurer": f"Платежи по {dat} с плавающей ставкой {'снизятся' if cut else 'вырастут'} с ближайшего "
                                 "пересмотра ставки по договорам",
                    "accountant": f"Проценты по {dat} в расходах {'снизятся' if cut else 'вырастут'} с месяца пересмотра "
                                  "ставки по договору — учтите в бюджете и прогнозе налога на прибыль"}
            if cut:
                return Channel("rate", "rate", "opportunity", take, f"ставка ↓ {lvl} → {what} дешевеют{how.replace(':', ' →')}",
                               [], [_rule("Рефинансирование долга по новой, более низкой ставке", strength="medium")],
                               [_rule("Определить, по каким договорам ставка пересматривается автоматически, а какие "
                                      "целесообразно рефинансировать",
                                      cta={"label": "Обсудить рефинансирование с менеджером", "kind": "manager"})], when)
            return Channel("rate", "rate", "risk", take, f"ставка ↑ {lvl} → {what} дорожают{how.replace(':', ' →')}",
                           [_rule("Платежи по долгу с плавающей ставкой вырастут", strength="high")], [],
                           [_rule("Оценить долю долга с плавающей ставкой; новые кредиты рассматривать с фиксированной ставкой")],
                           when)
        # решения в новости нет: анонс заседания, прогноз, обсуждение ставки
        scen = money(c_debt) if c_debt else ""
        take = {"cfo": (f"Решение по ставке — {_day(meeting)}" if meeting else "Решения по ставке в новости нет")
                       + (f"; прогноз Альфа-Банка — {near}" if near else "") + f": в плане платежей по {dat} закладывайте текущую ставку"
                       + (f"; {scen}" if scen else ""),
                "ceo": "Стоимость заёмных средств остаётся высокой: проекты в кредит оценивайте по текущей ставке"
                       + (f"; {bank}" if bank else ""),
                "treasurer": (f"Платежи по {dat} с плавающей ставкой до решения ЦБ {_day(meeting)} не изменятся" if meeting
                              else f"Платежи по {dat} с плавающей ставкой пока без изменений") + (f"; {scen}" if scen else ""),
                "accountant": f"Проценты по {dat} в расходах" + (f" до решения ЦБ {_day(meeting)}" if meeting else "")
                              + " не изменятся"}
        return Channel("rate", "rate", "neutral", take, f"ставка: решения нет → {what} → {bank or 'взгляда банка нет'}"
                       + (f" → {scen}" if scen else ""), [], [],
                       [_rule(FLOAT_CHECK)], when)
    if "депозиты" in products:
        how = f": {money(c_dep)}" if c_dep else ""
        if decision == "hold":
            take = f"Доходность депозитов сохраняется высокой: ставка {lvl or 'без изменений'}" + (f", {meet}" if meet else "")
            sign = "opportunity"
        elif decision in ("cut", "hike"):
            take = f"Доходность новых депозитов {'снизится' if decision == 'cut' else 'вырастет'}{how}"
            sign = "risk" if decision == "cut" else "opportunity"
        else:
            take, sign = "Доходность депозитов следует за ключевой ставкой" + (f"; {bank}" if bank else ""), "neutral"
        return Channel("deposits", "deposits", sign, {"treasurer": take, "cfo": take}, f"ставка → депозиты{how.replace(':', ' →')}",
                       [], [], [_rule("Сопоставить сроки размещения депозитов с прогнозом банка по ставке")], when)
    cut = decision == "cut"
    take = ("Кредит для покупателей " + ("становится доступнее: возможен рост спроса, финансируемого кредитом" if cut else
            "остаётся дорогим: роста спроса, финансируемого кредитом, ожидать не стоит")) if "demand" in by else \
        ("Кредитов в Альфа-Банке у компании нет: ставка влияет косвенно — через стоимость заимствований покупателей "
         "и поставщиков")
    return Channel("macro", "buyers" if "demand" in by else "market", "neutral", {"*": take},
                   "ставка → покупатели и поставщики", when=when)


def _fx(by: dict, p: dict, calcs: list[dict]) -> Channel | None:
    c = _calc(calcs, "Валютные расчёты")
    if "fx" not in by and not c:
        return None
    names = ", ".join(country_name(x) for x in p["foreign_trade"]["countries"][:3]) or "зарубежные партнёры"
    if not c:
        return Channel("fx", "fx", "neutral", {"*": f"Изменение курса влияет на расчёты с зарубежными контрагентами ({names}); "
                       f"{NO_FIGURE} — проверьте валюту ближайших платежей"}, f"курс → расчёты ({names})", [], [],
                       [_rule("Сверить график валютных платежей на ближайший месяц")])
    weaker = not c["value"].startswith(("−", "-"))
    move = "Ослабление рубля" if weaker else "Укрепление рубля"
    take = {"treasurer": f"{move}: платежи по контрактам ({names}) — {money(c)}",
            "cfo": f"{move}: валютные закупки — {money(c)}; для экспорта эффект обратный",
            "ceo": f"{move}: рублёвая стоимость импорта {'растёт' if weaker else 'снижается'} ({names})",
            "accountant": f"{move}: переоценка обязательств и авансов в валюте — курсовые {'убытки' if weaker else 'доходы'} ({names})"}
    risks = [_rule("Рост рублёвой стоимости импорта, если контракты не защищены от курса", strength="medium")] if weaker else []
    opps = [] if weaker else [_rule("Снижение рублёвой стоимости импорта — благоприятный момент для валютных закупок", strength="low")]
    return Channel("fx", "fx", c["sign"], take, f"курс → расчёты ({names}) → {money(c)}", risks, opps,
                   [_rule("Сверить график валютных платежей на ближайший месяц и решить, фиксировать ли курс",
                          cta={"label": "Обсудить валютные платежи с менеджером", "kind": "manager"})])


def _sanctions(by: dict) -> Channel | None:
    r = next(iter(by.get("sanctions", [])), None)
    if not r:
        return None
    if r.get("country"):
        name = country_name(r["country"])
        if r.get("eased"):
            return Channel("sanctions", "fx", "opportunity",
                           {"*": f"Ограничения в торговле ({name}) смягчаются: расчёты и поставки могут упроститься"},
                           f"ограничения ↓ → торговля ({name})", [],
                           [_rule(f"Возврат к прежним маршрутам платежей и поставок ({name}), если они дешевле", strength="low")],
                           [_rule(f"Уточнить у партнёров ({name}), какие ограничения сняты")])
        take = {"treasurer": f"Ограничения в торговле ({name}): возможны задержки платежей — проверьте платёжные маршруты",
                "ceo": f"Ограничения в торговле ({name}): заложите резерв по срокам поставок",
                "cfo": f"Ограничения в торговле ({name}): возможны задержки платежей и рост расходов на расчёты",
                "accountant": f"Ограничения в торговле ({name}): проверьте документы валютного контроля и сроки "
                              "расчётов по контрактам"}
        return Channel("sanctions", "fx", "risk", take, f"ограничения → торговля ({name})",
                       [_rule(f"Задержки платежей и поставок по контрактам ({name})", strength="medium")], [],
                       [_rule(f"Проверить контракты и маршруты платежей с партнёрами ({name})",
                              cta={"label": "Обсудить платежи с менеджером", "kind": "manager"})])
    goods = plain(r.get("sector"))
    if r.get("eased"):
        return Channel("sanctions", "market", "opportunity", {"*": f"Ограничения на продукцию отрасли «{goods}» смягчаются: "
                       "рынок сбыта может расшириться"}, f"ограничения ↓ → продукция ({goods})")
    return Channel("sanctions", "market", "risk", {"*": f"Ограничения затрагивают продукцию отрасли «{goods}»: возможны "
                   "изменения спроса и цен — оцените, каких покупателей это коснётся"}, f"ограничения → продукция ({goods})",
                   [_rule(f"Снижение спроса или цен на продукцию ({goods}) из-за ограничений", strength="medium")])


def _tax(text: str, calcs: list[dict], today: date) -> Channel | None:
    c = _calc(calcs, "Утильсбор", "Пошлина", "Налог")
    vat = VAT_RE.search(text)
    if not c and not vat:
        return None
    when = _when_in(text, today)
    if vat:
        a, b = int(vat.group(1)), int(vat.group(2))
        up = impact.num(((100 + b) / (100 + a) - 1) * 100, 1)
        pre = "Если решение примут: " if impact._modal(impact._low(vat.group(0) + " " + text[:200])) else ""
        tail = f" {when}" if when else ""
        take = {"cfo": f"{pre}НДС {b}% вместо {a}%{tail} — при тех же ценах без НДС цены для покупателей вырастут "
                       f"на {up}%: пересмотрите цены в договорах",
                "ceo": f"{pre}НДС {b}% вместо {a}%{tail}: цены для покупателей с учётом НДС вырастут на {up}% — "
                       "возможно снижение спроса",
                "treasurer": f"{pre}НДС {b}% вместо {a}%{tail}: платежи по НДС вырастут — заложите их в график платежей",
                "accountant": f"{pre}НДС {b}% вместо {a}%{tail}: потребуется перенастройка учёта и первичных документов; "
                              "проверьте переходящие договоры и полученные авансы"}
        return Channel("tax", "taxes", "risk", take, f"НДС {a}% → {b}%{tail} → цены с НДС +{up}%",
                       [_rule(f"Цены с НДС для покупателей вырастут на {up}%", strength="medium")], [],
                       [_rule("Пересчитать цены в договорах с покупателями под новую ставку НДС" + tail)], when)
    name = _base_title(c["title"])
    tail = f" — {when}" if when else ""
    take = {"cfo": f"{name}: {money(c)}{tail}; заложите в цены и бюджет",
            "ceo": f"{name} меняется{tail}: {money(c)} — пересмотрите цены",
            "treasurer": f"{name}: {money(c)}{tail}",
            "accountant": f"{name}{tail}: учтите в себестоимости и платежах в бюджет — {money(c)}"}
    risks = [_rule(f"{name}: рост расходов — {money(c)}", strength="medium")] if c["sign"] == "risk" else []
    opps = [_rule(f"{name}: снижение расходов — {money(c)}", strength="medium")] if c["sign"] == "opportunity" else []
    return Channel("tax", "taxes", c["sign"], take, f"{name} → {money(c)}{tail}", risks, opps,
                   [_rule(f"Заложить изменение ({name.lower()}) в цены и бюджет{tail}")], when)


def _buyers(by: dict, p: dict, calcs: list[dict]) -> Channel | None:
    r = next(iter(by.get("customers", [])), None)
    c = _calc(calcs, "Выручка от покупателей", "Выручка под влиянием")
    if not r and not c:
        return None
    share = r.get("share", 0) if r else 0
    whom = who(r["sector"]) if r else c["title"].split(" — ", 1)[-1]
    pct = f"{round(share * 100)}%" if share else ""
    line = f"{_cap(whom)} — {pct} ваших поступлений" if pct else _cap(whom)
    if c and not c["value"].startswith("≈"):
        down = c["value"].startswith(("−", "-"))
        take = {"ceo": f"{line}: спрос в их отрасли {'снижается' if down else 'растёт'} — {money(c)}", "cfo": f"{line}: {money(c)}"}
        if down and share >= RISK_MIN_SHARE:
            take["treasurer"] = f"{line}: снижение продаж у покупателей увеличивает сроки оплаты — проверьте дебиторскую задолженность"
            take["accountant"] = f"{line}: возможен рост просроченной дебиторской задолженности — проверьте резерв по сомнительным долгам"
        sign = "risk" if down else "opportunity"
    else:
        rub = c["value"] if c else (f"≈ {impact.rub(p['revenue_rub'] * share)} в год" if share else "")
        figures = ", ".join(x for x in (f"{pct} поступлений" if pct else "", rub) if x)
        line = f"Покупатели — {whom}" + (f" ({figures})" if figures else "")
        take = {"*": f"{line}: событие в их отрасли, {NO_FIGURE} — отслеживайте объём заказов и сроки оплаты"}
        if share >= RISK_MIN_SHARE:
            take["treasurer"] = f"{line}: событие в их отрасли — контролируйте сроки оплаты и дебиторскую задолженность"
        sign = "neutral"
    risky = sign == "risk" and share >= RISK_MIN_SHARE
    return Channel("buyers", "buyers", sign, take, f"{whom} ({pct or 'покупатели'})" + (f" → {money(c)}" if c else ""),
                   [_rule(f"Задержки оплат от покупателей ({whom}) — {pct} ваших поступлений", strength="medium")] if risky else [],
                   [], [_rule(f"Проверить дебиторскую задолженность покупателей ({whom}) и лимиты отсрочки",
                              cta={"label": "Узнать о факторинге", "kind": "product"})] if risky else [])


def _suppliers(by: dict, calcs: list[dict]) -> Channel | None:
    r = next(iter(by.get("suppliers", [])), None)
    c = _calc(calcs, "Закупки", "Маржа")
    if not r and not c:
        return None
    whom = who(r["sector"]) if r else ""
    pct = f"{round(r['share'] * 100)}%" if r and r.get("share") else ""
    if not c:
        return Channel("suppliers", "suppliers", "neutral",
                       {"*": f"Поставщики — {whom} ({pct} платежей): событие в их отрасли, {NO_FIGURE} — проверьте цены "
                             "и сроки в договорах"}, f"{whom} ({pct} платежей)", [], [],
                       [_rule("Проверить цены и сроки в договорах с поставщиками на следующий квартал")])
    name, up = _base_title(c["title"]), c["sign"] == "risk"
    take = {"cfo": f"{name}: {money(c)}", "treasurer": f"{name}: {money(c)}",
            "ceo": f"Себестоимость {'растёт' if up else 'снижается'}: {money(c)}" + (" — пересмотрите цены продажи" if up else "")}
    return Channel("suppliers", "suppliers", c["sign"], take, (f"{whom} ({pct} платежей)" if whom else "закупки") + f" → {money(c)}",
                   [_rule(f"Рост закупочных цен — {money(c)}", strength="medium")] if up else [],
                   [] if up else [_rule(f"Снижение закупочных цен — {money(c)}", strength="medium")],
                   [_rule("Проверить цены и сроки в договорах с поставщиками на следующий квартал")])


def _market(by: dict, p: dict, calcs: list[dict], text: str = "") -> Channel | None:
    rs = by.get("own", []) + by.get("demand", [])
    c = _calc(calcs, "Выручка, если", "Выручка:")
    if not rs and not c:
        return None
    r = next((x for x in rs if x.get("role") != "group"), rs[0] if rs else None)
    role = (r or {}).get("role", "own")
    sector = plain(r["sector"]) if r and r.get("sector") else plain(p["okved2"])
    if role == "group":
        return Channel("group", None, "neutral", {"*": f"Событие касается отрасли компаний группы («{_cap(sector)}»), а не компании "
                       "напрямую: информация для руководства группы"}, f"отрасль группы ({sector})")
    where = f"ваш рынок («{_cap(sector)}»)" if role == "own" else f"рынок вашей продукции («{_cap(sector)}»)"
    loc = f"в вашей отрасли («{_cap(sector)}»)" if role == "own" else f"на рынке вашей продукции («{_cap(sector)}»)"
    if c:
        take = {"ceo": f"{_cap(where)}: {money(c)}, если ваши продажи изменятся вместе с рынком",
                "cfo": f"{_cap(where)}: ориентир по выручке — {money(c)}"}
        return Channel("market", "market", c["sign"], take, f"{where} → {money(c)}")
    if r and r.get("kind") == "demand":
        return Channel("market", "market", "neutral", {"*": f"Изменение потребительского спроса; {NO_FIGURE} — сопоставьте "
                       "с динамикой собственных продаж"}, "спрос покупателей")
    kind, take, act = market_kind(text)
    return Channel("market", "market", "neutral", {"*": take.format(where=where, loc=loc)}, f"{where}: {kind}", [], [],
                   [_rule(act)] if act else [])


# Новость о рынке без чисел: что именно меняется — по словам новости. Направление цены код не угадывает:
# говорим, через что событие дойдёт до денег и что проверить
MARKET_KINDS = (
    ("регулирование", re.compile(r"фас|антимонопол|закон|законопроект|запрет|штраф|правительств|минфин|минэнерго|"
                                 r"минпромторг|регулир|госдум|постановлени|маркировк|лиценз", re.I),
     "Изменения регулирования {loc}: оцените влияние на цены, договоры и отчётность",
     "Оценить с юристом влияние новых требований на договоры и цены"),
    ("цены", re.compile(r"(?<![а-яё])цен|подорож|подешев|стоимост|котировк|тариф", re.I),
     "Изменение цен {loc}: сопоставьте с ценами закупок и реализации — расхождение влияет на маржу",
     "Сопоставить цены закупок и реализации с рыночными"),
    ("предложение", re.compile(r"поставк|экспорт|импорт|добыч|производств|выпуск|запас|мощност|завод|рудник|урожа", re.I),
     "Изменение предложения {loc} влияет на цены: сопоставьте с ценами закупок и реализации", ""),
    ("спрос", re.compile(r"спрос|продаж|потреблен|заказ", re.I),
     "Изменение спроса {loc}: сопоставьте с динамикой собственных продаж и заказов", ""),
)


def market_kind(text: str) -> tuple[str, str, str]:
    """(вид изменения, вывод с местами для {where} / {loc}, действие) для новости о рынке без чисел — по началу текста."""
    head = text[:400]
    for kind, rx, take, act in MARKET_KINDS:
        if rx.search(head):
            return kind, take, act
    return "без числа", "Событие {loc}; " + NO_FIGURE + " — сопоставьте с динамикой продаж и цен", ""


# ---------- сборка ----------
def build(event: dict, match: Match, p: dict, calcs: list[dict], today: date,
          meetings: list[date] | None = None) -> list[Channel]:
    """Каналы влияния события на компанию: сначала со знаком (риск или возможность), потом остальные."""
    by: dict[str, list[dict]] = {}
    for r in match.reasons:
        if r.get("weight", 0) > 1 and r.get("kind"):
            by.setdefault(r["kind"], []).append(r)
    text = (event.get("title", "") + ". " + event.get("text", "")).strip()
    # следующее заседание — после самого события: новость о решении 23 октября ведёт к декабрьскому заседанию
    published = (event.get("published") or "")[:10]
    ref = max(today, date.fromisoformat(published)) if published else today
    meeting = next_meeting(meetings or [], ref)
    rate = None
    if {"debt", "deposits", "macro"} & set(by) or _calc(calcs, "Стоимость долга", "Доход по депозитам") or \
            ("demand" in by and "ставк" in text.lower()):
        rate = _rate(text, by, p, calcs, today, meeting, rate_decision(event))
    out = [ch for ch in (rate, _tax(text, calcs, today), _fx(by, p, calcs), _sanctions(by), _buyers(by, p, calcs),
                         _suppliers(by, calcs), _market(by if not rate else {**by, "demand": []}, p, calcs, text)) if ch]
    return sorted(out, key=lambda ch: (ch.sign == "neutral", ch.id == "group"))


def takeaways(channels: list[Channel]) -> dict[str, str]:
    """Главный вывод под роль: первый по важности для роли канал, у которого есть вывод для неё."""
    out = {}
    for role in ROLES:
        ranked = sorted(channels, key=lambda ch: (ch.sign == "neutral", ORDER[role].index(ch.id)))
        out[role] = next((t for ch in ranked if (t := ch.for_role(role))), "")
    return out


def statements(channels: list[Channel], limit: int = 2) -> dict[str, list[dict]]:
    """Риски, возможности и действия краткого разбора — не больше двух каждого, без повторов."""
    out: dict[str, list[dict]] = {"risks": [], "opportunities": [], "actions": []}
    signed = [ch for ch in channels if ch.sign != "neutral"]
    for ch in signed or channels:  # есть риск или возможность — действия из каналов без числа их не разбавляют
        for k, items in out.items():
            for s in getattr(ch, k):
                if len(items) < limit and s["t"] not in {x["t"] for x in items}:
                    items.append(s)
    return out


def card_type(channels: list[Channel], calcs: list[dict]) -> str:
    """Тип карточки — по главному каналу со знаком: снижение ставки для заёмщика — возможность, даже если доход по его
    депозитам тоже снизится (это видно в расчётах). Каналов со знаком нет — по расчётам."""
    main = next((ch.sign for ch in channels if ch.sign != "neutral"), None)
    if main:
        return main
    signs = [c["sign"] for c in calcs]
    return "risk" if "risk" in signs else "opportunity" if "opportunity" in signs else "info"


def as_text(channels: list[Channel]) -> str:
    """Каналы по строке на канал — в трассу и в задание агенту."""
    return "\n".join(f"- {ch.chain}" + (f" ({ch.when})" if ch.when and ch.when not in ch.chain else "") for ch in channels)
