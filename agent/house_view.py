"""Взгляд банка (house view): действующие прогнозы Альфа-Банка по ставке, инфляции, курсу и ВВП.

Зачем: банк пересматривает прогнозы. 31 июля он ждал ставку 13,5% на конец 2026 года, 11 сентября —
14,0%. Поиск по словам (agent/kb.py) находит оба документа и не знает, какой из них действует; модель
могла выдать старый прогноз за позицию банка. Здесь решение принимает код: действует самый свежий
прогноз по тому же показателю и горизонту, более ранние другие значения — «пересмотрено».

Как устроено:
1. scripts/build_house_view.py без сети и модели вынимает прогнозы из data/bank/docs.json и постов
   экономистов банка → data/bank/house_view.json: {indicator, horizon, value, unit, doc_id, doc_title,
   date, page, quote, superseded_by, manual}. Цитата дословная и проверяется тестом.
2. current() — самый свежий прогноз не старше 12 месяцев; superseded() — более ранние прогнозы с другим
   значением; trail() — строка пересмотра для руководителя; as_context() — блок для промпта черновика.
3. revisions_in(text) — повторяет ли текст пересмотренное значение рядом со словами показателя.
   Этим пользуются поиск (пометка фрагмента) и проверка карточки (ошибка «позиция банка пересмотрена»).
4. revised_prefix() и divergence() — что видит модель: пометка «[ПЕРЕСМОТРЕНО 11.09 → 14,0% …]» у фрагмента
   и решение кода в шаге трассы «источники расходятся: … ; берём более свежий» (agent/tools.py).
Значения 87 и 87,2 ₽ — один прогноз (округление в тексте), а не пересмотр: допуск TOL.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "bank" / "house_view.json"
STALE_DAYS = 365
INDICATORS = ("key_rate", "inflation", "usd_rub", "cny_rub", "gdp")
LABELS = {"key_rate": "Ключевая ставка", "inflation": "Инфляция", "usd_rub": "Курс доллара",
          "cny_rub": "Курс юаня", "gdp": "Рост ВВП"}
SHORT = {"key_rate": "ставка", "inflation": "инфляция", "usd_rub": "курс доллара", "cny_rub": "курс юаня", "gdp": "рост ВВП"}
# Слова показателя: число рядом с ними относится к этому показателю
WORDS = {"key_rate": re.compile(r"ставк|\bКС\b", re.I), "inflation": re.compile(r"инфляц", re.I),
         "usd_rub": re.compile(r"руб\.?\s?/\s?\$|доллар|\bUSD\b|курс\w* рубл", re.I),
         "cny_rub": re.compile(r"юан|\bCNY\b", re.I), "gdp": re.compile(r"\bВВП\b")}
# Ставки по депозитам, ипотеке, купонам и налогам — не ключевая ставка
NOT_KEY_RATE = re.compile(r"депозит|вклад|ипотек|ипотечн|купон|доходност|ОФЗ|НДС|налог|пошлин|по кредит|кредитн|аренд", re.I)
TOL = {"usd_rub": 0.5, "cny_rub": 0.1}  # курс в тексте округляют до рубля; проценты сравниваются до десятых
NEAR = 150  # слово показателя — не дальше этого числа знаков от значения
MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря")
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[А-ЯЁA-Z«\"(])|\n+")
PART_SPLIT = re.compile(r",\s+|;\s+|:\s+|\s+[–—-]\s+")
NUM = re.compile(r"(?<![\d,.])\d{1,3}(?:[.,]\d{1,2})?(?![\d])")
RANGE = re.compile(r"\d\s?[–—-]\s?$")  # «11,5–13,5%» — диапазон, а не прогноз-точка

_cache: dict[str, tuple[float, list[dict]]] = {}


def load(path: Path | None = None) -> list[dict]:
    """Записи взгляда банка. Нет файла — пустой список (честно: прогнозов нет). Перечитывается при изменении файла."""
    path = path or PATH
    if not path.exists():
        return []
    mt = path.stat().st_mtime
    hit = _cache.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    entries = json.loads(path.read_text(encoding="utf-8"))
    _cache[str(path)] = (mt, entries)
    return entries


def same(indicator: str, a: float, b: float) -> bool:
    """Одно и то же значение прогноза с учётом округления в тексте."""
    return abs(a - b) < TOL.get(indicator, 0.05)


def fmt_value(value: float, unit: str) -> str:
    """14.0, '%' → «14,0%»; 93.0, '₽' → «93,0 ₽»."""
    s = f"{value:.1f}".replace(".", ",")
    return f"{s}%" if unit == "%" else f"{s} {unit}"


def short_title(title: str) -> str:
    """«Ключевая ставка ЦБ РФ: ставка без изменения…» → «Ключевая ставка ЦБ РФ»: в промпте хватает начала названия."""
    head = title.split(":")[0].strip()
    return head if len(head) <= 60 else head[:57].rsplit(" ", 1)[0] + "…"


def day(iso: str) -> str:
    """«2026-09-11» → «11.09»."""
    return f"{iso[8:10]}.{iso[5:7]}"


def human_date(iso: str, year: bool = False) -> str:
    """«2026-09-11» → «11 сентября» (с годом — «11 сентября 2026»)."""
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS_GEN[d.month - 1]}" + (f" {d.year}" if year else "")


def horizon_text(horizon: str, indicator: str = "") -> str:
    """«end-2026» → «на конец 2026» (рост ВВП — «за 2026 год»); «2027-04» → «до апреля 2027»."""
    if horizon.startswith("end-"):
        return f"за {horizon[4:]} год" if indicator == "gdp" else f"на конец {horizon[4:]}"
    y, m = horizon.split("-")
    return f"до {MONTHS_GEN[int(m) - 1]} {y}"


def _year(horizon: str) -> str:
    return horizon[4:] if horizon.startswith("end-") else horizon[:4]


def _active(entries: list[dict], indicator: str, horizon: str, today: date) -> list[dict]:
    """Записи показателя и горизонта, известные на дату today и не старше 12 месяцев; от старых к новым."""
    out = [e for e in entries if e["indicator"] == indicator and e["horizon"] == horizon and e["date"] <= today.isoformat()
           and (today - date.fromisoformat(e["date"])).days <= STALE_DAYS]
    return sorted(out, key=lambda e: e["date"])  # сортировка устойчива: в один день — порядок файла (текст раньше таблицы)


def current(indicator: str, horizon: str, today: date | None = None, entries: list[dict] | None = None) -> dict | None:
    """Действующий прогноз: самый свежий не старше 12 месяцев. В один день — первая запись (текст, а не таблица)."""
    act = _active(load() if entries is None else entries, indicator, horizon, today or date.today())
    if not act:
        return None
    newest = act[-1]["date"]
    return next(e for e in act if e["date"] == newest)


def superseded(indicator: str, horizon: str, today: date | None = None, entries: list[dict] | None = None) -> list[dict]:
    """Более ранние прогнозы того же показателя и горизонта с другим значением — банк их пересмотрел."""
    entries = load() if entries is None else entries
    cur = current(indicator, horizon, today, entries)
    if not cur:
        return []
    return [e for e in _active(entries, indicator, horizon, today or date.today())
            if e["date"] < cur["date"] and not same(indicator, e["value"], cur["value"])]


def trail(indicator: str, horizon: str, today: date | None = None, entries: list[dict] | None = None) -> str:
    """Строка пересмотра: «31 июля: 13,5% → 11 сентября: 14,0% (действует)». Пересмотров нет — пустая строка.

    Подряд идущие равные значения склеиваются: дата — первого прогноза, значение — последнего (точнее записано)."""
    act = _active(load() if entries is None else entries, indicator, horizon, today or date.today())
    groups: list[list[dict]] = []
    for e in act:
        if groups and same(indicator, groups[-1][-1]["value"], e["value"]):
            groups[-1].append(e)
        else:
            groups.append([e])
    if len(groups) < 2:
        return ""
    years = len({g[0]["date"][:4] for g in groups}) > 1
    parts = [f"{human_date(g[0]['date'], years)}: {fmt_value(g[-1]['value'], g[-1]['unit'])}" for g in groups]
    return " → ".join(parts) + " (действует)"


def keys(entries: list[dict] | None = None) -> list[tuple[str, str]]:
    """Пары (показатель, горизонт) в порядке INDICATORS, затем горизонта."""
    entries = load() if entries is None else entries
    return sorted({(e["indicator"], e["horizon"]) for e in entries}, key=lambda k: (INDICATORS.index(k[0]), k[1]))


def source_label(cur: dict, sources: list, entries: list[dict]) -> str:
    """Метка источника сессии (S…), во фрагменте которого стоит действующий прогноз: тот же документ и дословная
    цитата любой записи этого прогноза (текст или таблица). Нет такого фрагмента — пустая строка."""
    same_day = [e for e in entries if (e["indicator"], e["horizon"], e["date"]) == (cur["indicator"], cur["horizon"], cur["date"])
                and same(cur["indicator"], e["value"], cur["value"])]
    for s in sources:
        if getattr(s, "kind", "") == "bank" and any(getattr(s, "doc", "") == e["doc_title"] and norm(e["quote"]) in norm(s.text)
                                                     for e in same_day):
            return s.label
    return ""


def as_context(today: date | None = None, entries: list[dict] | None = None, sources: list | None = None) -> str:
    """Короткий блок для промпта: действующие прогнозы банка с датой и документом, пересмотренные — отдельно.
    sources — источники сессии агента: у строки, чей фрагмент среди них, стоит метка «источник S2»."""
    entries = load() if entries is None else entries
    lines = []
    for ind, hor in keys(entries):
        cur = current(ind, hor, today, entries)
        if not cur:
            continue
        label = source_label(cur, sources, entries) if sources else ""
        line = (f"- {LABELS[ind]} {horizon_text(hor, ind)}: {fmt_value(cur['value'], cur['unit'])} "
                f"({day(cur['date'])}.{cur['date'][:4]}, «{short_title(cur['doc_title'])}»{', источник ' + label if label else ''})")
        old: dict[str, list[str]] = {}
        for e in superseded(ind, hor, today, entries):
            dates = old.setdefault(fmt_value(e["value"], e["unit"]), [])
            if day(e["date"]) not in dates:
                dates.append(day(e["date"]))
        line += "".join(f"; прежний прогноз {v} ({', '.join(d)}) пересмотрен — не используй его" for v, d in old.items())
        lines.append(line)
    if not lines:
        return ""
    return ("Текущий взгляд Альфа-Банка (самые свежие прогнозы банка; при расхождении источников действует более свежий):\n"
            + "\n".join(lines))


def _sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def _numbers(sent: str):
    """(значение, позиция) чисел предложения; числа из диапазонов «11,5–13,5%» пропускаются."""
    for m in NUM.finditer(sent):
        if RANGE.search(sent[:m.start()]) or re.match(r"\s?[–—-]\s?\d", sent[m.end():]):
            continue
        yield float(m.group(0).replace(",", ".")), m.start()


def _near_word(sent: str, indicator: str, pos: int) -> bool:
    lo, hi = max(0, pos - NEAR), pos + 60
    return bool(WORDS[indicator].search(sent[lo:hi]))


def _other_rate(sent: str, pos: int) -> bool:
    """Ближайшая к числу «ставка» — не ключевая: «ставки по депозитам — около 13,5%», «ипотечная ставка», «ставка НДС».
    Уточнение стоит рядом со словом, а не обязательно в той же части предложения, что и число."""
    words = list(WORDS["key_rate"].finditer(sent))
    if not words:
        return False
    w = min(words, key=lambda m: abs(m.start() - pos))
    if re.search(r"ключев\w*\s+$", sent[:w.start()], re.I):
        return False  # «ключевой ставки» — она и есть, что бы ни стояло рядом
    return bool(NOT_KEY_RATE.search(sent[max(0, w.start() - 20):w.end() + 25]))


def _clause(sent: str, pos: int) -> str:
    """Часть предложения между запятыми, точками с запятой и тире, в которой стоит число."""
    lo, hi = 0, len(sent)
    for m in PART_SPLIT.finditer(sent):
        if m.end() <= pos:
            lo = m.end()
        elif m.start() >= pos:
            hi = m.start()
            break
    return sent[lo:hi]


def revisions_in(text: str, today: date | None = None, before: str | None = None,
                 entries: list[dict] | None = None) -> list[dict]:
    """Пересмотренные прогнозы, которые текст повторяет как действующие.

    Совпадение — значение пересмотренного прогноза рядом со словами показателя (ставка, инфляция, курс, ВВП).
    Не считается: в том же предложении рядом есть действующее значение («раньше 13,5%, теперь 14,0%» — это и есть
    пересмотр); часть предложения с этим числом называет другой год; число из диапазона; «ставка» по депозитам,
    ипотеке, купонам. before — дата фрагмента: материал не старше действующего прогноза не помечается.
    Возвращает [{indicator, horizon, old (значение), current (запись)}] без повторов."""
    entries = load() if entries is None else entries
    found: dict[tuple, dict] = {}
    for ind, hor in keys(entries):
        old = superseded(ind, hor, today, entries)
        if not old:
            continue
        cur = current(ind, hor, today, entries)
        if before and before >= cur["date"]:
            continue
        for sent in _sentences(text):
            if not WORDS[ind].search(sent):
                continue
            nums = list(_numbers(sent))
            for v, pos in nums:
                hit = next((e for e in old if same(ind, e["value"], v)), None)
                if not hit or not _near_word(sent, ind, pos):
                    continue
                part = _clause(sent, pos)
                years = set(re.findall(r"\b20\d\d\b", part))
                if (years and _year(hor) not in years) or (ind == "key_rate" and (NOT_KEY_RATE.search(part) or _other_rate(sent, pos))):
                    continue
                if any(same(ind, cur["value"], w) and abs(p - pos) <= 200 for w, p in nums):
                    continue
                found.setdefault((ind, hor), {"indicator": ind, "horizon": hor, "old": hit["value"], "current": cur})
    return list(found.values())


def revision_note(rev: dict) -> str:
    """«14,0% (ставка на конец 2026)» — что банк ждёт сейчас вместо пересмотренного значения."""
    cur = rev["current"]
    return f"{fmt_value(cur['value'], cur['unit'])} ({SHORT[rev['indicator']]} {horizon_text(rev['horizon'], rev['indicator'])})"


def revised_label(revs: list[dict]) -> str:
    """Пометка фрагмента поиска: «пересмотрено 11.09: банк ждёт 14,0% (ставка на конец 2026)»."""
    by_day: dict[str, list[str]] = {}
    for r in revs:
        by_day.setdefault(day(r["current"]["date"]), []).append(revision_note(r))
    return "; ".join(f"пересмотрено {d}: банк ждёт {', '.join(n)}" for d, n in by_day.items())


def revised_prefix(revs: list[dict]) -> str:
    """Пометка фрагмента в тексте для модели: «[ПЕРЕСМОТРЕНО 11.09 → 14,0% (ставка на конец 2026)]»."""
    return "[ПЕРЕСМОТРЕНО " + "; ".join(f"{day(r['current']['date'])} → {revision_note(r)}" for r in revs) + "]"


def divergence(revs: list[dict]) -> str:
    """Решение кода, когда в выдаче поиска есть и пересмотренный, и действующий прогноз:
    «источники расходятся: ставка на конец 2026 — 13,5% (31.07) и 14,0% (11.09); берём более свежий».

    revs — [{indicator, horizon, old, current, date}], date — дата фрагмента со старым значением.
    Нечего сравнивать — пустая строка."""
    groups: dict[tuple, dict] = {}
    for r in revs:
        g = groups.setdefault((r["indicator"], r["horizon"]), {"cur": r["current"], "old": {}})
        dates = g["old"].setdefault(r["old"], set())
        if r.get("date"):
            dates.add(r["date"])
    parts = []
    for (ind, hor), g in sorted(groups.items(), key=lambda x: (INDICATORS.index(x[0][0]), x[0][1])):
        cur = g["cur"]
        olds = " и ".join(fmt_value(v, cur["unit"]) + (f" ({', '.join(day(d) for d in sorted(ds))})" if ds else "")
                          for v, ds in sorted(g["old"].items(), key=lambda x: min(x[1], default="")))
        parts.append(f"{SHORT[ind]} {horizon_text(hor, ind)} — {olds} и {fmt_value(cur['value'], cur['unit'])} "
                     f"({day(cur['date'])})")
    return ("источники расходятся: " + "; ".join(parts) + "; берём более свежий") if parts else ""


def norm(text: str) -> str:
    """Для сверки цитат: ё → е, пробелы схлопнуты."""
    return re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip()
