"""Взгляд банка: прогнозы Альфа-Банка из его документов и постов экономистов → data/bank/house_view.json.

    python scripts/build_house_view.py      # пересобрать файл и напечатать таблицу для ручной сверки

Без сети и без модели: регулярные выражения по data/bank/docs.json и постам каналов банка (data/news).
Повторный запуск на тех же данных даёт тот же файл байт в байт.

Что берём (только то, что разбирается однозначно):
1. Предложения с голосом банка («мы ожидаем», «наш прогноз», «прогноз … остается»): в одной части предложения
   (между запятыми) ровно одно значение и ровно один горизонт — «на конец 2026 г.», «по итогам текущего года»,
   «до конца года». Части с «раньше», «изначально», «в среднем», «м/м», «диапазон» пропускаются: это прежний
   прогноз, среднее или месячные данные. Предложения с консенсусом аналитиков и прогнозом ЦБ — не позиция банка.
2. Строки таблицы «Основные макроэкономические показатели» (столбцы 2026П, 2027П): строка берётся, только если
   после названия стоит ровно столько чисел, сколько лет в шапке. Испорченные распознаванием строки
   («Курс, конец года, руб./юань 10,1 fa Rea …» в обзоре 1П26) не разбираются и не попадают в файл.
3. Посты экономистов банка (каналы с bank_voice в data/sources.json, в тексте — главный экономист):
   прогноз ставки «до апреля 2027 года». На горизонте, который уже есть в документах, пост добавляется,
   только если меняет значение (иначе формулировка документа точнее).
Что не разобралось, но нужно, — в MANUAL вручную (manual: true); такие цитаты проходят ту же сверку.

Цитата — часть предложения с прогнозом, если она понятна сама по себе (голос банка, слово показателя, горизонт),
иначе всё предложение. Сверка (verify): цитата дословно стоит на своей странице или в посте и сама называет
значение со словом показателя; не прошла — скрипт останавливается, файл не пишется.

Пересмотр: тот же показатель и горизонт, более поздняя дата, другое значение (с допуском на округление
87 ↔ 87,2 ₽) → у старой записи superseded_by = документ действующего прогноза.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск «python scripts/build_house_view.py» из корня репозитория
    sys.path.insert(0, str(ROOT))

from agent.house_view import INDICATORS, MONTHS_GEN, PART_SPLIT, WORDS, norm, same  # noqa: E402

DOCS = ROOT / "data" / "bank" / "docs.json"
OUT = ROOT / "data" / "bank" / "house_view.json"
EVENTS = (ROOT / "data" / "news" / "events.sample.json", ROOT / "data" / "news" / "events.json")
SOURCES = ROOT / "data" / "sources.json"

UNITS = {"key_rate": "%", "inflation": "%", "gdp": "%", "usd_rub": "₽", "cny_rub": "₽"}
VOICE = re.compile(r"\b(?:мы|наш\w*)\b|ожидаем|жд[её]м|прогноз", re.I)
NOT_BANK = re.compile(r"консенсус|аналитик|прогноз\w* ЦБ|по прогнозу ЦБ|Минэк", re.I)
SKIP_PART = re.compile(r"раньше|ранее|изначально|первоначально|предыдущ|вместо|средн|м/м|н/н|кв/кв|диапазон", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"(])|\n+")
# Части длинного предложения для цитаты: «…по данным 1П26: мы понизили прогноз…», «…года, На конец года мы ожидаем…»
# (распознавание склеило два предложения через запятую). Каждая такая граница — и граница PART_SPLIT.
SEG_SPLIT = re.compile(r";\s+|:\s+|\s+[–—-]\s+|,\s+(?=[А-ЯЁ][а-яё])")
PCT = re.compile(r"(?<![\d,.])(-?(?:0|[1-9]\d?)(?:,\d{1,2})?)\s?%")          # «13,5%», но не «01%» (ошибка OCR)
RUB = re.compile(r"(?<![\d,.])([1-9]\d{0,2}(?:,\d{1,2})?)\s?руб\.?\s?/\s?(\$|юан)", re.I)
YEAR_H = re.compile(r"(?:на конец|по итогам|к концу|на|в)\s+(20\d\d)\s*(?:г\.?|год\w*)", re.I)
THIS_YEAR_H = re.compile(r"(?:до конца|на конец|по итогам|к концу)\s+(?:текущего\s+|этого\s+)?года", re.I)
MONTH_H = re.compile(r"до\s+(" + "|".join(MONTHS_GEN) + r")\s+(20\d\d)", re.I)
ECONOMIST = re.compile(r"Орлов\w*|главн\w* экономист\w*|экономист\w* Альфа-Банка|аналитик\w* Альфа-Банка|"
                       r"по (?:мнению|оценке|прогнозу) Альфа-Банка|Альфа-Банк\w* (?:считает|ожидает|прогнозирует|жд[её]т)")
# Строки таблицы прогнозов: название строки → показатель
TABLE_ROWS = {"key_rate": r"Ставка, конец года, %", "inflation": r"Инфляция, % дек\./дек\.",
              "usd_rub": r"Курс, конец года, руб\./\$", "cny_rub": r"Курс, конец года, руб\./юань",
              "gdp": r"Темпы роста ВВП, % г/г"}
TABLE_HEAD = re.compile(r"(?:20\d\d\s*[ПпNn]?\s*\|?\s*){5,}")
CELL = r"-?\d{1,3}(?:,\d{1,2})?%?"

# Ручные записи для значений, которые не разбираются регулярными выражениями. Формат — как у записи файла
# (без superseded_by): {"indicator", "horizon", "value", "unit", "doc_id", "page", "quote"}; цитата — дословно со страницы.
MANUAL: list[dict] = []


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _horizons(text: str, year: str) -> set[str]:
    hs = {f"end-{y}" for y in YEAR_H.findall(text)}
    if THIS_YEAR_H.search(text):
        hs.add(f"end-{year}")
    hs |= {f"{y}-{MONTHS_GEN.index(m.lower()) + 1:02d}" for m, y in MONTH_H.findall(text)}
    return hs


def _indicators(text: str) -> set[str]:
    return {k for k in INDICATORS if WORDS[k].search(text)}


def _values(text: str, indicator: str) -> list[float]:
    if UNITS[indicator] == "%":
        return [_num(v) for v in PCT.findall(text)]
    want = "$" if indicator == "usd_rub" else "юан"
    return [_num(v) for v, cur in RUB.findall(text) if cur.lower() == want]


def _clean(sentence: str) -> str:
    """Цитата — предложение без маркеров списка и значков в начале («Я Мы понижаем…» → «Мы понижаем…»)."""
    s = re.sub(r"^[^\wА-Яа-яЁё«]+", "", sentence.strip())
    s = re.sub(r"^(?:[ЯЧО]|С|О)\s+(?=[А-ЯЁ])", "", s)  # OCR превращает маркеры списка в «Я», «Ч», «О»
    return norm(s)


def from_sentences(text: str, year: str) -> list[tuple[str, str, float, str]]:
    """(показатель, горизонт, значение, цитата) из предложений с голосом банка."""
    out = []
    for sent in SENT_SPLIT.split(text):
        if not VOICE.search(sent) or NOT_BANK.search(sent):
            continue
        sent_inds = _indicators(sent)
        for part in PART_SPLIT.split(sent):
            if SKIP_PART.search(part):
                continue
            inds = _indicators(part) or (sent_inds if len(sent_inds) == 1 else set())
            hs = _horizons(part, year)
            if len(inds) != 1 or len(hs) != 1:
                continue
            ind, hor = inds.pop(), hs.pop()
            vals = _values(part, ind)
            if len(vals) == 1:
                out.append((ind, hor, vals[0], _quote(sent, part, ind, hor, year)))
    return out


def _quote(sent: str, part: str, ind: str, hor: str, year: str) -> str:
    """Цитата — часть предложения, где стоит прогноз, если она сама себя объясняет: голос банка, слово показателя,
    тот же горизонт. «…слабого роста производства, очевидного по данным 1П26: мы понизили наш прогноз … ожидаем
    роста ВВП всего на 0,1% г/г» → цитата начинается с «мы понизили». Иначе — всё предложение."""
    seg = next((x for x in SEG_SPLIT.split(sent) if part in x), sent)
    if seg != sent and VOICE.search(seg) and WORDS[ind].search(seg) and _horizons(seg, year) == {hor}:
        return _clean(seg)
    return _clean(sent)


def from_table(page: str) -> list[tuple[str, str, float, str]]:
    """(показатель, горизонт, значение, строка таблицы) из таблицы прогнозов со столбцами 2026П/2027П."""
    head = TABLE_HEAD.search(page)
    if not head:
        return []
    cols = re.findall(r"(20\d\d)\s*([ПпNn]?)", head.group(0))
    out = []
    for ind, label in TABLE_ROWS.items():
        m = re.search(label + r"((?:\s+" + CELL + r"){" + str(len(cols)) + r"})(?!\s+" + CELL + r"(?:\s|$))", page[head.end():])
        if not m:
            continue
        cells = m.group(1).split()
        for (y, mark), cell in zip(cols, cells, strict=True):
            if mark:  # П — прогноз банка; фактические годы не нужны
                out.append((ind, f"end-{y}", _num(cell.rstrip("%")), norm(m.group(0))))
    return out


def from_docs(docs: list[dict]) -> list[dict]:
    entries = []
    for d in docs:
        if not d.get("date"):
            continue
        for no, page in enumerate(d["pages"], start=1):
            for ind, hor, val, quote in from_sentences(page, d["date"][:4]) + from_table(page):
                entries.append({"indicator": ind, "horizon": hor, "value": val, "unit": UNITS[ind], "doc_id": d["id"],
                                "doc_title": d["title"], "date": d["date"], "page": no, "quote": quote,
                                "url": d.get("url", ""), "manual": False})
    return entries


def bank_voice_names() -> set[str]:
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    return {s["name"] for kind in ("rss", "telegram") for s in src.get(kind, []) if s.get("bank_voice")}


def load_posts() -> list[dict]:
    """Посты из сохранённых событий: образец в git и свежий сбор (свежий сбор — поверх образца)."""
    by_id = {}
    for path in EVENTS:
        if path.exists():
            by_id.update({e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))})
    return [by_id[k] for k in sorted(by_id)]


def from_posts(posts: list[dict], docs_entries: list[dict]) -> list[dict]:
    """Прогнозы экономистов банка из постов его каналов. Цитата — предложение поста дословно."""
    names = bank_voice_names()
    entries = []
    for ev in posts:
        if not (ev.get("bank_voice") or ev.get("source") in names) or not ECONOMIST.search(ev.get("text", "")):
            continue
        dd = ev["published"]
        for sent in SENT_SPLIT.split(ev["text"]):
            if NOT_BANK.search(sent) or SKIP_PART.search(sent):
                continue
            inds, hs = _indicators(sent), _horizons(sent, dd[:4])
            if len(inds) != 1 or len(hs) != 1:
                continue
            ind, hor = inds.pop(), hs.pop()
            vals = _values(sent, ind)
            if len(vals) != 1:
                continue
            docs_same = [e for e in docs_entries if e["indicator"] == ind and e["horizon"] == hor and e["date"] <= dd]
            if docs_same and same(ind, max(docs_same, key=lambda e: e["date"])["value"], vals[0]):
                continue  # документ банка уже говорит то же самое — его формулировка точнее
            entries.append({"indicator": ind, "horizon": hor, "value": vals[0], "unit": UNITS[ind], "doc_id": ev["id"],
                            "doc_title": f"{ev['source']}, пост от {dd[8:10]}.{dd[5:7]}.{dd[:4]}", "date": dd,
                            "page": None, "quote": _clean(sent), "url": ev.get("url", ""), "manual": False})
    return entries


def from_manual(docs: list[dict]) -> list[dict]:
    by_id = {d["id"]: d for d in docs}
    out = []
    for m in MANUAL:
        d = by_id[m["doc_id"]]
        out.append({**m, "doc_title": d["title"], "date": d["date"], "url": d.get("url", ""), "manual": True})
    return out


def verify(entries: list[dict], docs: list[dict], posts: list[dict]) -> None:
    """Каждая цитата — дословно со своей страницы или из поста (с точностью до пробелов и ё/е), иначе ошибка."""
    pages = {(d["id"], no): norm(p) for d in docs for no, p in enumerate(d["pages"], start=1)}
    texts = {e["id"]: norm(e.get("text", "")) for e in posts}
    for e in entries:
        where = pages.get((e["doc_id"], e["page"])) if e["page"] else texts.get(e["doc_id"])
        if not where or norm(e["quote"]) not in where:
            raise SystemExit(f"цитата не найдена в источнике: {e['doc_id']} стр. {e['page']}: «{e['quote'][:80]}»")
        if not states_value(e):
            raise SystemExit(f"цитата не называет значение {e['value']} ({e['indicator']}): {e['doc_id']} стр. {e['page']}: "
                             f"«{e['quote'][:80]}»")


QUOTE_NUM = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{1,2})?(?![\d])")


def _value_at(e: dict) -> int | None:
    """Позиция последнего упоминания значения прогноза в цитате (14% = 14,0%); нет — None."""
    pos = [m.start() for m in QUOTE_NUM.finditer(e["quote"]) if abs(_num(m.group(0)) - abs(e["value"])) < 1e-9]
    return pos[-1] if pos else None


def states_value(e: dict) -> bool:
    """Цитата сама называет прогноз: в ней есть слово показателя и само значение."""
    return bool(WORDS[e["indicator"]].search(e["quote"])) and _value_at(e) is not None


def build() -> list[dict]:
    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    posts = load_posts()
    entries = from_docs(docs) + from_manual(docs)
    entries += from_posts(posts, entries)
    verify(entries, docs, posts)
    # одна запись на (показатель, горизонт, документ, страница, значение): текст и таблица одной страницы — разные записи
    uniq = {}
    for e in entries:
        uniq.setdefault((e["indicator"], e["horizon"], e["doc_id"], e["page"] or 0, e["value"]), e)
    entries = sorted(uniq.values(), key=lambda e: (INDICATORS.index(e["indicator"]), e["horizon"], e["date"],
                                                    e["page"] or 0, e["doc_id"], e["quote"]))
    for e in entries:
        newer = [x for x in entries if (x["indicator"], x["horizon"]) == (e["indicator"], e["horizon"]) and x["date"] > e["date"]]
        latest = max(newer, key=lambda x: x["date"]) if newer else None
        newest = [x for x in newer if latest and x["date"] == latest["date"]]
        e["superseded_by"] = newest[0]["doc_id"] if newest and not same(e["indicator"], newest[0]["value"], e["value"]) else None
    keys = ("indicator", "horizon", "value", "unit", "doc_id", "doc_title", "date", "page", "quote", "url",
            "superseded_by", "manual")
    return [{k: e[k] for k in keys} for e in entries]


def write(entries: list[dict], path: Path = OUT) -> None:
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def hand_check(entries: list[dict]) -> str:
    """Таблица для ручной сверки: действующий прогноз по каждой паре и пересмотренные значения."""
    from agent.house_view import fmt_value

    rows = []
    for ind in INDICATORS:
        for hor in sorted({e["horizon"] for e in entries if e["indicator"] == ind}):
            group = [e for e in entries if (e["indicator"], e["horizon"]) == (ind, hor)]
            cur = [e for e in group if not e["superseded_by"]]
            shown = [max(cur, key=lambda e: e["date"])] + [e for e in group if e["superseded_by"]]
            seen = set()
            for e in shown:
                if (e["doc_id"], e["value"]) in seen:
                    continue
                seen.add((e["doc_id"], e["value"]))
                mark = "пересмотрено" if e["superseded_by"] else "действует"
                rows.append(f"{ind:<9} {hor:<8} {fmt_value(e['value'], e['unit']):>8}  {e['date']}  {e['doc_id']:<14} "
                            f"стр.{e['page'] or '—':<2} {mark:<12} {_excerpt(e)}")
    return "\n".join(rows)


def _excerpt(e: dict, width: int = 80) -> str:
    """Начало цитаты (80 знаков), а если значение стоит дальше — 80 знаков вокруг него: сверять нужно само число."""
    q, pos = e["quote"], _value_at(e) or 0
    if len(q) <= width or pos < width - 10:
        return q[:width]
    start = min(pos - width // 2, len(q) - width)
    return "…" + q[start:start + width]


if __name__ == "__main__":
    out = build()
    write(out)
    print(f"{len(out)} записей → {OUT.relative_to(ROOT)}\n")
    print(hand_check(out))
