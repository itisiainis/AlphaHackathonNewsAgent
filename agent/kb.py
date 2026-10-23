"""Поиск по аналитике банка и новостям: BM25 с фильтром по дате и пометкой устаревания.

Почему BM25, а не векторный поиск: в списке разрешённых моделей кейса нет embedding-моделей,
а внешние AI-сервисы запрещены. BM25 работает локально, бесплатно и объяснимо — в трассе видно,
по каким словам найден фрагмент.

Как работает BM25 (Okapi): для запроса q и фрагмента d
    score(d, q) = Σ_по словам w из q  idf(w) · f(w,d)·(k1+1) / (f(w,d) + k1·(1 − b + b·|d|/avgdl))
где f(w,d) — сколько раз слово встречается во фрагменте, |d| — длина фрагмента в словах,
avgdl — средняя длина, idf(w) = ln(1 + (N − n(w) + 0.5)/(n(w) + 0.5)), N — число фрагментов,
n(w) — во скольких фрагментах слово есть. Редкие слова весят больше частых; длинный фрагмент
не выигрывает только за счёт длины (параметр b).

Морфология русского языка упрощена усечением слова до первых STEM_LEN букв: «ставки», «ставку»,
«ставкой» → «ставк». Для поиска по десятку документов этого хватает и не требует словарей.

Пересмотренные прогнозы (agent/house_view.py). BM25 не знает, какой прогноз банка действует: 31 июля банк
ждал ставку 13,5% на конец 2026 года, 11 сентября — 14,0%, и оба фрагмента похожи на запрос. После ранжирования
фрагмент со старым значением получает пометку revised («пересмотрено 11.09: банк ждёт 14,0% (…)») и встаёт после
фрагмента с действующим прогнозом; если того нет в выдаче, он добавляется. Новейший прогноз всегда рядом со старым.

Голос банка вне документов: посты каналов банка (bank_voice в data/sources.json), где говорит его экономист,
добавляются в индекс как аналитика банка — add_posts(): «Альфа-Инвестиции, пост от 20.10.2026», дата поста.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import house_view
from .contract.validator import is_stale
from .okved import short_name

STEM_LEN = 6
SECTOR_BOOST = 1.5
K1, B = 1.5, 0.75
WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
STOP = {"и", "в", "во", "на", "с", "со", "по", "к", "о", "об", "от", "до", "за", "из", "у", "не", "что",
        "это", "как", "для", "а", "но", "или", "же", "при", "мы", "наш", "его", "ее", "их", "то", "так"}


def tokens(text: str) -> list[str]:
    out = []
    for w in WORD_RE.findall(text.lower().replace("ё", "е")):
        if w in STOP or len(w) < 2:
            continue
        out.append(w[:STEM_LEN] if not w.isdigit() else w)
    return out


@dataclass
class Hit:
    doc_id: str
    title: str
    date: str | None
    page: int | None
    text: str
    score: float
    stale: bool
    url: str = ""
    kind: str = "bank"
    doc_kind: str = ""                     # macro | rates | bonds | commodity | industry
    sectors: tuple = ()                    # классы ОКВЭД2, упомянутые во фрагменте
    revised: str = ""                      # «пересмотрено 11.09: банк ждёт 14,0% (ставка на конец 2026)»
    revisions: tuple = ()                  # [{indicator, horizon, old, current}] — для пометки и «источники расходятся»


class BM25Index:
    def __init__(self, items: list[dict], house: list[dict] | None = None):
        """items: [{doc_id, title, date, page, text, url, kind}]; house — записи взгляда банка (house_view.json)."""
        self.items = items
        self.house = house or []
        self._anchors: dict[tuple, int | None] = {}
        self.tf = [Counter(tokens(i["title"] + " " + i["text"])) for i in items]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = Counter(w for c in self.tf for w in c)
        n = len(items)
        self.idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}

    def search(self, query: str, top_k: int = 5, max_age_days: int | None = None,
               today: date | None = None, include_stale: bool = True, boost_sectors: set | None = None) -> list[Hit]:
        """boost_sectors — отрасли компании: фрагменты про них поднимаются в выдаче (×SECTOR_BOOST)."""
        today = today or date.today()
        q = tokens(query)
        scored = []
        for i, item in enumerate(self.items):
            d = item.get("date")
            if max_age_days is not None and d and (today - date.fromisoformat(d)).days > max_age_days:
                continue
            stale = bool(d) and is_stale(d, today)
            if stale and not include_stale:
                continue
            s = 0.0
            for w in q:
                f = self.tf[i].get(w)
                if f:
                    s += self.idf[w] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.len[i] / self.avg))
            if s > 0 and boost_sectors and boost_sectors & set(item.get("sectors", ())):
                s *= SECTOR_BOOST
            if s > 0:
                scored.append((s, i, stale))
        scored.sort(reverse=True)
        picked, revs = self._revise(scored, top_k, today) if self.house else (scored[:top_k], {})
        return [Hit(self.items[i]["doc_id"], self.items[i]["title"], self.items[i].get("date"),
                    self.items[i].get("page"), self.items[i]["text"], round(s, 3), stale,
                    self.items[i].get("url", ""), self.items[i].get("kind", "bank"),
                    self.items[i].get("doc_kind", ""), tuple(self.items[i].get("sectors", ())),
                    house_view.revised_label(revs[i]) if i in revs else "", tuple(revs.get(i, ())))
                for s, i, stale in picked]

    def _anchor(self, entry: dict) -> int | None:
        """Фрагмент, в котором стоит цитата действующего прогноза (тот же документ и страница)."""
        key = (entry["doc_id"], entry.get("page"), entry["quote"])
        if key not in self._anchors:
            q = house_view.norm(entry["quote"])
            self._anchors[key] = next((i for i, it in enumerate(self.items) if it["doc_id"] == entry["doc_id"]
                                       and (entry.get("page") is None or it.get("page") == entry["page"])
                                       and q in house_view.norm(it["text"])), None)
        return self._anchors[key]

    def _revise(self, scored: list, top_k: int, today: date) -> tuple[list, dict]:
        """Пересмотренный прогноз — после действующего: пометка revised, действующий фрагмент добавляется в выдачу.

        Выдача остаётся не длиннее top_k за счёт самых слабых обычных фрагментов; действующий прогноз
        и помеченный старый не выбрасываются — их расхождение и есть то, что агент должен увидеть."""
        out = list(scored[:top_k])
        revs: dict[int, list[dict]] = {}
        for _, i, _ in out:
            it = self.items[i]
            if it.get("kind", "bank") == "bank":
                r = house_view.revisions_in(it["text"], today, before=it.get("date"), entries=self.house)
                if r:
                    revs[i] = r
        if not revs:
            return out, revs
        score = {i: s for s, i, _ in scored}
        keep: set[int] = set(revs)
        for i, rs in revs.items():
            for r in rs:
                a = self._anchor(r["current"])
                if a is None:
                    continue
                keep.add(a)
                ids = [j for _, j, _ in out]
                if a not in ids:
                    d = self.items[a].get("date")
                    out.insert(ids.index(i), (score.get(a, 0.0), a, bool(d) and is_stale(d, today)))
                    continue
                ia, ii = ids.index(a), ids.index(i)
                if ii < ia:  # старый прогноз стоял выше действующего — ставим его сразу после
                    out.insert(ia, out.pop(ii))
        while len(out) > top_k:
            drop = next((k for k in range(len(out) - 1, -1, -1) if out[k][1] not in keep), None)
            if drop is None:
                break
            out.pop(drop)
        return out, revs


def load_bank_index(path: Path) -> BM25Index:
    """data/bank/docs.json → индекс по фрагментам. Если файла нет — пустой индекс (честно: нет аналитики).
    Взгляд банка (house_view.json рядом) включает пометку пересмотренных прогнозов."""
    house = house_view.load(path.with_name("house_view.json"))
    if not path.exists():
        return BM25Index([], house)
    items = []
    for d in json.loads(path.read_text(encoding="utf-8")):
        for ch in d["chunks"]:
            items.append({"doc_id": d["id"], "title": d["title"], "date": d.get("date"), "page": ch["page"],
                          "text": ch["text"], "url": d.get("url", ""), "kind": "bank",
                          "doc_kind": d.get("kind", ""), "sectors": ch.get("sectors", [])})
    return BM25Index(items, house)


def add_fragment(index: BM25Index, item: dict) -> BM25Index:
    """Индекс с ещё одним фрагментом (фрагмент с тем же doc_id заменяется). Возвращает НОВЫЙ индекс: старый
    не меняется, поэтому поиск, идущий в другом потоке, не увидит полусобранного состояния."""
    return add_fragments(index, [item])


def add_fragments(index: BM25Index, new: list[dict]) -> BM25Index:
    ids = {i["doc_id"] for i in new}
    return BM25Index([i for i in index.items if i["doc_id"] not in ids] + new, index.house)


# Экономист банка в тексте поста: только его прогноз — позиция банка; реклама продуктов и мнения
# инвестсоветников канала ею не считаются
ECONOMIST_RE = re.compile(r"Орлов\w*|главн\w* экономист\w*|экономист\w* Альфа-Банка|аналитик\w* Альфа-Банка|"
                          r"по (?:мнению|оценке|прогнозу) Альфа-Банка|Альфа-Банк\w* (?:считает|ожидает|прогнозирует|жд[её]т)")
SOURCES_PATH = Path(__file__).resolve().parents[1] / "data" / "sources.json"


def bank_voice_sources(path: Path = SOURCES_PATH) -> set[str]:
    """Названия источников с bank_voice: true — каналы самого банка."""
    if not path.exists():
        return set()
    src = json.loads(path.read_text(encoding="utf-8"))
    return {s["name"] for kind in ("rss", "telegram") for s in src.get(kind, []) if s.get("bank_voice")}


def post_fragment(event: dict, voices: set[str] | None = None) -> dict | None:
    """Пост канала банка, где говорит его экономист → фрагмент аналитики банка; иначе None."""
    voices = bank_voice_sources() if voices is None else voices
    if not (event.get("bank_voice") or event.get("source") in voices) or not ECONOMIST_RE.search(event.get("text", "")):
        return None
    d = event.get("published") or ""
    return {"doc_id": event["id"], "title": f"{event.get('source', 'Альфа-Банк')}, пост от {d[8:10]}.{d[5:7]}.{d[:4]}",
            "date": d or None, "page": None, "text": event.get("text", ""), "url": event.get("url", ""), "kind": "bank",
            "doc_kind": "macro", "sectors": []}


def add_posts(index: BM25Index, events: list[dict]) -> BM25Index:
    """Индекс аналитики банка + посты его экономистов из потока новостей (повторный вызов не дублирует)."""
    voices = bank_voice_sources()
    frags = [f for f in (post_fragment(e, voices) for e in events) if f]
    return add_fragments(index, frags) if frags else index


def load_bank_meta(path: Path) -> list[dict]:
    """Документы банка без текста: id, название, дата, тип, отрасли — для правила «есть ли отраслевая оценка»."""
    if not path.exists():
        return []
    return [{k: d.get(k) for k in ("id", "title", "date", "kind", "okved_prefixes", "url")}
            for d in json.loads(path.read_text(encoding="utf-8"))]


def industry_assessment(meta: list[dict], okved_codes: list[str], today: date) -> dict:
    """Есть ли у банка отраслевой обзор по отраслям компании и свежий ли он.

    Отраслевой — документ типа industry или commodity, один из префиксов ОКВЭД которого совпадает
    с началом кода компании (основного или класса из её группы). Сравнение по префиксу полного кода,
    а не по двузначному классу: обзор рынка золота (24.41) не оценка для опта металлами (46.72).
    Макрообзоры отраслевой оценкой не считаются. Свежесть — те же 12 месяцев, что и везде.
    Результат детерминирован и попадает в карточку дословно: решение «есть / нет / устарела»
    принимает код, а не модель (CONTRIBUTING.md, принцип 5)."""
    docs = [d for d in meta if d.get("kind") in ("industry", "commodity")
            and any(c.startswith(pfx) for pfx in d.get("okved_prefixes") or [] for c in okved_codes)]
    fresh = [d for d in docs if d.get("date") and not is_stale(d["date"], today)]
    if fresh:
        d = max(fresh, key=lambda x: x["date"])
        return {"status": "fresh", "note": f"Есть отраслевой обзор банка: «{d['title']}» ({d['date']})."}
    if docs:
        d = max(docs, key=lambda x: x["date"] or "")
        return {"status": "stale", "note": f"Последняя отраслевая оценка банка — «{d['title']}» ({d['date']}): "
                                           "старше 12 месяцев, в выводе не используется."}
    return {"status": "none", "note": no_industry_view(okved_codes[0] if okved_codes else None)}


def no_industry_view(okved2: str | None) -> str:
    """Отраслевой оценки нет — словами: код отрасли руководителю не нужен (CONTRIBUTING.md, принцип 4а)."""
    return (f"Отраслевой оценки банка по отрасли «{short_name(okved2)}» нет" if okved2 else "Отраслевой оценки банка нет") + \
        " — вывод опирается на макроаналитику банка."
