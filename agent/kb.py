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
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .contract.validator import is_stale

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


class BM25Index:
    def __init__(self, items: list[dict]):
        """items: [{doc_id, title, date, page, text, url, kind}]"""
        self.items = items
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
        return [Hit(self.items[i]["doc_id"], self.items[i]["title"], self.items[i].get("date"),
                    self.items[i].get("page"), self.items[i]["text"], round(s, 3), stale,
                    self.items[i].get("url", ""), self.items[i].get("kind", "bank"),
                    self.items[i].get("doc_kind", ""), tuple(self.items[i].get("sectors", ())))
                for s, i, stale in scored[:top_k]]


def load_bank_index(path: Path) -> BM25Index:
    """data/bank/docs.json → индекс по фрагментам. Если файла нет — пустой индекс (честно: нет аналитики)."""
    if not path.exists():
        return BM25Index([])
    items = []
    for d in json.loads(path.read_text(encoding="utf-8")):
        for ch in d["chunks"]:
            items.append({"doc_id": d["id"], "title": d["title"], "date": d.get("date"), "page": ch["page"],
                          "text": ch["text"], "url": d.get("url", ""), "kind": "bank",
                          "doc_kind": d.get("kind", ""), "sectors": ch.get("sectors", [])})
    return BM25Index(items)


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
    return {"status": "none", "note": f"Отраслевой оценки банка по ОКВЭД {okved_codes[0] if okved_codes else '—'} нет — "
                                      "вывод опирается на макроаналитику банка."}
