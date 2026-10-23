"""Парсер аналитических материалов Альфа-Банка: PDF → текст → фрагменты для поиска.

Что делает: превращает папку с PDF (data/raw/bank/) в data/bank/docs.json — список документов
с названием, датой, постраничным текстом и фрагментами (chunks), по которым работает поиск агента.

Как устроено:
1. Текст страницы берётся из текстового слоя PDF (pypdf). Материалы кейса — сканы без текстового
   слоя, поэтому для пустых страниц включается OCR: pdftoppm рендерит страницу в PNG (200 dpi),
   Tesseract распознаёт её (rus+eng). OCR — локальный, без LLM и без расхода бюджета.
2. Дата документа ищется в начале первой страницы по шаблону «11 сентября 2026 г.»;
   если не нашлась — берётся из имени файла (…_14092026.pdf, …15Apr2021…).
3. Название — первая содержательная строка после шапки «Центр макроэкономического анализа».
4. Текст режется на фрагменты по абзацам, ~900 символов, с номером страницы — чтобы вывод агента
   можно было связать с конкретной страницей первоисточника.
Результат кешируется: повторный запуск пересчитывает только новые или изменённые файлы (по хешу).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from .. import okved

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
EN_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
RU_DATE_RE = re.compile(r"(\d{1,2})\s+([а-яё]+)\s+(20\d\d)(?:\s*г)?", re.I)
HEADER_LINES = ("центр макроэкономического анализа", "альфа-банк", "alfa-bank", "alfa bank")
CHUNK_CHARS = 900


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _month(word: str) -> int | None:
    word = word.lower()
    # «мая» и «марта» оба начинаются на «ма»: проверяем длинные основы первыми
    for stem in sorted(MONTHS, key=len, reverse=True):
        if word.startswith(stem):
            return MONTHS[stem]
    return None


def detect_date(first_page: str, filename: str) -> str | None:
    """Дата выпуска документа в ISO-формате или None."""
    for m in RU_DATE_RE.finditer(first_page[:1500]):
        month = _month(m.group(2))
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1))).isoformat()
            except ValueError:
                continue
    m = re.search(r"(\d{2})(\d{2})(20\d\d)", filename)            # 14092026
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})([A-Za-z]{3})(20\d\d)", filename)     # 15Apr2021
    if m and m.group(2).lower() in EN_MONTHS:
        return date(int(m.group(3)), EN_MONTHS[m.group(2).lower()], int(m.group(1))).isoformat()
    return None


def detect_title(first_page: str) -> str:
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    picked = []
    for ln in lines[:12]:
        low = ln.lower()
        if any(low.startswith(h) for h in HEADER_LINES) or RU_DATE_RE.search(ln) or len(ln) < 4:
            continue
        picked.append(ln)
        if len(picked) == 2:  # заголовок + подзаголовок
            break
    return ": ".join(picked)[:200] or "Без названия"


def _text_layer(path: Path) -> list[str]:
    from pypdf import PdfReader

    try:
        return [(p.extract_text() or "").strip() for p in PdfReader(path).pages]
    except Exception:
        return []


def _ocr_page(path: Path, page_no: int) -> str:
    """Распознаёт одну страницу (нумерация с 1). Пустая строка, если OCR недоступен."""
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "p"
        try:
            subprocess.run(["pdftoppm", "-r", "200", "-f", str(page_no), "-l", str(page_no), "-png",
                            str(path), str(prefix)], check=True, capture_output=True, timeout=120)
            png = next(Path(td).glob("p*.png"))
            out = subprocess.run(["tesseract", str(png), "-", "-l", "rus+eng"],
                                 check=True, capture_output=True, timeout=180)
            return out.stdout.decode("utf-8", "replace").strip()
        except (OSError, subprocess.SubprocessError, StopIteration):
            return ""


# Латинские буквы, которые OCR путает с кириллическими (правятся только внутри смешанных слов)
HOMOGLYPHS = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "k": "к",
                            "m": "м", "n": "п", "r": "г", "u": "и", "A": "А", "B": "В", "E": "Е", "K": "К",
                            "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "X": "Х"})
# Типовые ошибки OCR в макроаналитике, найденные в материалах кейса (замер: ~50 на 161 тыс. символов)
OCR_FIXES = [
    (r"\b(?:r/r|rir|г/r|r/г)\b", "г/г"),          # «год к году»
    (r"\b1?1126\b", "1П26"),                     # «первое полугодие 2026»
    (r"\b([1-4])[KК]([0-9]{2})\b", r"\1К\2"),    # кварталы: латинская K → кириллическая К
    (r"\b6\.п\.", "б.п."),                        # базисные пункты
    (r"\bpy[6б]\.", "руб."),
    (r"[|`©~^_=]{2,}", " "),                      # мусор с графиков
]


def _fix_mixed_word(m: re.Match) -> str:
    w = m.group(0)
    return w.translate(HOMOGLYPHS) if re.search(r"[а-яёА-ЯЁ]", w) and re.search(r"[A-Za-z]", w) else w


def clean_ocr(text: str) -> str:
    """Склейка строк абзаца, переносов и исправление типовых ошибок распознавания. Идемпотентна."""
    text = re.sub(r"-\n(?=[а-яё])", "", text)            # перенос слова по слогам
    text = re.sub(r"(?<![.:;!?\n])\n(?!\n)", " ", text)  # строка внутри абзаца
    for pat, rep in OCR_FIXES:
        text = re.sub(pat, rep, text)
    text = re.sub(r"[A-Za-zА-Яа-яЁё]+", _fix_mixed_word, text)
    return re.sub(r"[ \t]+", " ", text).strip()


# Ссылка на рисунок или таблицу документа: «(cm Рис.7)», «(см. Рис. 7)», «(см Таблицу ниже)». В цитате без самого
# документа она ничего не значит; «cm» — латиница, как её распознал OCR
ARTEFACT_RE = re.compile(r"\s*\((?:cm|см)\.?\s*(?:рис|табл|граф|диаграм)[^()]{0,40}\)", re.I)


def strip_artefacts(text: str) -> str:
    """Предложение без ссылок на рисунки и таблицы. Чистая функция: docs.json не меняется (по его тексту
    сверяются цитаты взгляда банка), ссылки убираются только там, где текст показывают руководителю."""
    return ARTEFACT_RE.sub("", text).strip()


def chunk_pages(pages: list[str], size: int = CHUNK_CHARS) -> list[dict]:
    """Фрагменты по абзацам: [{page, text, sectors, topics}], страницы нумеруются с 1.

    Отрасли (ОКВЭД2) и темы размечаются у каждого фрагмента автоматически словарями okved.py —
    новый документ размечается без ручной работы. Тип документа (макро / ставка / облигации /
    отраслевой обзор) задаётся в overrides.json, потому что словарь по всему документу шумит."""
    chunks = []
    for no, page in enumerate(pages, start=1):
        buf = ""
        for para in re.split(r"\n\s*\n", page):
            para = para.strip()
            if not para:
                continue
            if buf and len(buf) + len(para) > size:
                chunks.append({"page": no, "text": buf})
                buf = ""
            buf = f"{buf}\n{para}".strip()
        if buf:
            chunks.append({"page": no, "text": buf})
    for ch in chunks:
        ch["sectors"] = okved.sectors_in(ch["text"])
        ch["topics"] = okved.topics_in(ch["text"])
    return chunks


def parse_pdf(path: Path, ocr: bool = True) -> dict:
    pages = _text_layer(path)
    ocr_pages = 0
    for i, txt in enumerate(pages):
        if len(txt) < 50 and ocr:
            txt = _ocr_page(path, i + 1)
            ocr_pages += 1 if txt else 0
        pages[i] = clean_ocr(txt) if txt else ""
    first = pages[0] if pages else ""
    return {
        "file": path.name,
        "hash": file_hash(path),
        "title": detect_title(first),
        "date": detect_date(first, path.name),
        "pages": pages,
        "ocr_pages": ocr_pages,
        "chunks": chunk_pages(pages),
    }


def build_index(raw_dir: Path, out_path: Path, overrides: dict | None = None) -> list[dict]:
    """Пересобирает data/bank/docs.json. overrides — {файл: {title, date, url}} для ручных поправок."""
    old = {}
    if out_path.exists():
        old = {d["file"]: d for d in json.loads(out_path.read_text(encoding="utf-8"))}
    docs = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        h = file_hash(pdf)
        doc = old[pdf.name] if pdf.name in old and old[pdf.name]["hash"] == h else parse_pdf(pdf)
        # чистка и нарезка дешёвые — пересчитываются всегда, дорогой OCR берётся из кеша
        doc["pages"] = [clean_ocr(pg) for pg in doc["pages"]]
        doc["chunks"] = chunk_pages(doc["pages"])
        doc.update((overrides or {}).get(pdf.name, {}))
        doc["id"] = "bank-" + h[:8]
        docs.append(doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    return docs


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "raw" / "bank"
    over_path = root / "data" / "bank" / "overrides.json"
    over = json.loads(over_path.read_text(encoding="utf-8")) if over_path.exists() else {}
    for d in build_index(raw, root / "data" / "bank" / "docs.json", over):
        print(f"{d['date']}  {d['file']}  стр.{len(d['pages'])} OCR:{d['ocr_pages']}  фрагм.{len(d['chunks'])}  {d['title'][:70]}")
