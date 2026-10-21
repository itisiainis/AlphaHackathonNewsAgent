"""Проверка карточки события без LLM: структура + правила честности.

Что делает: возвращает списки ошибок (карточку показывать нельзя) и предупреждений
(показывать можно, но стоит посмотреть). Отчёт попадает в трассу шагом «Самопроверка».

Как устроено:
1. Структура проверяется JSON Schema (card.schema.json) — тот же контракт, что рисует фронт.
2. Поверх схемы — правила, которые схемой не выразить:
   - у fact/bank-утверждения обязан быть src, и индекс должен существовать;
   - утверждение bank ссылается только на источник kind=bank;
   - позиция банка старше 12 месяцев обязана быть помечена stale (CONTRIBUTING.md, принцип 5);
   - числа из fact/bank-утверждений должны встречаться в тексте цитируемого источника
     (если текст источника передан): так выдуманная цифра ловится кодом, а не обещанием модели;
   - входные данные расчётов (blocks.calc[].inputs) проверяются так же, как утверждения.
3. Понятность выводов (style): коды отраслей, жаргон и повтор заголовка события в выводах ИИ.
   Показывать карточку можно, но агент отправляет её на правку — руководитель должен понять вывод без словаря.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).with_name("card.schema.json")
STALE_DAYS = 365
STMT_BLOCKS = ("what", "why", "risks", "opportunities", "actions")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# то, что руководитель не обязан знать: коды отраслей и внутренние термины платёжной выборки
JARGON = {r"\(\d{2}(?:\.\d+)*\)|оквэд": "код отрасли", r"\bдкп\b": "«ДКП» без расшифровки",
          r"контрагент": "«контрагенты» вместо «покупатели / поставщики»",
          r"(?:входящ|исходящ)\w* поток": "«входящий поток» вместо «поступления / платежи»"}
JARGON_RE = {re.compile(k, re.I): v for k, v in JARGON.items()}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)  # понятность выводов: не мешает показу, но повод для правки

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.ok and not self.warnings and not self.style:
            return "Все проверки пройдены"
        parts = []
        if self.errors:
            parts.append(f"ошибок: {len(self.errors)}")
        if self.warnings:
            parts.append(f"предупреждений: {len(self.warnings)}")
        if self.style:
            parts.append(f"замечаний по понятности: {len(self.style)}")
        return ", ".join(parts) + ". " + "; ".join((self.errors + self.warnings + self.style)[:5])


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def is_stale(d: str, today: date) -> bool:
    """Материал старше 12 месяцев считается устаревшим."""
    return (today - date.fromisoformat(d)).days > STALE_DAYS


def _norm_num(s: str) -> str:
    """«14,0» → «14», «1,5» → «1.5»: одно число в разных записях сравнивается как одно."""
    s = s.replace(",", ".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_in(text: str) -> set[str]:
    # пробелы-разделители разрядов («2 300») склеиваем, чтобы не дробить число
    text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)
    return {_norm_num(m) for m in NUM_RE.findall(text)}


def _iter_stmts(card: dict):
    """(место, утверждение, строгий режим). Чат — нестрогий: ответы ссылаются на источники карточки."""
    for k, s in enumerate(card.get("key_points", [])):
        yield f"key_points[{k}]", s, True
    for b in STMT_BLOCKS:
        for k, s in enumerate(card.get("blocks", {}).get(b, [])):
            yield f"blocks.{b}[{k}]", s, True
    for ci, c in enumerate(card.get("blocks", {}).get("calc", [])):
        for k, s in enumerate(c.get("inputs", [])):
            yield f"blocks.calc[{ci}].inputs[{k}]", s, True
    for qi, item in enumerate(card.get("chat", [])):
        for k, s in enumerate(item.get("a", [])):
            yield f"chat[{qi}].a[{k}]", s, False


def validate_card(card: dict, today: date | None = None, source_texts: dict[int, str] | None = None) -> Report:
    """Проверяет карточку. source_texts — {индекс источника: текст}, если агент их видел."""
    today = today or date.today()
    rep = Report()

    for err in sorted(_schema_validator().iter_errors(card), key=lambda e: list(e.path)):
        where = ".".join(str(p) for p in err.path) or "карточка"
        rep.errors.append(f"{where}: {err.message}")
    if rep.errors:
        return rep  # правила ниже опираются на корректную структуру

    sources = card["sources"]
    for where, s, strict in _iter_stmts(card):
        p, src = s["p"], s.get("src")
        if src is not None and src >= len(sources):
            rep.errors.append(f"{where}: src={src}, а источников {len(sources)}")
            continue
        if p in ("fact", "bank") and src is None:
            (rep.errors if strict else rep.warnings).append(f"{where}: утверждение «{p}» без источника")
            continue
        if p == "bank" and sources[src]["kind"] != "bank":
            rep.errors.append(f"{where}: позиция банка ссылается на источник вида «{sources[src]['kind']}»")
        if src is not None and is_stale(sources[src]["date"], today) and where.startswith("key_points"):
            rep.warnings.append(f"{where}: в коротком слое цитируется материал старше 12 месяцев")
        if source_texts and p in ("fact", "bank") and src in source_texts:
            missing = numbers_in(s["t"]) - numbers_in(source_texts[src])
            if missing:
                rep.errors.append(f"{where}: числа {sorted(missing)} не найдены в источнике [{src + 1}]")

    bank = card["blocks"]["bank"]
    if bank["quote"] and is_stale(bank["date"], today) and not bank["stale"]:
        rep.errors.append("blocks.bank: материалу больше 12 месяцев, но stale=false")
    if bank["stale"] and not is_stale(bank["date"], today):
        rep.warnings.append("blocks.bank: stale=true, но материалу меньше 12 месяцев")

    for k, a in enumerate(card["blocks"]["actions"]):
        if a.get("cta") and a["p"] != "ai":
            rep.warnings.append(f"blocks.actions[{k}]: кнопка действия у утверждения «{a['p']}»")
    rep.style = style_issues(card)
    return rep


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яёa-z]{4,}", text.lower())}


def repeats_title(text: str, title: str, share: float = 0.8) -> bool:
    """Текст почти целиком из слов заголовка события — пересказ, а не вывод."""
    st = _stems(text)
    return len(st) >= 3 and len(st & _stems(title)) / len(st) >= share


def style_issues(card: dict) -> list[str]:
    """Понятность выводов ИИ: без кодов отраслей и жаргона, без повтора заголовка. Факты и цитаты банка не трогаем —
    они переписываются из источника как есть."""
    texts = [(f"takeaway.{r}", t) for r, t in card["takeaway"].items()] + [("why_short", card["why_short"])]
    texts += [(w, s["t"]) for w, s, strict in _iter_stmts(card) if strict and s["p"] == "ai"]
    if card["blocks"]["bank"].get("plain"):
        texts.append(("blocks.bank.plain", card["blocks"]["bank"]["plain"]))
    out = []
    for where, t in texts:
        for rx, what in JARGON_RE.items():
            if rx.search(t):
                out.append(f"{where}: {what} — «{t[:60]}»")
                break
    for k, s in enumerate(card["key_points"]):
        if repeats_title(s["t"], card["title"]):
            out.append(f"key_points[{k}]: повторяет заголовок события")
    return out
