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
3. Понятность выводов (style): коды отраслей, жаргон и повтор заголовка события в выводах ИИ и выводах по правилам.
   Показывать карточку можно, но агент отправляет её на правку — руководитель должен понять вывод без словаря.
4. Пересмотренная позиция банка (check_revised=True, agent/house_view.py): позиция банка, цитата банка, заголовок
   или главный вывод повторяют прогноз, который банк уже пересмотрел (13,5% на конец 2026 года после 11.09),
   — ошибка «позиция банка пересмотрена: с 11.09 банк ждёт 14,0%». Вывод ИИ с таким числом — предупреждение.
   По умолчанию выключено: так загружаются уже готовые карточки; включают цикл агента и загрузка при старте.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from .. import house_view

SCHEMA_PATH = Path(__file__).with_name("card.schema.json")
STALE_DAYS = 365
STMT_BLOCKS = ("what", "why", "risks", "opportunities", "actions")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# то, что руководитель не обязан знать: коды отраслей и внутренние термины платёжной выборки
JARGON = {r"\(\d{2}(?:\.\d+)*\)|оквэд": "код отрасли", r"\bдкп\b": "«ДКП» без расшифровки",
          r"контрагент": "«контрагенты» вместо «покупатели / поставщики»",
          r"(?:входящ|исходящ)\w* поток": "«входящий поток» вместо «поступления / платежи»"}
JARGON_RE = {re.compile(k, re.I): v for k, v in JARGON.items()}
# Иероглифы, кана, хангыль, арабское письмо в русском выводе — сбой модели (так ошибается GLM): такой вывод не показываем
FOREIGN_SCRIPT = re.compile(r"[\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


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


_SCHEMA_RU = {"maxItems": "слишком много элементов (не больше {v})", "minItems": "слишком мало элементов (нужно не меньше {v})",
              "required": "нет обязательного поля", "enum": "недопустимое значение (можно: {v})", "type": "неверный тип (нужен {v})",
              "maxLength": "слишком длинно (не больше {v} знаков)", "additionalProperties": "лишнее поле"}


def _human(err) -> str:
    """Понятное замечание вместо дампа всего значения: «key_points: слишком много элементов (не больше 3)»."""
    tpl = _SCHEMA_RU.get(err.validator)
    if not tpl:
        return err.message[:200]
    v = err.validator_value
    return tpl.format(v=", ".join(map(str, v)) if isinstance(v, list) else v)


def validate_card(card: dict, today: date | None = None, source_texts: dict[int, str] | None = None,
                  check_revised: bool = False) -> Report:
    """Проверяет карточку. source_texts — {индекс источника: текст}, если агент их видел.
    check_revised — искать пересмотренные прогнозы банка, выданные за действующие (см. revised_issues)."""
    today = today or date.today()
    rep = Report()

    for err in sorted(_schema_validator().iter_errors(card), key=lambda e: list(e.path)):
        where = ".".join(str(p) for p in err.path) or "карточка"
        rep.errors.append(f"{where}: {_human(err)}")
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
        if a.get("cta") and a["p"] not in ("ai", "rule"):  # действие предлагает аналитик или правило, а не источник
            rep.warnings.append(f"blocks.actions[{k}]: кнопка действия у утверждения «{a['p']}»")
    rep.style = style_issues(card)
    rep.errors += script_issues(card)
    if check_revised:
        errors, warnings = revised_issues(card, today, source_texts)
        rep.errors += errors
        rep.warnings += warnings
    return rep


def revised_issues(card: dict, today: date, source_texts: dict[int, str] | None = None) -> tuple[list[str], list[str]]:
    """Пересмотренный прогноз банка, выданный за действующий: (ошибки, предупреждения).

    Ошибка — в позиции банка (p=bank, blocks.bank.quote / plain), заголовке и главном выводе (takeaway).
    Материал не старше действующего прогноза не проверяется: свежий документ сам говорит «раньше 13,5%, теперь 14,0%».
    Если старое значение называет сама новость (факт из новости или текст события), заголовок и главный вывод
    его повторяют законно: «ЦБ снизил ставку до 13,5%» — это событие, а не позиция банка. Вывод ИИ — предупреждение."""
    sources = card.get("sources", [])

    def src(i) -> dict:
        return sources[i] if isinstance(i, int) and 0 <= i < len(sources) else {}

    def found(text: str, before: str | None = None) -> list[dict]:
        return house_view.revisions_in(text or "", today, before=before)

    # что называет сама новость: факты из новостей и текст события
    news_says = {(r["indicator"], r["horizon"]) for _, s, _ in _iter_stmts(card)
                 if s["p"] == "fact" and src(s.get("src")).get("kind") == "news" for r in found(s["t"])}
    news_says |= {(r["indicator"], r["horizon"]) for i, t in (source_texts or {}).items()
                  if src(i).get("kind") == "news" for r in found(t)}

    def msg(where: str, text: str, revs: list[dict]) -> str:
        cur = "; ".join(f"с {house_view.day(r['current']['date'])} банк ждёт {house_view.revision_note(r)}" for r in revs)
        return f"{where}: позиция банка пересмотрена: {cur} — «{text[:60]}»"

    errors, warnings = [], []
    for where, s, strict in _iter_stmts(card):
        if s["p"] == "bank":
            revs = found(s["t"], src(s.get("src")).get("date"))
            if revs:
                (errors if strict else warnings).append(msg(where, s["t"], revs))
        elif s["p"] in ("ai", "rule"):
            revs = [r for r in found(s["t"]) if (r["indicator"], r["horizon"]) not in news_says]
            if revs:
                warnings.append(msg(where, s["t"], revs))
    bank = card.get("blocks", {}).get("bank", {})
    for key in ("quote", "plain"):
        revs = found(bank.get(key, ""), bank.get("date"))
        if revs:
            errors.append(msg(f"blocks.bank.{key}", bank[key], revs))
    for where, text in [("title", card.get("title", ""))] + [(f"takeaway.{r}", t) for r, t in card.get("takeaway", {}).items()]:
        revs = [r for r in found(text) if (r["indicator"], r["horizon"]) not in news_says]
        if revs:
            errors.append(msg(where, text, revs))
    return errors, warnings


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яёa-z]{4,}", text.lower())}


def repeats_title(text: str, title: str, share: float = 0.8) -> bool:
    """Текст почти целиком из слов заголовка события — пересказ, а не вывод."""
    st = _stems(text)
    return len(st) >= 3 and len(st & _stems(title)) / len(st) >= share


def _conclusions(card: dict) -> list[tuple[str, str]]:
    """Тексты выводов (не фактов и не цитат): главный вывод под роль, «почему», выводы ИИ и по правилам, «проще говоря»."""
    texts = [(f"takeaway.{r}", t) for r, t in card["takeaway"].items()] + [("why_short", card["why_short"])]
    texts += [(w, s["t"]) for w, s, strict in _iter_stmts(card) if strict and s["p"] in ("ai", "rule")]
    if card["blocks"]["bank"].get("plain"):
        texts.append(("blocks.bank.plain", card["blocks"]["bank"]["plain"]))
    return texts


def script_issues(card: dict) -> list[str]:
    """Чужое письмо в выводе — ошибка: карточка уходит на правку, а не к клиенту."""
    return [f"{where}: символы не русского письма — «{FOREIGN_SCRIPT.search(t).group()}»"
            for where, t in _conclusions(card) if FOREIGN_SCRIPT.search(t)]


def style_issues(card: dict) -> list[str]:
    """Понятность выводов ИИ и выводов по правилам: без кодов отраслей и жаргона, без повтора заголовка. Факты и цитаты
    банка не трогаем — они переписываются из источника как есть."""
    texts = _conclusions(card)
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
