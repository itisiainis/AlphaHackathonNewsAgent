"""Агентный цикл: план → инструменты → черновик → самопроверка → (повторный поиск → правка) → карточка.

Свой цикл без агентных фреймворков (требование кейса). Модель на каждом шаге выбирает действие
JSON-объектом; код выполняет инструмент и возвращает результат как наблюдение. Состояние
(«контекст» из требований кейса) — это история сообщений и реестр источников ToolSession:
черновик и проверка опираются на всё, что найдено на предыдущих шагах.

Самопроверка двухслойная:
  1) код — validate_card: схема, происхождение, ссылки, устаревание, сверка чисел с текстом источников;
  2) модель-проверяющий (другая модель, не та, что писала черновик) — полнота и противоречия.
Если проверка не пройдена и есть запас ревизий, агент делает повторный поиск по подсказке
проверяющего и переписывает черновик с учётом замечаний. Каждый шаг пишется в трассу.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from . import house_view, profiles
from .contract import Report, validate_card
from .contract.validator import numbers_in
from .ingest.bank_docs import strip_artefacts
from .kb import BM25Index, no_industry_view
from .llm import LLM, NO_REASONING, ROUTING, LLMUnavailable, extract_json
from .matching import Match
from .offline import NO_BANK_VIEW, _bank_quote, _stems
from .prompts import AGENT_SYSTEM, CHECK_SYSTEM, DRAFT_SYSTEM, ROLES_RU, house_view_block, tools_block
from .tools import ToolSession

MAX_TOOL_STEPS = 5
# Что показать пользователю, пока модель отвечает на шаг с этим названием
PENDING = {"План": "Модель составляет план и выбирает инструменты…", "Черновик карточки": "Модель пишет черновик карточки…",
           "Правка черновика": "Модель исправляет черновик по замечаниям проверки…",
           "Самопроверка (модель)": "Проверяющая модель (другая) читает черновик и источники…",
           "Шаг": "Модель выбирает следующий шаг…"}


BANK_VIEW = re.compile(r"прогноз|ожида|считае|полагае|оценивае|рекоменд|альфа-банк|по мнению банка|по оценке банка", re.I)


def quote_matches(quote: str, source: str, share: float = 0.6) -> bool:
    """Утверждение опирается на текст документа: не меньше 60% его слов (по основе из 5 букв) есть в источнике.
    Мягкая проверка пересказа («Проще говоря» рядом с цитатой, позиция банка в коротком слое);
    цитата blocks.bank.quote — только verbatim()."""
    words = {w[:5] for w in re.findall(r"[а-яёa-z]{5,}", quote.lower())}
    if not words:
        return True
    src = {w[:5] for w in re.findall(r"[а-яёa-z]{5,}", source.lower())}
    return len(words & src) / len(words) >= share


QUOTE_MIN = 20  # короче — не цитата, а пара слов из документа


def norm_quote(text: str) -> str:
    """Текст для сверки дословности. Различия, которых читатель не видит или которые внёс OCR, не считаются:
    регистр, ё/е, пробелы, мягкий перенос и перенос слова по строкам («ожи-\\nдаем»), вид кавычек и тире,
    ссылки на рисунки («(cm Рис.7)»); точку, запятую и точку с запятой OCR путает — для сверки это один знак."""
    t = strip_artefacts(text).replace("\xad", "").lower().replace("ё", "е")
    t = re.sub(r"[‐‑‒–—―−]", "-", t)
    t = re.sub(r"(?<=[а-яa-z])-\s+(?=[а-яa-z])", "", t)
    t = re.sub(r"[«»„“”‟\"'‘’‚]", '"', t)
    t = re.sub(r"[.,;]", ",", t)
    return re.sub(r"\s+", " ", t).strip()


def verbatim(quote: str, source: str) -> bool:
    """Цитата банка — непрерывный кусок текста документа после norm_quote: от границы слова до границы слова и
    без отрезанного отрицания («ожидаем снижения» из «не ожидаем снижения» — не цитата). Внешние кавычки и
    многоточия по краям не мешают; многоточие внутри — пропуск текста, такая «цитата» не дословна."""
    q = norm_quote(quote).strip(' "…,')
    if len(q) < QUOTE_MIN:
        return False
    src = norm_quote(source)
    at = src.find(q)
    while at >= 0:
        end = at + len(q)
        if ((at == 0 or not src[at - 1].isalnum()) and (end == len(src) or not src[end].isalnum())
                and not re.search(r"(?:^|\s)н[еи]\s*$", src[max(0, at - 4):at])):
            return True
        at = src.find(q, at + 1)
    return False


CODE_RE = re.compile(r"\s*\((?:ОКВЭД\s*)?\d{2}(?:\.\d+)*\)|\s+(?:по\s+)?ОКВЭД\s*\d{2}(?:\.\d+)*")


def no_codes(text: str) -> str:
    """«Автодилеры (45)» → «Автодилеры», «оценка банка по ОКВЭД 47.71 недоступна» → «оценка банка недоступна»:
    коды отраслей в выводе руководителю не нужны — убирает код, а не модель."""
    return CODE_RE.sub("", text)


OLD_NO_VIEW = re.compile(r"Отраслевой оценки банка по ОКВЭД (\d{2}[\d.]*|—) нет — вывод опирается на макроаналитику банка\.")


def refresh_saved(card: dict) -> dict:
    """Готовый разбор (data/cards) — по правилам нынешнего кода, без модели и без изменения смысла выводов:
    итог трассы — по её же шагам, коды отраслей убраны из выводов ИИ и из пометки об отраслевой оценке."""
    def plain(s):
        return {**s, "t": no_codes(s["t"])} if isinstance(s, dict) and s.get("p") == "ai" and isinstance(s.get("t"), str) else s

    blocks = dict(card.get("blocks") or {})
    for k in ("what", "why", "risks", "opportunities", "actions"):
        if isinstance(blocks.get(k), list):
            blocks[k] = [plain(s) for s in blocks[k]]
    bank, basis = blocks.get("bank"), blocks.get("basis")
    if isinstance(bank, dict) and isinstance(bank.get("plain"), str):
        blocks["bank"] = {**bank, "plain": no_codes(bank["plain"])}
    if isinstance(basis, dict) and isinstance(basis.get("coverage"), str):
        blocks["basis"] = {**basis, "coverage": OLD_NO_VIEW.sub(
            lambda m: no_industry_view(None if m.group(1) == "—" else m.group(1)), basis["coverage"])}
    return {**card, "takeaway": {r: no_codes(str(t)) for r, t in (card.get("takeaway") or {}).items()},
            "why_short": no_codes(str(card.get("why_short", ""))), "key_points": [plain(s) for s in card.get("key_points") or []],
            "blocks": blocks, "trace": restate_final(card.get("trace") or [])}


def _count(result) -> int:
    """Сколько объектов вернул инструмент: список, фрагменты поиска по аналитике или один объект."""
    if isinstance(result, dict) and "fragments" in result:
        return len(result["fragments"])
    return len(result) if isinstance(result, list) else 1
MAX_REVISIONS = 1


@dataclass
class Result:
    card: dict | None
    report: Report
    trace: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    error: str = ""
    unavailable: bool = False  # модель недоступна (нет ключа, бюджет, сбой) — можно показать разбор правилами


def final_note(code: str, verdict: dict | None) -> str:
    """Итог трассы — как было на самом деле: код (сводка проверки), проверяющая модель, правка."""
    parts = ["Проверка кодом пройдена" if code == "Все проверки пройдены" else f"Проверка кодом: {code}"]
    if verdict is not None:
        if verdict.get("ok") is None:
            parts.append("проверка второй моделью не выполнена")
        elif verdict.get("ok") is False:
            parts.append("проверяющая модель оставила замечания: " + "; ".join(map(str, verdict.get("issues", [])))[:400])
        else:
            parts.append("вторая модель замечаний не нашла")
    return " · ".join(p for p in parts if p)


def restate_final(trace: list[dict]) -> list[dict]:
    """Итог готового разбора — заново из его же шагов. Разборы, посчитанные до правки итога, писали «Все проверки
    пройдены» и тогда, когда замечания проверяющей модели никто повторно не проверил; шаги трассы при этом верны."""
    if not trace or trace[-1].get("kind") != "done" or trace[-1].get("title") != "Карточка готова":
        return trace
    code, verdict, revised = "", None, False
    kept = (code, verdict)  # что было до правки: правку, не прошедшую проверку кодом, цикл откатывает
    for st in trace[:-1]:
        title, detail = str(st.get("title") or ""), str(st.get("detail") or "")
        if st.get("kind") == "check" and "(код)" in title:
            code = detail.split(" Исправлено до проверки:")[0].strip()
        elif st.get("kind") == "check" and "(модель)" in title:
            revised = False
            verdict = ({"ok": None} if detail.startswith("не выполнена") else
                       {"ok": True} if detail.strip() in ("", "замечаний нет") else {"ok": False, "issues": [detail]})
        elif st.get("kind") == "check" and title == "Правка черновика":
            if revised:  # в карточке — прежний черновик, к нему и относятся замечания
                code, verdict = kept
            revised = False
        elif st.get("kind") == "draft":
            kept, revised = (code, verdict), verdict is not None
    note = final_note(code or "Все проверки пройдены", verdict)
    if revised and verdict and verdict.get("ok") is False:
        note += " · черновик исправлен по замечаниям, повторно моделью не проверен"
    return trace[:-1] + [{**trace[-1], "detail": note}]


class Agent:
    def __init__(self, llm: LLM, bank: BM25Index, news: BM25Index, today: date | None = None,
                 bank_meta: list[dict] | None = None):
        self.llm, self.bank, self.news = llm, bank, news
        self.bank_meta = bank_meta or []
        self.today = today or date.today()
        self.on_step = None  # обратный вызов для показа шагов в интерфейсе в реальном времени
        self.lessons: list[str] = []      # уроки прошлых проверок (agent/learning.py)
        self.precedents: list[dict] = []  # прошлые разборы этой же компании по теме

    def _emit(self, trace: list, pending: str = "") -> None:
        """Отдаёт текущее состояние трассы наружу (поток SSE). pending — что агент делает прямо сейчас."""
        if self.on_step:
            self.on_step({"trace": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trace],
                          "pending": pending})

    # ---------- вспомогательное ----------
    def _ask(self, role: str, messages: list[dict], trace: list, kind: str, title: str, max_tokens=6000, think=None):
        """Вызов модели с одной повторной попыткой, если ответ не JSON. think — включить скрытые рассуждения."""
        self._emit(trace, PENDING.get(title, "Модель работает…"))
        think = role not in NO_REASONING if think is None else think
        c = self.llm.complete(ROUTING[role], messages, max_tokens=max_tokens, json_mode=True, reasoning=think)
        cost, ms = c.cost_rub, c.ms
        try:
            data = extract_json(c.text)
        except ValueError:
            c2 = self.llm.complete(ROUTING[role], messages + [
                {"role": "assistant", "content": c.text},
                {"role": "user", "content": "Ответ не разобрался как JSON. Верни только JSON-объект."}],
                max_tokens=max_tokens, json_mode=True, reasoning=think)
            cost, ms = cost + c2.cost_rub, ms + c2.ms
            data = extract_json(c2.text)
        trace.append({"kind": kind, "title": title, "detail": "", "ms": ms, "cost": round(cost, 4),
                      "_model": c.model, "_cached": c.cached})
        return data

    @staticmethod
    def _sources_block(ses: ToolSession) -> str:
        return "\n\n".join(f"[{s.label}] {s.kind} · {s.name} · {s.date}{' · УСТАРЕЛО' if s.stale else ''}"
                           f"{' · ' + s.revised if s.revised else ''}\n{s.text[:1200]}"
                           for s in ses.sources)

    # ---------- основной сценарий ----------
    def run(self, event: dict, company_id: str, match: Match) -> Result:
        trace: list[dict] = []
        ses = ToolSession(company_id, event, self.bank, self.news, self.today, self.bank_meta)
        try:
            observations = self._explore(ses, match, trace)
            self._ensure_bank(ses, observations, trace)
            self._calculate(ses, observations, trace)
            draft = self._draft(ses, match, observations, trace, issues=[])
            card, rep = self._assemble(draft, ses, event, company_id, match, trace)
            verdict = self._check(card, ses, trace) if rep.ok else {"ok": False, "issues": [], "research_query": ""}
            for _ in range(MAX_REVISIONS):
                # замечания по понятности (жаргон, коды отраслей, повтор заголовка) — тоже повод для правки
                issues = rep.errors + rep.style + list(verdict.get("issues") or [])
                if rep.ok and verdict.get("ok") is not False and not rep.style:
                    break
                q = (verdict.get("research_query") or "").strip()
                if q:
                    # где искать — решает проверка: позиция банка — в аналитике банка, факт события — в новостях
                    tool = "search_news" if verdict.get("research_in") == "news" else "search_bank_analytics"
                    res, entry = ses.call(tool, {"query": q})
                    trace.append({"kind": "research", "title": f"{tool}(\"{q}\")",
                                  "detail": f"повторный поиск по замечанию проверки: {_count(res)} объект(ов)" + ses.note(res),
                                  "ms": entry["ms"], "cost": 0})
                    observations.append({"tool": tool, "args": {"query": q}, "result": res})
                good = (card, rep) if rep.ok else None
                try:
                    draft = self._draft(ses, match, observations, trace, issues=issues)
                    card, rep = self._assemble(draft, ses, event, company_id, match, trace)
                except (LLMUnavailable, ValueError, KeyError, TypeError) as e:
                    if not good:
                        raise
                    # правка не удалась — остаётся черновик, прошедший проверку кодом; замечание видно в трассе
                    card, rep = good
                    trace.append({"kind": "check", "title": "Правка черновика", "ms": 0, "cost": 0,
                                  "detail": f"не выполнена ({e}); оставлен черновик, прошедший проверку кодом"})
                if rep.ok is False and good:
                    card, rep = good
                    trace.append({"kind": "check", "title": "Правка черновика", "ms": 0, "cost": 0,
                                  "detail": "правка не прошла проверку кодом — оставлен прежний черновик, прошедший проверку"})
                elif rep.ok:
                    verdict = self._check(card, ses, trace, again=True)  # правку проверяет та же модель ещё раз
        except LLMUnavailable as e:
            return Result(None, Report(errors=[str(e)]), trace, ses.log, error=str(e), unavailable=True)
        except (ValueError, KeyError, TypeError) as e:
            return Result(None, Report(errors=[f"ответ модели не разобран: {e}"]), trace, ses.log, error=str(e))

        trace.append({"kind": "done", "title": "Карточка готова" if rep.ok else "Карточка не прошла проверку",
                      "detail": final_note(rep.summary(), verdict if rep.ok else None), "ms": 0, "cost": 0})
        self._emit(trace)
        card["trace"] = [self._public_step(t) for t in trace]
        return Result(card if rep.ok else None, rep, trace, ses.log)

    def _explore(self, ses: ToolSession, match: Match, trace: list) -> list[dict]:
        system = (AGENT_SYSTEM.replace("{tools}", tools_block(ses.TOOLS))
                  .replace("{max_steps}", str(MAX_TOOL_STEPS)))
        reasons = "; ".join(f"{r['feature']} ({r['why']})" for r in match.reasons) or "явных совпадений нет"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Событие (источник S1): {json.dumps(ses.get_event(), ensure_ascii=False)[:4000]}\n\n"
                     f"Профиль компании: {json.dumps(ses.get_company_profile(), ensure_ascii=False)}\n\n"
                     f"Предварительный отбор сработал на признаки: {reasons}.\n"
                     "Новость и профиль уже даны — не запрашивай их повторно. Ищи позицию банка и связанные события.\n"
                     "В первом ответе добавь поле \"plan\" — список шагов анализа."}]
        observations = []
        for step in range(MAX_TOOL_STEPS + 1):
            try:
                act = self._ask("agent", messages, trace, "plan", "План" if step == 0 else "Шаг")
            except ValueError as e:
                # модель дважды ответила не JSON — идём дальше с тем, что уже найдено (банк и расчёт код добавит сам)
                trace.append({"kind": "plan", "title": "План" if step == 0 else "Шаг", "ms": 0, "cost": 0,
                              "detail": f"ответ модели не разобран ({e}) — исследование завершено досрочно"})
                break
            if step == 0:
                trace[-1]["detail"] = "; ".join(map(str, act.get("plan") or [])) or str(act.get("thought", ""))
                think = {"ms": 0, "cost": 0}
                self._emit(trace)
            else:
                # выбор следующего действия — не отдельный шаг трассы: его время и цена идут в шаг инструмента
                think = trace.pop()
            name = act.get("action")
            if name == "final" or step == MAX_TOOL_STEPS:
                if trace:
                    trace[-1]["ms"] += think["ms"]
                    trace[-1]["cost"] = round(trace[-1]["cost"] + think["cost"], 4)
                break
            result, entry = ses.call(name, act.get("args"))
            args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (act.get("args") or {}).items())
            trace.append({"kind": "tool", "title": f"{name}({args})", "ms": entry["ms"] + think["ms"], "cost": think["cost"],
                          "detail": (act.get("thought") or "")[:200] + (" · ошибка: " + result["error"] if not entry["ok"] else
                                     f" · результат: {_count(result)} объект(ов)" + ses.note(result))})
            observations.append({"tool": name, "args": act.get("args"), "result": result})
            self._emit(trace, "Модель выбирает следующий шаг…")
            messages += [{"role": "assistant", "content": json.dumps(act, ensure_ascii=False)},
                         {"role": "user", "content": f"Результат {name}: " + json.dumps(result, ensure_ascii=False)[:6000]}]
        return observations

    def _ensure_bank(self, ses: ToolSession, observations: list[dict], trace: list) -> None:
        """Позиция банка — обязательная часть разбора (кейс): если модель не искала её сама, ищет код."""
        if any(o["tool"] == "search_bank_analytics" and isinstance(o["result"], dict) and "fragments" in o["result"]
               for o in observations):
            return
        q = " ".join([ses.event.get("title", "")[:150], *ses.event.get("topics", [])])
        res, entry = ses.call("search_bank_analytics", {"query": q})
        observations.append({"tool": "search_bank_analytics", "args": {"query": q}, "result": res})
        trace.append({"kind": "tool", "title": f"search_bank_analytics(\"{q[:60]}\")", "ms": entry["ms"], "cost": 0,
                      "detail": f"обязательный шаг (модель не искала позицию банка): {_count(res)} фрагм." + ses.note(res)})
        self._emit(trace)

    def _calculate(self, ses: ToolSession, observations: list[dict], trace: list) -> None:
        """Расчёт эффекта кодом — всегда, даже если модель не вызвала калькулятор: цифры не зависят от модели."""
        done = next((o for o in observations if o["tool"] == "calculate_impact" and isinstance(o["result"], list)), None)
        if done:
            ses.calcs = done["result"]
            return
        res, entry = ses.call("calculate_impact", {})
        ses.calcs = res if isinstance(res, list) else []
        summary = "; ".join(f"{c['title']}: {c['value']}" for c in ses.calcs)
        trace.append({"kind": "tool", "title": "calculate_impact()", "ms": entry["ms"], "cost": 0,
                      "detail": "расчёт по формулам кода: " + summary[:200] if ses.calcs else "числового шока в новости нет — расчёта нет"})
        self._emit(trace)

    def _draft(self, ses, match, observations, trace, issues: list[str]) -> dict:
        profile = ses.get_company_profile()
        user = (f"Профиль компании: {json.dumps(profile, ensure_ascii=False)}\n\n"
                f"Признаки, на которые сработал отбор: {json.dumps(match.matched, ensure_ascii=False)}\n\n"
                f"Источники (ссылайся на метки):\n{self._sources_block(ses)}")
        if ses.calcs:
            user += "\n\nРасчёты эффекта (код):\n" + "\n".join(
                f"- {c['title']}: {c['value']} ({c['formula']}); допущения: {'; '.join(c['assumptions'])}" for c in ses.calcs)
        if self.precedents:
            user += ("\n\nПрошлые разборы для этой компании (контекст, не источник — цифры из них не бери): "
                     "если тема та же, скажи, что изменилось с тех пор.\n" + "\n".join(
                         f"- {pr['date']}: «{pr['title']}» → {pr['takeaway']}" for pr in self.precedents))
        if self.lessons:
            user += "\n\nУроки прошлых разборов (частые замечания проверки — не повторяй):\n- " + "\n- ".join(self.lessons)
        if issues:
            user += "\n\nЗамечания проверки, которые нужно исправить:\n- " + "\n- ".join(issues[:10])
        system = (DRAFT_SYSTEM.format(roles=", ".join(f"{k} — {v}" for k, v in ROLES_RU.items()))
                  + house_view_block(self.today, ses.sources))  # действующие прогнозы банка (agent/house_view.py)
        draft = self._ask("draft", [{"role": "system", "content": system}, {"role": "user", "content": user}],
                          trace, "draft", "Черновик карточки" if not issues else "Правка черновика", max_tokens=10000, think=not issues)
        n = sum(len(draft.get("blocks", {}).get(b, [])) for b in ("what", "why", "risks", "opportunities", "actions"))
        trace[-1]["detail"] = f"{n} утверждений, источников в работе: {len(ses.sources)}"
        self._emit(trace, "Код проверяет происхождение, ссылки и цифры…")
        return draft

    def _labelled(self, card: dict) -> dict:
        """Карточка для проверяющей модели: src — метки S…, как в списке источников, а не индексы sources[].
        Раньше проверка видела «src=1» рядом с «[S2]» и половину замечаний писала о несуществующей путанице."""
        labels = getattr(self, "_cited", [])

        def walk(x):
            if isinstance(x, dict):
                return {k: (labels[v] if k == "src" and isinstance(v, int) and 0 <= v < len(labels) else walk(v))
                        for k, v in x.items() if k not in ("trace", "sources")}
            return [walk(i) for i in x] if isinstance(x, list) else x
        return walk(card)

    def _check(self, card: dict, ses: ToolSession, trace: list, again: bool = False) -> dict:
        user = f"Черновик:\n{json.dumps(self._labelled(card), ensure_ascii=False)}\n\nИсточники:\n{self._sources_block(ses)}"
        title = "Повторная самопроверка (модель)" if again else "Самопроверка (модель)"
        try:
            verdict = self._ask("check", [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}],
                                trace, "check", title, max_tokens=6000)
        except (LLMUnavailable, ValueError) as e:
            # проверяющая модель не ответила — карточка, прошедшая проверку кодом, не теряется; в трассе и в итоге видно,
            # что второй проверки не было (а не «все проверки пройдены»)
            trace.append({"kind": "check", "title": title, "detail": f"не выполнена: {e}", "ms": 0, "cost": 0})
            self._emit(trace)
            return {"ok": None, "issues": []}
        trace[-1]["detail"] = "замечаний нет" if verdict.get("ok") else "; ".join(map(str, verdict.get("issues", [])))[:1500]
        self._emit(trace)
        return verdict

    @staticmethod
    def _public_step(t: dict) -> dict:
        """Шаг трассы для карточки: служебные поля → модель и признак кеша (критерий 11)."""
        step = {k: v for k, v in t.items() if not k.startswith("_")}
        if t.get("_model"):
            step["model"] = str(t["_model"])
        if t.get("_cached"):
            step["cached"] = True
        return step

    def _assemble(self, draft: dict, ses: ToolSession, event: dict, company_id: str, match: Match, trace):
        """Черновик модели → карточка контракта. Метки S… → индексы sources[], детерминированные поля — кодом."""
        by_label = {s.label: s for s in ses.sources}
        cited: list[str] = ["S1"]
        self._cited = cited  # порядок sources[] карточки — для перевода индексов обратно в метки при проверке

        def idx(label):
            if not label or label not in by_label:
                return None
            if label not in cited:
                cited.append(label)
            return cited.index(label)

        repairs: list[str] = []

        def ground(s: dict) -> dict | None:
            """Сверка цифр утверждения с источником до проверки (исправляет, а не только ругается).

            Если числа факта или позиции банка не встречаются в указанном источнике, ищем среди всех
            найденных агентом источников тот, где они есть (для bank — только аналитику банка), и
            перенаправляем ссылку. Не нашли — утверждение убирается: неподтверждённая цифра хуже,
            чем отсутствие утверждения. Факт без чисел и без источника становится выводом ИИ."""
            p, lbl = s.get("p"), s.get("src")
            if p == "fact" and lbl in by_label and by_label[lbl].kind == "bank" and BANK_VIEW.search(s.get("t", "")):
                # прогноз или ожидание из документа банка — это позиция банка, а не факт
                repairs.append("прогноз банка: факт → позиция банка")
                s, p = {**s, "p": "bank"}, "bank"
            if p not in ("fact", "bank"):
                return s
            nums = numbers_in(s.get("t", ""))
            ok = lambda src: (src.kind == "bank" or p == "fact") and nums <= numbers_in(src.text)  # noqa: E731
            if lbl in by_label and ok(by_label[lbl]):
                return s
            alt = next((src for src in ses.sources if nums and ok(src)), None)
            if alt:
                repairs.append(f"ссылка {lbl or '—'}→{alt.label}")
                return {**s, "src": alt.label}
            if not nums and lbl not in by_label:
                repairs.append("факт без источника → вывод ИИ")
                return {**s, "p": "ai", "src": None}
            repairs.append(f"убрано: «{s.get('t', '')[:50]}…»")
            return None

        def stmts(arr, allow=("t", "p", "strength", "cta")):
            out = []
            for s in arr or []:
                if not isinstance(s, dict) or not s.get("t"):
                    continue
                if s.get("p") not in ("fact", "bank", "ai"):
                    s = {**s, "p": "ai"}  # происхождение не указано — честнее считать выводом модели
                s = ground(s)
                if s is None:
                    continue
                if s["p"] == "ai":
                    s = {**s, "t": no_codes(s["t"])}
                st = {k: s[k] for k in allow if k in s and s[k] not in (None, "")}
                if st.get("strength") not in (None, "high", "medium", "low"):
                    st.pop("strength")
                cta = st.get("cta")
                if cta and (st["p"] != "ai" or not (isinstance(cta, dict) and cta.get("label") and cta.get("kind") in ("manager", "product"))):
                    st.pop("cta")  # кнопка действия — только у вывода ИИ: действие предлагает аналитик, а не источник
                i = idx(s.get("src"))
                if i is not None:
                    st["src"] = i
                out.append(st)
            return out

        b = draft.get("blocks", {})
        blocks = {k: stmts(b.get(k)) for k in ("what", "why", "risks", "opportunities", "actions")}
        key_points = stmts(draft.get("key_points"))[:3]
        bank_in = b.get("bank") if isinstance(b.get("bank"), dict) else {}
        bank_src = by_label.get(bank_in.get("src") or "")
        if bank_src and bank_src.kind != "bank":
            bank_src = None
        quote = str(bank_in.get("quote") or "").strip()
        plain = no_codes(str(bank_in.get("plain") or "").strip())
        if bank_src and not (quote and verbatim(quote, bank_src.text)):
            # цитата — только дословно: пересказ или дописанный хвост («…», при этом инфляция …) цитатой не показываем
            said, quote = quote, ""
            if not bank_src.stale:
                # дословное предложение-позицию берём из того же фрагмента, как в разборе без модели (agent/offline.py)
                ev = ses.event
                quote = _bank_quote(bank_src.text, _stems(" ".join([ev.get("title", ""), ev.get("text", "")[:1500], said])),
                                    title=bank_src.doc, event=ev, today=self.today, doc_date=bank_src.date)
            if quote:
                repairs.append("цитата банка не дословная → дословное предложение того же документа" if said else
                               "цитата банка → дословное предложение документа")
                if said and not plain and quote_matches(said, quote) and not house_view.revisions_in(said, self.today):
                    plain = no_codes(said)  # пересказ той же мысли — «проще говоря», вывод ИИ рядом с дословной цитатой
            elif said:
                repairs.append("позиция банка не совпала с текстом документа → убрана")
            if not quote and not bank_src.stale:
                bank_src = None  # дословной позиции банка во фрагменте нет — честная плашка
        if bank_src and bank_src.stale and not quote:
            idx(bank_src.label)
            bank = {"quote": "", "doc": f"Свежей оценки банка нет. Последняя — «{bank_src.doc or bank_src.name}» "
                                       f"от {bank_src.date}: старше 12 месяцев, в выводе не используется",
                    "date": bank_src.date, "stale": True}
        elif bank_src:
            idx(bank_src.label)
            bank = {"quote": quote, "doc": bank_src.doc or bank_src.name, "date": bank_src.date, "stale": bank_src.stale}
            if quote and plain and not bank_src.stale:
                bank["plain"] = plain  # «проще говоря» — вывод ИИ рядом с дословной цитатой
        else:
            bank = {"quote": "", "doc": NO_BANK_VIEW, "date": self.today.isoformat(), "stale": False}

        # расчёты — после текста и позиции банка, чтобы источники карточки шли в порядке чтения
        calc = [c | {"inputs": [{**i, "src": idx(i["src"])} for i in c["inputs"]]} for c in ses.calcs]
        p = profiles.get(company_id)
        dq = profiles.data_quality(p)
        cust = p["customers"]
        flow = (f"Отрасль известна для {profiles.pct(cust['known_industry_share'])} входящего потока."
                if cust["n"] and not dq else "")
        # отраслевая оценка банка: есть / устарела / нет — решает код, текст попадает в карточку дословно
        text_note = "Текст новости — только анонс источника, полный текст недоступен." if event.get("text_kind") == "summary" else ""
        coverage = " ".join(x for x in [ses.industry["note"], text_note, "; ".join(dq), flow] if x)
        sources = [{"name": by_label[lbl].name, "kind": by_label[lbl].kind, "date": by_label[lbl].date or self.today.isoformat(),
                    "url": by_label[lbl].url} for lbl in cited]
        card = {
            "id": f"{event['id']}-{company_id}",
            "type": draft.get("type", "info"), "impact": draft.get("impact", "low"),
            "topics": [t for t in draft.get("topics", []) if t] or event.get("topics", []) or ["отрасль"],
            "is_new": True, "published": event.get("published") or self.today.isoformat(),
            "title": draft.get("title") or event.get("title", ""),
            "takeaway": {r: no_codes(str(t)) for r, t in (draft.get("takeaway") or {}).items()}, "key_points": key_points,
            "why_short": no_codes(str(draft.get("why_short", ""))), "matched": match.matched,
            "sources": sources,
            "blocks": blocks | ({"calc": calc} if calc else {}) | {
                "bank": bank, "basis": {"features": match.matched + [f"Продукты банка: {', '.join(p['bank_products']) or 'нет'}"],
                                        "coverage": coverage}},
            "trace": [], "chat": [],
        }
        texts = {i: by_label[lbl].text for i, lbl in enumerate(cited)}
        rep = validate_card(card, today=self.today, source_texts=texts)
        detail = rep.summary() + (f" Исправлено до проверки: {'; '.join(repairs[:4])}." if repairs else "")
        trace.append({"kind": "check", "title": "Самопроверка (код)", "detail": detail, "ms": 0, "cost": 0})
        self._emit(trace)
        return card, rep
