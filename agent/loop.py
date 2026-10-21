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
from dataclasses import dataclass, field
from datetime import date

from . import profiles
from .contract import Report, validate_card
from .contract.validator import numbers_in
from .kb import BM25Index
from .llm import LLM, ROUTING, LLMUnavailable, extract_json
from .matching import Match
from .prompts import AGENT_SYSTEM, CHECK_SYSTEM, DRAFT_SYSTEM, ROLES_RU, tools_block
from .tools import ToolSession

MAX_TOOL_STEPS = 5
# Что показать пользователю, пока модель отвечает на шаг с этим названием
PENDING = {"План": "Модель составляет план и выбирает инструменты…", "Черновик карточки": "Модель пишет черновик карточки…",
           "Правка черновика": "Модель исправляет черновик по замечаниям проверки…",
           "Самопроверка (модель)": "Проверяющая модель (другая) читает черновик и источники…",
           "Шаг": "Модель выбирает следующий шаг…"}


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


class Agent:
    def __init__(self, llm: LLM, bank: BM25Index, news: BM25Index, today: date | None = None,
                 bank_meta: list[dict] | None = None):
        self.llm, self.bank, self.news = llm, bank, news
        self.bank_meta = bank_meta or []
        self.today = today or date.today()
        self.on_step = None  # обратный вызов для показа шагов в интерфейсе в реальном времени

    def _emit(self, trace: list, pending: str = "") -> None:
        """Отдаёт текущее состояние трассы наружу (поток SSE). pending — что агент делает прямо сейчас."""
        if self.on_step:
            self.on_step({"trace": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trace],
                          "pending": pending})

    # ---------- вспомогательное ----------
    def _ask(self, role: str, messages: list[dict], trace: list, kind: str, title: str, max_tokens=6000):
        """Вызов модели с одной повторной попыткой, если ответ не JSON."""
        self._emit(trace, PENDING.get(title, "Модель работает…"))
        c = self.llm.complete(ROUTING[role], messages, max_tokens=max_tokens, json_mode=True)
        cost, ms = c.cost_rub, c.ms
        try:
            data = extract_json(c.text)
        except ValueError:
            c2 = self.llm.complete(ROUTING[role], messages + [
                {"role": "assistant", "content": c.text},
                {"role": "user", "content": "Ответ не разобрался как JSON. Верни только JSON-объект."}],
                max_tokens=max_tokens, json_mode=True)
            cost, ms = cost + c2.cost_rub, ms + c2.ms
            data = extract_json(c2.text)
        trace.append({"kind": kind, "title": title, "detail": "", "ms": ms, "cost": round(cost, 4),
                      "_model": c.model, "_cached": c.cached})
        return data

    @staticmethod
    def _sources_block(ses: ToolSession) -> str:
        return "\n\n".join(f"[{s.label}] {s.kind} · {s.name} · {s.date}{' · УСТАРЕЛО' if s.stale else ''}\n{s.text[:1200]}"
                           for s in ses.sources)

    # ---------- основной сценарий ----------
    def run(self, event: dict, company_id: str, match: Match) -> Result:
        trace: list[dict] = []
        ses = ToolSession(company_id, event, self.bank, self.news, self.today, self.bank_meta)
        try:
            observations = self._explore(ses, match, trace)
            draft = self._draft(ses, match, observations, trace, issues=[])
            card, rep = self._assemble(draft, ses, event, company_id, match, trace)
            for _ in range(MAX_REVISIONS):
                verdict = self._check(card, ses, trace) if rep.ok else {"ok": False, "issues": [], "research_query": ""}
                issues = rep.errors + list(verdict.get("issues") or [])
                if rep.ok and verdict.get("ok", True):
                    break
                q = (verdict.get("research_query") or "").strip()
                if q:
                    res, entry = ses.call("search_bank_analytics", {"query": q})
                    trace.append({"kind": "research", "title": f"search_bank_analytics(\"{q}\")",
                                  "detail": f"повторный поиск по замечанию проверки: {_count(res)} фрагм.",
                                  "ms": entry["ms"], "cost": 0})
                    observations.append({"tool": "search_bank_analytics", "args": {"query": q}, "result": res})
                draft = self._draft(ses, match, observations, trace, issues=issues)
                card, rep = self._assemble(draft, ses, event, company_id, match, trace)
        except LLMUnavailable as e:
            return Result(None, Report(errors=[str(e)]), trace, ses.log, error=str(e))
        except (ValueError, KeyError, TypeError) as e:
            return Result(None, Report(errors=[f"ответ модели не разобран: {e}"]), trace, ses.log, error=str(e))

        trace.append({"kind": "done", "title": "Карточка готова" if rep.ok else "Карточка не прошла проверку",
                      "detail": rep.summary(), "ms": 0, "cost": 0})
        self._emit(trace)
        card["trace"] = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trace]
        return Result(card if rep.ok else None, rep, trace, ses.log)

    def _explore(self, ses: ToolSession, match: Match, trace: list) -> list[dict]:
        system = (AGENT_SYSTEM.replace("{tools}", tools_block(ses.TOOLS))
                  .replace("{max_steps}", str(MAX_TOOL_STEPS)))
        reasons = "; ".join(f"{r['feature']} ({r['why']})" for r in match.reasons) or "явных совпадений нет"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Событие: «{ses.event.get('title')}» ({ses.event.get('published')}).\n"
                     f"Компания: {ses.company_id}. Предварительный отбор сработал на признаки: {reasons}.\n"
                     "В первом ответе добавь поле \"plan\" — список шагов анализа."}]
        observations = []
        for step in range(MAX_TOOL_STEPS + 1):
            act = self._ask("agent", messages, trace, "plan", "План" if step == 0 else "Шаг")
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
                                     f" · результат: {_count(result)} объект(ов)")})
            observations.append({"tool": name, "args": act.get("args"), "result": result})
            self._emit(trace, "Модель выбирает следующий шаг…")
            messages += [{"role": "assistant", "content": json.dumps(act, ensure_ascii=False)},
                         {"role": "user", "content": f"Результат {name}: " + json.dumps(result, ensure_ascii=False)[:6000]}]
        return observations

    def _draft(self, ses, match, observations, trace, issues: list[str]) -> dict:
        profile = ses.get_company_profile()
        user = (f"Профиль компании: {json.dumps(profile, ensure_ascii=False)}\n\n"
                f"Признаки, на которые сработал отбор: {json.dumps(match.matched, ensure_ascii=False)}\n\n"
                f"Источники (ссылайся на метки):\n{self._sources_block(ses)}")
        if issues:
            user += "\n\nЗамечания проверки, которые нужно исправить:\n- " + "\n- ".join(issues[:10])
        system = DRAFT_SYSTEM.format(roles=", ".join(f"{k} — {v}" for k, v in ROLES_RU.items()))
        draft = self._ask("draft", [{"role": "system", "content": system}, {"role": "user", "content": user}],
                          trace, "draft", "Черновик карточки" if not issues else "Правка черновика", max_tokens=10000)
        n = sum(len(draft.get("blocks", {}).get(b, [])) for b in ("what", "why", "risks", "opportunities", "actions"))
        trace[-1]["detail"] = f"{n} утверждений, источников в работе: {len(ses.sources)}"
        self._emit(trace, "Код проверяет происхождение, ссылки и цифры…")
        return draft

    def _check(self, card: dict, ses: ToolSession, trace: list) -> dict:
        user = f"Черновик:\n{json.dumps(card, ensure_ascii=False)}\n\nИсточники:\n{self._sources_block(ses)}"
        verdict = self._ask("check", [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}],
                            trace, "check", "Самопроверка (модель)", max_tokens=6000)
        trace[-1]["detail"] = "замечаний нет" if verdict.get("ok") else "; ".join(verdict.get("issues", [])[:3])
        self._emit(trace)
        return verdict

    def _assemble(self, draft: dict, ses: ToolSession, event: dict, company_id: str, match: Match, trace):
        """Черновик модели → карточка контракта. Метки S… → индексы sources[], детерминированные поля — кодом."""
        by_label = {s.label: s for s in ses.sources}
        cited: list[str] = ["S1"]

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
                st = {k: s[k] for k in allow if k in s and s[k] not in (None, "")}
                if st.get("strength") not in (None, "high", "medium", "low"):
                    st.pop("strength")
                cta = st.get("cta")
                if cta and not (isinstance(cta, dict) and cta.get("label") and cta.get("kind") in ("manager", "product")):
                    st.pop("cta")
                i = idx(s.get("src"))
                if i is not None:
                    st["src"] = i
                out.append(st)
            return out

        b = draft.get("blocks", {})
        blocks = {k: stmts(b.get(k)) for k in ("what", "why", "risks", "opportunities", "actions")}
        key_points = stmts(draft.get("key_points"))
        bank_src = by_label.get((b.get("bank") or {}).get("src") or "")
        if bank_src and bank_src.kind == "bank":
            idx(bank_src.label)
            bank = {"quote": (b.get("bank") or {}).get("quote", ""), "doc": bank_src.doc or bank_src.name,
                    "date": bank_src.date, "stale": bank_src.stale}
        else:
            bank = {"quote": "", "doc": "Отраслевой оценки банка по этому событию нет — вывод на основе данных компании и новости",
                    "date": self.today.isoformat(), "stale": False}

        p = profiles.get(company_id)
        dq = profiles.data_quality(p)
        cust = p["customers"]
        flow = (f"Отрасль известна для {round(cust['known_industry_share'] * 100)}% входящего потока."
                if cust["n"] and not dq else "")
        # отраслевая оценка банка: есть / устарела / нет — решает код, текст попадает в карточку дословно
        coverage = " ".join(x for x in [ses.industry["note"], "; ".join(dq), flow] if x)
        sources = [{"name": by_label[lbl].name, "kind": by_label[lbl].kind, "date": by_label[lbl].date or self.today.isoformat(),
                    "url": by_label[lbl].url} for lbl in cited]
        card = {
            "id": f"{event['id']}-{company_id}",
            "type": draft.get("type", "info"), "impact": draft.get("impact", "low"),
            "topics": [t for t in draft.get("topics", []) if t] or event.get("topics", []) or ["отрасль"],
            "is_new": True, "published": event.get("published") or self.today.isoformat(),
            "title": draft.get("title") or event.get("title", ""),
            "takeaway": draft.get("takeaway", {}), "key_points": key_points,
            "why_short": draft.get("why_short", ""), "matched": match.matched,
            "sources": sources,
            "blocks": blocks | {"bank": bank, "basis": {"features": match.matched + [f"Продукты банка: {', '.join(p['bank_products']) or 'нет'}"],
                                                       "coverage": coverage}},
            "trace": [], "chat": [],
        }
        texts = {i: by_label[lbl].text for i, lbl in enumerate(cited)}
        rep = validate_card(card, today=self.today, source_texts=texts)
        detail = rep.summary() + (f" Исправлено до проверки: {'; '.join(repairs[:4])}." if repairs else "")
        trace.append({"kind": "check", "title": "Самопроверка (код)", "detail": detail, "ms": 0, "cost": 0})
        self._emit(trace)
        return card, rep
