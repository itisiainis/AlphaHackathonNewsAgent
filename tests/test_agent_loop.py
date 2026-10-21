"""Агентный цикл на сценарной модели (без сети и ключей): шаги, источники, самопроверка, безопасность."""
import json
from datetime import date

from agent import okved, profiles
from agent.kb import BM25Index
from agent.llm import Completion, LLMUnavailable
from agent.loop import Agent
from agent.matching import match
from agent.tools import ToolSession

TODAY = date(2026, 10, 20)
BANK = BM25Index([{
    "doc_id": "bank-rate", "title": "Ключевая ставка ЦБ РФ: ставка без изменения на 14,0%, риторика стала жестче",
    "date": "2026-09-11", "page": 1, "kind": "bank", "url": "",
    "text": "Мы ожидаем сохранения паузы на заседании 23 октября. Теперь мы ожидаем сохранения ставки "
            "на текущем значении 14,0% до конца года.",
}])
NEWS = BM25Index([])
EVENT = {"id": "ev-rate", "title": "ЦБ сохранил ключевую ставку 14%", "published": "2026-09-11",
         "source": "Банк России", "kind": "news", "url": "https://www.cbr.ru/press/",
         "text": "Совет директоров Банка России 11 сентября принял решение сохранить ключевую ставку на уровне 14% годовых."}
EVENT["topics"] = okved.topics_in(EVENT["title"] + " " + EVENT["text"])
EVENT["sectors"] = okved.sectors_in(EVENT["title"] + " " + EVENT["text"])

DRAFT = {
    "type": "risk", "impact": "high", "topics": ["ставка", "долг"],
    "title": "ЦБ сохранил ставку 14%",
    "takeaway": {"cfo": "Долг не подешевеет до конца года", "ceo": "Закладывайте дорогие деньги в план",
                 "treasurer": "Платежи по кредиту и лизингу не снизятся"},
    "why_short": "У компании кредит и лизинг",
    "key_points": [{"t": "У компании кредит и лизинг", "p": "ai"},
                   {"t": "Банк ждёт сохранения 14,0% до конца года", "p": "bank", "src": "S2"}],
    "blocks": {
        "what": [{"t": "11 сентября ЦБ сохранил ключевую ставку на уровне 14%.", "p": "fact", "src": "S1"}],
        "why": [{"t": "Стоимость кредита и лизинга зависит от ставки.", "p": "ai"}],
        "risks": [{"t": "Рефинансирование на лучших условиях маловероятно.", "p": "ai", "strength": "high"}],
        "opportunities": [],
        "actions": [{"t": "Обсудить фиксацию ставки.", "p": "ai", "cta": {"label": "Обсудить с менеджером", "kind": "manager"}}],
        "bank": {"quote": "Ожидаем сохранения ставки 14,0% до конца года", "src": "S2"},
    },
}


class ScriptedLLM:
    """Отвечает по сценарию: цикл выбирает инструменты, затем черновик и проверка."""

    def __init__(self, draft=DRAFT, check_ok=True):
        self.calls = []
        self.actions = [
            {"thought": "нужна новость", "plan": ["прочитать новость", "профиль", "позиция банка"], "action": "get_event", "args": {}},
            {"thought": "профиль", "action": "get_company_profile", "args": {}},
            {"thought": "чужая компания", "action": "get_company_profile", "args": {"company_id": "U_1595"}},
            {"thought": "позиция банка", "action": "search_bank_analytics", "args": {"query": "ключевая ставка пауза"}},
            {"thought": "достаточно", "action": "final"},
        ]
        self.draft, self.check_ok = draft, check_ok

    def complete(self, role, messages, **kw):
        self.calls.append(role)
        system = messages[0]["content"]
        if "проверяющий" in system:
            out = {"ok": self.check_ok, "issues": [] if self.check_ok else ["нет оценки ликвидности"], "research_query": ""}
        elif "составь карточку" in system:
            out = self.draft
        else:
            out = self.actions.pop(0)
        return Completion(json.dumps(out, ensure_ascii=False), role, 100, 50, 0.01, 5, False)


def run(llm):
    p = profiles.get("U_25835")
    m = match(EVENT, profiles.features(p), p["bank_products"])
    return Agent(llm, BANK, NEWS, today=TODAY).run(EVENT, "U_25835", m)


def test_full_cycle_produces_valid_card():
    res = run(ScriptedLLM())
    assert res.card is not None, res.report.errors
    c = res.card
    assert [s["kind"] for s in c["sources"]] == ["news", "bank"]
    assert c["key_points"][1]["src"] == 1 and c["blocks"]["what"][0]["src"] == 0
    assert c["blocks"]["bank"]["date"] == "2026-09-11" and not c["blocks"]["bank"]["stale"]
    kinds = [t["kind"] for t in c["trace"]]
    assert kinds[0] == "plan" and "tool" in kinds and "draft" in kinds and "check" in kinds and kinds[-1] == "done"
    assert "кредит" in c["matched"]


def test_tool_session_cannot_read_other_company():
    res = run(ScriptedLLM())
    denied = [e for e in res.tool_log if e["args"].get("company_id")]
    assert denied and not denied[0]["ok"], "попытка прочитать чужой профиль должна отклоняться"


def test_unknown_tool_rejected():
    ses = ToolSession("U_25835", EVENT, BANK, NEWS, TODAY)
    result, entry = ses.call("delete_everything", {})
    assert not entry["ok"] and "не разрешён" in result["error"]


def test_invented_number_is_removed_and_revision_requested():
    """Факт с выдуманной цифрой убирается; если блок «что произошло» опустел — черновик переписывается."""
    bad = json.loads(json.dumps(DRAFT))
    bad["blocks"]["what"][0]["t"] = "11 сентября ЦБ сохранил ключевую ставку на уровне 15%."
    llm = ScriptedLLM(draft=bad)
    res = run(llm)
    assert res.card is None, "карточка без единого факта о событии не публикуется"
    assert any("убрано" in t["detail"] for t in res.trace if t["title"] == "Самопроверка (код)")
    assert llm.calls.count("glm") >= 2 and any(t["title"] == "Правка черновика" for t in res.trace)


def test_wrong_citation_is_repointed_not_rejected():
    """Цифра есть в другом найденном источнике — ссылка исправляется, карточка проходит."""
    d = json.loads(json.dumps(DRAFT))
    # «23 октября» есть только во фрагменте аналитики банка (S2), а модель сослалась на новость (S1)
    d["key_points"][1] = {"t": "Следующее заседание по ставке — 23 октября", "p": "fact", "src": "S1"}
    res = run(ScriptedLLM(draft=d))
    assert res.card is not None, res.report.errors
    assert res.card["key_points"][1]["src"] == 1


def test_missing_provenance_defaults_to_ai():
    d = json.loads(json.dumps(DRAFT))
    d["blocks"]["actions"][0].pop("p")
    res = run(ScriptedLLM(draft=d))
    assert res.card is not None and res.card["blocks"]["actions"][0]["p"] == "ai"


def test_no_key_gives_honest_error():
    class Offline:
        def complete(self, *a, **k):
            raise LLMUnavailable("ключ модели не задан")
    res = run(Offline())
    assert res.card is None and "ключ" in res.error


def test_industry_assessment_rule_on_real_bank_meta():
    """Правило «есть ли отраслевая оценка банка» на настоящих метаданных 9 документов кейса."""
    from pathlib import Path

    from agent.kb import industry_assessment, load_bank_meta
    meta = load_bank_meta(Path(__file__).parents[1] / "data" / "bank" / "docs.json")
    assert industry_assessment(meta, ["46.72"], TODAY)["status"] == "stale", "опт металлами: только сталь 2021 года"
    assert industry_assessment(meta, ["24.41"], TODAY)["status"] == "fresh", "драгметаллы: обзор золота август 2026"
    assert industry_assessment(meta, ["45.31.1"], TODAY)["status"] == "none"
    assert "золота" not in industry_assessment(meta, ["46.72"], TODAY)["note"], "обзор золота — не оценка для опта металлами"


def test_card_basis_states_missing_industry_assessment():
    meta = [{"id": "x", "title": "Российская сталь", "date": "2021-04-15", "kind": "industry", "okved_prefixes": ["24.1"]}]
    p = profiles.get("U_25835")
    m = match(EVENT, profiles.features(p), p["bank_products"])
    res = Agent(ScriptedLLM(), BANK, NEWS, today=TODAY, bank_meta=meta).run(EVENT, "U_25835", m)
    assert res.card and res.card["blocks"]["basis"]["coverage"].startswith("Отраслевой оценки банка по ОКВЭД 45.31.1 нет")
