"""Агентный цикл на сценарной модели (без сети и ключей): шаги, источники, самопроверка, безопасность."""
import json
from datetime import date

from agent import okved, profiles
from agent.kb import BM25Index
from agent.llm import Completion, LLMUnavailable
from agent.loop import Agent, verbatim
from agent.matching import match
from agent.offline import NO_BANK_VIEW
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


def run(llm, event=EVENT):
    p = profiles.get("U_25835")
    m = match(event, profiles.features(p), p["bank_products"])
    return Agent(llm, BANK, NEWS, today=TODAY).run(event, "U_25835", m)


def test_full_cycle_produces_valid_card():
    res = run(ScriptedLLM())
    assert res.card is not None, res.report.errors
    c = res.card
    assert [s["kind"] for s in c["sources"]] == ["news", "bank"]
    assert "calc" not in c["blocks"], "ставка не изменилась — расчёта эффекта нет"
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
    assert res.card and res.card["blocks"]["basis"]["coverage"].startswith("Отраслевой оценки банка по отрасли «Торговля и ремонт автотранспорта» нет")


def test_failed_revision_keeps_card_that_passed_code_check():
    """Проверяющая модель нашла замечание, а правка упала на лимите токенов — карточка не теряется."""
    class FailingRedraft(ScriptedLLM):
        def complete(self, role, messages, **kw):
            if "составь карточку" in messages[0]["content"] and "Замечания проверки" in messages[-1]["content"]:
                raise LLMUnavailable("модель glm: лимит токенов израсходован на рассуждения")
            return super().complete(role, messages, **kw)

    res = run(FailingRedraft(check_ok=False))
    assert res.card is not None
    assert any(t["title"] == "Правка черновика" and "не выполнена" in t["detail"] for t in res.trace)


def test_bank_quote_written_by_model_is_not_shown_as_quote():
    """Текст модели — не цитата: вместо него дословное предложение-позиция из того же фрагмента банка; чужой по смыслу
    текст не становится и «проще говоря»."""
    d = json.loads(json.dumps(DRAFT))
    d["blocks"]["bank"] = {"quote": "Отраслевой оценки влияния дефицита меди на металлотрейдеров в аналитике банка нет", "src": "S2"}
    res = run(ScriptedLLM(draft=d))
    bank = res.card["blocks"]["bank"]
    assert bank["quote"] and verbatim(bank["quote"], BANK.items[0]["text"]) and "меди" not in bank["quote"]
    assert "plain" not in bank
    assert any("не дословная" in t["detail"] for t in res.trace if t["title"] == "Самопроверка (код)")


# ---------- цитата банка: только дословно (аудит: U_25835 дописал в кавычки «при этом инфляция … ДКП») ----------
def page_1h26() -> str:
    """Стр. 3 «Макростатистики 1П26» (31.07) — источник цитаты U_25835; OCR поставил «,» вместо «.» перед «На конец года»."""
    from pathlib import Path
    docs = json.loads((Path(__file__).parents[1] / "data" / "bank" / "docs.json").read_text(encoding="utf-8"))
    return next(ch["text"] for d in docs if d["id"] == "bank-84899553" for ch in d["chunks"] if "заложить в прогноз" in ch["text"])


def test_quote_with_appended_tail_is_not_verbatim():
    quote = ("«Мы считаем возможным заложить в прогноз ещё два снижения ключевой ставки до конца текущего года. На конец "
             "года мы ожидаем ключевую ставку на уровне 13,5%», при этом инфляция по итогам 2026 г. — 7,5% г/г будет "
             "сдерживать смягчение ДКП")
    assert not verbatim(quote, page_1h26())
    # сама цитата без хвоста — дословна: регистр, ё/е, кавычки и путаница OCR «,» вместо «.» не мешают
    assert verbatim(quote.split("»,")[0] + "»", page_1h26())


def test_ocr_hyphenated_sentence_is_verbatim():
    src = "Если раньше мы ориенти-\nровались на ставку 13,5%, то теперь мы ожи- даем ее сохранения на теку\xadщем значении 14,0% до конца года."
    assert verbatim("Теперь мы ожидаем её сохранения на текущем значении 14,0% до конца года.", src)
    assert verbatim("…мы ориентировались на ставку 13,5%, то теперь…", src)
    assert not verbatim("мы ожидаем ее сохранения … до конца года", src), "пропуск внутри — не дословно"
    assert not verbatim("ожидаем снижения ставки", "Мы не ожидаем снижения ставки в этом году."), "отрезанное «не»"
    assert not verbatim("на уровне", src), "пара слов — не цитата"


def test_paraphrase_becomes_plain_next_to_verbatim_quote():
    d = json.loads(json.dumps(DRAFT))
    d["blocks"]["bank"] = {"quote": "Банк ждёт сохранения ставки на текущем значении 14,0% до конца года", "src": "S2"}
    res = run(ScriptedLLM(draft=d))
    bank = res.card["blocks"]["bank"]
    assert verbatim(bank["quote"], BANK.items[0]["text"]) and "14,0%" in bank["quote"]
    assert bank["plain"] == "Банк ждёт сохранения ставки на текущем значении 14,0% до конца года"


def test_no_position_in_fragment_shows_honest_plate():
    """Во фрагменте только расписание заседаний — ни цитаты модели, ни подстановки: плашка «позиции банка нет»."""
    bank = BM25Index([{**BANK.items[0], "text": "До конца года состоится два заседания ЦБ по ключевой ставке "
                                                "(23 октября и 18 декабря). Аукционы ОФЗ пройдут по средам."}])
    d = json.loads(json.dumps(DRAFT))
    d["blocks"]["bank"] = {"quote": "Банк ждёт сохранения ставки 14,0% до конца года", "src": "S2"}
    d["key_points"] = d["key_points"][:1] + [{"t": "Ставка — главный риск для долга", "p": "ai"}]
    p = profiles.get("U_25835")
    res = Agent(ScriptedLLM(draft=d), bank, NEWS, today=TODAY).run(EVENT, "U_25835", match(EVENT, profiles.features(p), p["bank_products"]))
    assert res.card["blocks"]["bank"] == {"quote": "", "doc": NO_BANK_VIEW, "date": TODAY.isoformat(), "stale": False}


# ---------- замкнутая самопроверка (по аудиту: правка не перепроверялась, метки источников расходились) ----------
class CheckerSpy(ScriptedLLM):
    """Запоминает, что видит проверяющая модель; первая проверка — замечание, вторая — по сценарию."""

    def __init__(self, second_ok=True, fail_check=False):
        super().__init__()
        self.seen, self.second_ok, self.fail_check = [], second_ok, fail_check

    def complete(self, role, messages, **kw):
        if "проверяющий" in messages[0]["content"]:
            self.calls.append(role)
            if self.fail_check:
                raise LLMUnavailable("проверяющая модель недоступна")
            self.seen.append(messages[1]["content"])
            ok = len(self.seen) > 1 and self.second_ok
            out = {"ok": ok, "issues": [] if ok else ["уточни горизонт прогноза"], "research_query": ""}
            return Completion(json.dumps(out, ensure_ascii=False), role, 100, 50, 0.01, 5, False)
        return super().complete(role, messages, **kw)


def test_checker_sees_source_labels_not_indices():
    llm = CheckerSpy()
    run(llm)
    draft_part = llm.seen[0].split("Источники:")[0]
    assert '"src": "S2"' in draft_part and '"src": 1' not in draft_part


def test_revised_draft_is_checked_again_and_trace_tells_the_truth():
    llm = CheckerSpy(second_ok=False)
    res = run(llm)
    assert len(llm.seen) == 2                                  # правка прошла повторную проверку моделью
    titles = [t["title"] for t in res.card["trace"]]
    assert "Повторная самопроверка (модель)" in titles
    assert "оставила замечания" in res.card["trace"][-1]["detail"]   # итог не пишет «все проверки пройдены»


def test_checker_failure_is_reported_not_hidden():
    res = run(CheckerSpy(fail_check=True))
    assert res.card is not None
    assert "второй моделью не выполнена" in res.card["trace"][-1]["detail"]


def test_trace_records_model_per_step():
    res = run(ScriptedLLM())
    assert any(t.get("model") for t in res.card["trace"] if t["kind"] in ("plan", "draft", "check"))
