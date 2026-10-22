"""Правило дымовой проверки scripts/smoke_demo.py: трасса не пишет «Все проверки пройдены» после проваленной проверки.

Заглушка модели в живой проверке всегда отвечает «замечаний нет», поэтому провалы проверок здесь воспроизводятся
сценарной моделью из тестов цикла агента — правило должно держаться и на них."""
import json
from pathlib import Path

from agent.loop import no_codes, refresh_saved, restate_final
from scripts.smoke_demo import trace_problem
from tests.test_agent_loop import CheckerSpy, ScriptedLLM, run

ROOT = Path(__file__).resolve().parent.parent


def step(kind, title, detail=""):
    return {"kind": kind, "title": title, "detail": detail}


DRAFT, REDRAFT = step("draft", "Черновик карточки", "14 утверждений"), step("draft", "Правка черновика", "16 утверждений")
CODE_OK = step("check", "Самопроверка (код)", "Все проверки пройдены")
CODE_FAIL = step("check", "Самопроверка (код)", "ошибок: 1. key_points: [...] is too long")
MODEL_OK = step("check", "Самопроверка (модель)", "замечаний нет")
MODEL_REMARK = step("check", "Самопроверка (модель)", "в источнике S1 нет даты 23 октября")
CLAIM_DONE = step("done", "Карточка готова", "Все проверки пройдены")


def test_code_error_before_final_claim_is_caught():
    # так выглядит готовый разбор U_25835 из аудита: ошибка кода, а итог — «Все проверки пройдены»
    why = trace_problem([DRAFT, CODE_OK, MODEL_REMARK, REDRAFT, CODE_FAIL, CLAIM_DONE])
    assert why.startswith("шаг 6 «Карточка готова»") and "ошибок: 1" in why


def test_unchecked_model_remark_before_final_claim_is_caught():
    # замечание проверяющей модели, правка проверена только кодом, итог заявляет, что пройдены все проверки
    assert "в источнике S1 нет даты" in trace_problem([DRAFT, CODE_OK, MODEL_REMARK, REDRAFT, CODE_OK, CLAIM_DONE])


def test_checker_not_run_is_a_failed_check():
    not_run = step("check", "Самопроверка (модель)", "не выполнена: проверяющая модель недоступна")
    assert trace_problem([DRAFT, CODE_OK, not_run, CLAIM_DONE])


def test_new_draft_passing_code_check_is_not_a_contradiction():
    # провал кода у первого черновика, правка, у неё «Все проверки пройдены» — это правда о новом черновике
    done = step("done", "Карточка готова", "Проверка кодом пройдена · вторая модель замечаний не нашла")
    assert trace_problem([DRAFT, CODE_FAIL, REDRAFT, CODE_OK, MODEL_OK, done]) == ""
    recheck = step("check", "Повторная самопроверка (модель)", "замечаний нет")
    assert trace_problem([DRAFT, CODE_OK, MODEL_REMARK, REDRAFT, CODE_OK, recheck, CLAIM_DONE]) == ""


def test_agent_loop_traces_stay_truthful_when_checks_fail():
    """Настоящий цикл агента: замечание при повторной проверке, проверяющая модель недоступна, правка упала."""
    class FailingRedraft(ScriptedLLM):
        def complete(self, role, messages, **kw):
            if "составь карточку" in messages[0]["content"] and "Замечания проверки" in messages[-1]["content"]:
                from agent.llm import LLMUnavailable
                raise LLMUnavailable("лимит токенов")
            return super().complete(role, messages, **kw)

    for llm in (ScriptedLLM(), ScriptedLLM(check_ok=False), CheckerSpy(), CheckerSpy(second_ok=False), CheckerSpy(fail_check=True),
                FailingRedraft(check_ok=False)):
        res = run(llm)
        trace = res.card["trace"] if res.card else res.trace
        assert trace and trace[-1]["kind"] == "done"
        assert trace_problem(trace) == "", (type(llm).__name__, [(t["title"], t["detail"][:60]) for t in trace])
        # итог, пересказанный по шагам, совпадает с итогом самого цикла — для свежих разборов restate_final ничего не меняет
        assert restate_final(trace) == trace, (type(llm).__name__, trace[-1]["detail"], restate_final(trace)[-1]["detail"])


# ---------- готовые разборы (data/cards), посчитанные до правки итога: итог — из их же шагов ----------
def test_unchecked_remark_is_restated_not_claimed():
    out = restate_final([DRAFT, CODE_OK, MODEL_REMARK, REDRAFT, CODE_OK, CLAIM_DONE])
    assert trace_problem(out) == ""
    assert out[-1]["detail"] == ("Проверка кодом пройдена · проверяющая модель оставила замечания: в источнике S1 нет "
                                 "даты 23 октября · черновик исправлен по замечаниям, повторно моделью не проверен")


def test_restate_keeps_steps_and_reports_rechecked_and_reverted_revisions():
    recheck = step("check", "Повторная самопроверка (модель)", "замечаний нет")
    trace = [DRAFT, CODE_OK, MODEL_REMARK, REDRAFT, CODE_OK, recheck, CLAIM_DONE]
    assert restate_final(trace)[:-1] == trace[:-1]
    assert restate_final(trace)[-1]["detail"] == "Проверка кодом пройдена · вторая модель замечаний не нашла"
    # правка не прошла проверку кодом — в карточке прежний черновик: замечание модели относится к нему, правки в нём нет
    reverted = step("check", "Правка черновика", "правка не прошла проверку кодом — оставлен прежний черновик")
    out = restate_final([DRAFT, CODE_OK, MODEL_REMARK, REDRAFT, CODE_FAIL, reverted, CLAIM_DONE])
    assert out[-1]["detail"] == "Проверка кодом пройдена · проверяющая модель оставила замечания: в источнике S1 нет даты 23 октября"
    failed = step("done", "Карточка не прошла проверку", "Проверка кодом: ошибок: 1")
    assert restate_final([DRAFT, CODE_FAIL, failed]) == [DRAFT, CODE_FAIL, failed]


def test_precomputed_cards_tell_the_truth_on_a_fresh_cache_and_after_a_restart(tmp_path, monkeypatch):
    """Свежий кеш и кеш прошлого запуска (APP_CACHE_DIR переживает перезапуск, в нём — разборы как они были в файлах):
    итог трассы пересказан по шагам, разбор с пересмотренным прогнозом банка убран из ленты."""
    import backend.app as app
    from backend.store import Store

    files = sorted((ROOT / "data" / "cards").glob("*/*.json"))
    for name, before in (("fresh", False), ("restart", True)):
        st = Store(tmp_path / name / "app.db")
        monkeypatch.setattr(app, "store", st)
        monkeypatch.setattr(app, "SKIPPED_PRECOMPUTED", {})
        if before:  # прошлый запуск, до проверок: разборы сохранены как есть
            for f in files:
                st.put_card(f.parent.name, json.loads(f.read_text(encoding="utf-8")), "agent")
        app._load_precomputed()
        loaded = [c for f in files if (c := st.card(f.parent.name, f.stem))]
        assert loaded and len(loaded) + len(app.SKIPPED_PRECOMPUTED) == len(files), name
        assert "U_25835/ev-e31c37f5eb-U_25835" in app.SKIPPED_PRECOMPUTED, "ставка 13,5% — до пересмотра 11.09"
        for card in loaded:
            assert trace_problem(card["trace"]) == "", (name, card["id"], card["trace"][-1]["detail"])
            shown = {k: v for k, v in card.items() if k not in ("trace", "sources", "matched", "chat")}
            shown["blocks"] = {k: v for k, v in card["blocks"].items() if k != "basis"} | {"coverage": card["blocks"]["basis"].get("coverage")}
            assert "ОКВЭД" not in json.dumps(shown, ensure_ascii=False), (name, card["id"])
        created = [st.card_meta(f.parent.name, f.stem) for f in files]
        app._load_precomputed()  # повторный запуск ничего не меняет и не поднимает карточки в ленте
        assert [st.card_meta(f.parent.name, f.stem) for f in files] == created, name
    raw = json.loads((ROOT / "data/cards/U_177581/ev-e31c37f5eb-U_177581.json").read_text(encoding="utf-8"))
    assert trace_problem(raw["trace"]), "файл остаётся как есть — меняется только итог при загрузке"


def test_saved_card_loses_industry_codes_but_not_meaning():
    assert no_codes("Компания работает в строительстве зданий (ОКВЭД 41.20.9) — отрасль") == "Компания работает в строительстве зданий — отрасль"
    assert no_codes("отраслевая оценка банка по ОКВЭД 47.71 в этой карточке недоступна") == "отраслевая оценка банка в этой карточке недоступна"
    assert no_codes("Автодилеры (45) и ставка 14,0% (на конец 2026)") == "Автодилеры и ставка 14,0% (на конец 2026)"
    raw = json.loads((ROOT / "data/cards/U_177581/ev-e31c37f5eb-U_177581.json").read_text(encoding="utf-8"))
    card = refresh_saved(raw)
    assert card["blocks"]["basis"]["coverage"].startswith("Отраслевой оценки банка по отрасли «Строительство зданий» нет — ")
    facts = [(s["t"], s["p"]) for k in ("what", "why") for s in raw["blocks"][k] if s["p"] != "ai"]
    assert facts == [(s["t"], s["p"]) for k in ("what", "why") for s in card["blocks"][k] if s["p"] != "ai"], "факты и цитаты — дословно"
    assert refresh_saved(card) == card
