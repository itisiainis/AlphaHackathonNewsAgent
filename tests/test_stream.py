"""Поток шагов агента (SSE): шаги приходят по мере работы, итог — в конце, отказ без ключа — честный."""
import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient

from tests.test_agent_loop import BANK, EVENT, ScriptedLLM


@pytest.fixture()
def app_module(tmp_path):
    os.environ["APP_CACHE_DIR"] = str(tmp_path)
    os.environ["APP_TODAY"] = "2026-10-20"
    os.environ["RATE_LIMIT_PER_MIN"] = "100"
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as m
    return importlib.reload(m)


def events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


BODY = {"company_id": "U_25835", "title": EVENT["title"], "text": EVENT["text"]}


def test_stream_without_key_reports_error(app_module):
    r = TestClient(app_module.app).post("/api/analyze/stream", json=BODY)
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    assert ev[-1]["type"] == "error" and "ключ" in ev[-1]["detail"]


def test_stream_shows_steps_then_card(app_module):
    # сценарная модель цитирует фрагмент про ставку 14,0% — даём ей тот же маленький индекс, что в тестах цикла
    app_module.pipe.llm, app_module.pipe.bank = ScriptedLLM(), BANK
    ev = events(TestClient(app_module.app).post("/api/analyze/stream", json=BODY))
    steps = [e for e in ev if e["type"] == "step"]
    assert len(steps) >= 5, "шаги должны приходить по ходу работы, а не одним ответом"
    assert any(e["pending"] for e in steps), "пользователь видит, что агент делает прямо сейчас"
    assert [t["kind"] for t in steps[-1]["trace"]][-1] == "done"
    assert ev[-1]["type"] == "result" and ev[-1]["ok"] and ev[-1]["card"]["blocks"]["what"]
