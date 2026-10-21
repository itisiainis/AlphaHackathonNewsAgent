"""API: чтение, честные ошибки без ключа, валидация входа, лимит платных запросов, заголовки."""
import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["APP_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))
    os.environ["APP_TODAY"] = "2026-10-20"
    os.environ["RATE_LIMIT_PER_MIN"] = "3"
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_index_serves_ui_in_api_mode(client):
    r = client.get("/")
    assert r.status_code == 200 and '<script id="data" type="application/json">null</script>' in r.text
    assert r.headers["X-Content-Type-Options"] == "nosniff" and "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


def test_bootstrap_has_seed_cards_and_all_companies(client):
    d = client.get("/api/bootstrap").json()
    assert len(d["all_companies"]) == 115
    assert d["origins"]["U_25835"]["i-rate-auto"] == "seed"
    assert d["status"]["llm_online"] is False


def test_unknown_company_404(client):
    assert client.get("/api/companies/U_0").status_code == 404
    assert client.get("/api/companies/U_0/events").status_code == 404


def test_events_are_ranked_with_reasons(client):
    ev = client.get("/api/companies/U_25835/events").json()
    assert all(e["matched"] and e["score"] >= 2.5 and e["reasons"] for e in ev)


def test_analyze_without_key_gives_labelled_rules_card(client):
    r = client.post("/api/analyze", json={"company_id": "U_25835", "title": "ЦБ снизил ключевую ставку",
                                          "text": "Совет директоров Банка России снизил ключевую ставку с 16% до 14% годовых. "
                                                  "Регулятор допустил дальнейшее снижение в следующем году."})
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "rules" and "ключ" in d["reason"] and "без выводов ИИ" in d["notice"]
    c = d["card"]
    stmts = c["key_points"] + [s for b in ("what", "why", "risks", "opportunities", "actions") for s in c["blocks"][b]]
    assert stmts and all(s["p"] != "ai" for s in stmts), "без модели нет выводов ИИ"
    assert c["blocks"]["calc"][0]["value"].startswith("−20 млн ₽"), "расчёт работает без модели"
    assert "краткий разбор" in c["blocks"]["basis"]["coverage"]
    assert client.get("/api/metrics").json()["agent_runs"]["total"] == 0, "разбор правилами не считается запуском агента"
    origins = client.get("/api/companies/U_25835").json()["origins"]
    assert origins[c["id"]] == "rules"


def test_input_validation(client):
    assert client.post("/api/analyze", json={"company_id": "U_25835", "title": "x"}).status_code == 422
    assert client.post("/api/cards/i-rate-auto/feedback",
                       json={"company_id": "U_25835", "role": "admin", "vote": "up"}).status_code == 422


def test_feedback_counts_in_metrics(client):
    r = client.post("/api/cards/i-rate-auto/feedback", json={"company_id": "U_25835", "role": "cfo", "vote": "up"})
    assert r.status_code == 200
    assert client.get("/api/metrics").json()["feedback"]["useful"] == 1


def test_rate_limit_on_costly_endpoints(client):
    body = {"company_id": "U_25835", "title": "Новость для проверки лимита", "text": ""}
    codes = [client.post("/api/analyze", json=body).status_code for _ in range(5)]
    assert 429 in codes
