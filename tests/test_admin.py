"""Панель управления банка: доступ по токену, обзор, модерация, настройки в пределах, журнал действий."""
import importlib
import os
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token"  # noqa: S105 — токен только для теста
H = {"X-Admin-Token": TOKEN, "X-Admin-Name": quote("Проверка")}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    os.environ["APP_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))
    os.environ["APP_TODAY"] = "2026-10-20"
    os.environ["ADMIN_TOKEN"] = TOKEN
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as app_module
    importlib.reload(app_module)
    yield app_module, TestClient(app_module.app)
    os.environ.pop("ADMIN_TOKEN", None)


@pytest.fixture(autouse=True)
def no_lockout():
    """Неверные попытки одного теста не должны блокировать следующие."""
    from backend import admin
    admin._fails.clear()
    yield
    admin._fails.clear()


def test_admin_requires_token(ctx):
    _, c = ctx
    assert c.get("/api/admin/overview").status_code == 401
    assert c.get("/api/admin/overview", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert c.get("/api/admin/overview", headers=H).status_code == 200


def test_overview_flags_requests_and_offline_model(ctx):
    _, c = ctx
    c.post("/api/cards/i-rate-auto/request", json={"company_id": "U_25835", "role": "cfo", "kind": "manager", "label": "Связаться"})
    o = c.get("/api/admin/overview", headers=H).json()
    assert o["requests_new"] == 1 and any("Модель не подключена" in a["text"] for a in o["attention"])
    rid = c.get("/api/admin/requests", headers=H).json()[0]["id"]
    assert c.post(f"/api/admin/requests/{rid}", json={"status": "done"}, headers=H).status_code == 200
    assert c.get("/api/admin/overview", headers=H).json()["requests_new"] == 0


def test_overview_names_precomputed_cards_left_out(ctx):
    """Готовый разбор с прогнозом банка до пересмотра 11.09 клиентам не показан — панель говорит, что пересчитать."""
    _, c = ctx
    o = c.get("/api/admin/overview", headers=H).json()
    assert "U_25835/ev-e31c37f5eb-U_25835" in o["skipped_precomputed"]
    assert any("спрятано от клиентов" in a["text"] and "изменил прогноз" in a["text"] and "Пересчитать ИИ" in a["text"]
               for a in o["attention"])


def test_hidden_card_disappears_for_client(ctx):
    _, c = ctx
    assert "i-rate-auto" in [x["id"] for x in c.get("/api/companies/U_25835").json()["insights"]]
    r = c.post("/api/admin/cards/U_25835/i-rate-auto/hide", json={"hidden": True, "note": "проверка"}, headers=H)
    assert r.status_code == 200
    assert "i-rate-auto" not in [x["id"] for x in c.get("/api/companies/U_25835").json()["insights"]]
    listed = c.get("/api/admin/cards?company=U_25835", headers=H).json()
    assert any(x["id"] == "i-rate-auto" and x["hidden"] and not x["rerunnable"] for x in listed)   # пример команды — не из потока
    assert any(x["origin"] == "rules" and x["rerunnable"] for x in listed)                         # краткий разбор события из ленты
    c.post("/api/admin/cards/U_25835/i-rate-auto/hide", json={"hidden": False}, headers=H)
    assert "i-rate-auto" in [x["id"] for x in c.get("/api/companies/U_25835").json()["insights"]]


def test_settings_validated_applied_persisted_and_audited(ctx):
    app_module, c = ctx
    from agent import matching
    from agent.llm import ROUTING

    old = dict(ROUTING), matching.SELECT_MIN
    try:
        assert c.put("/api/admin/settings", json={"budget_rub": 400}, headers=H).status_code == 422   # выше доли сервиса
        assert c.put("/api/admin/settings", json={"routing": {"draft": "gpt"}}, headers=H).status_code == 422  # не из списка кейса
        assert c.put("/api/admin/settings", json={"unknown": 1}, headers=H).status_code == 422
        r = c.put("/api/admin/settings", json={"budget_rub": 12, "routing": {"draft": "qwen"}, "select_min": 3}, headers=H)
        assert r.status_code == 200
        assert app_module.llm.budget == 12 and ROUTING["draft"] == "qwen" and matching.SELECT_MIN == 3
        assert app_module.store.settings()["routing"] == {"draft": "qwen"}
        assert any(a["actor"] == "Проверка" and "budget_rub" in a["detail"] for a in c.get("/api/admin/audit", headers=H).json())
    finally:
        ROUTING.clear()
        ROUTING.update(old[0])
        matching.SELECT_MIN = old[1]


def test_learning_and_runs_endpoints(ctx):
    _, c = ctx
    s = c.get("/api/admin/learning", headers=H).json()
    assert {"trend", "routine", "lessons", "weights_readable", "constants", "enabled"} <= set(s)
    assert c.get("/api/admin/runs", headers=H).json() == []
    assert c.post("/api/admin/cards/U_25835/i-rate-auto/rerun", headers=H).status_code == 409  # пример команды — не из потока


def test_panel_closed_without_admin_token(ctx, monkeypatch):
    _, c = ctx
    monkeypatch.delenv("ADMIN_TOKEN")
    assert c.get("/api/admin/overview", headers=H).status_code == 503   # запрет по умолчанию


def test_every_admin_route_requires_token(ctx):
    _, c = ctx
    from backend.admin import _fails, router
    routes = router.routes
    assert len(routes) >= 12
    for r in routes:
        path = r.path.replace("{cid}", "U_25835").replace("{card_id}", "i-rate-auto").replace("{rowid}", "1")
        for method in r.methods - {"HEAD"}:
            assert c.request(method, path, json={}).status_code == 401, (method, path)
            _fails.clear()  # иначе после 10 отказов сработает ограничение попыток (429)


def test_failed_attempts_are_limited_and_token_not_logged(ctx, caplog):
    _, c = ctx
    from backend import admin
    with caplog.at_level("WARNING", logger="newsagent"):
        for _ in range(admin.FAIL_LIMIT):
            assert c.get("/api/admin/overview", headers={"X-Admin-Token": "guess-" + TOKEN}).status_code == 401
        assert c.get("/api/admin/overview", headers=H).status_code == 429   # подбор остановлен даже с верным токеном
    assert "отказ в доступе" in caplog.text and TOKEN not in caplog.text
    admin._fails.clear()
    assert c.get("/api/admin/overview", headers=H).status_code == 200
