"""«Данные в другом разрезе» из профиля: без модели, с графиком, с честной долей неизвестного."""
import pytest

from agent.slices import data_answer


def test_customers_by_industry_with_unknown_share():
    r = data_answer("U_25835", "Покупатели по отраслям")
    pts = dict(r["chart"]["points"])
    assert pts["Торговля и ремонт автотранспорта"] == 45.8 and pts["Отрасль неизвестна"] == 49.3
    assert sum(pts.values()) == pytest.approx(100, abs=2), "доли с неизвестной частью дают около 100%"
    assert all(s["t"].endswith(".") for s in r["a"]), "числа с запятой не ломают конец предложения"


def test_missing_suppliers_said_honestly():
    r = data_answer("U_200890", "А поставщики?")
    assert "нет данных о поставщиках" in r["a"][0]["t"] and "chart" not in r


def test_top_counterparties_before_generic_customers():
    assert data_answer("U_25835", "Крупнейшие покупатели")["chart"]["title"] == "Крупнейшие покупатели"


def test_bank_questions_go_to_model():
    assert data_answer("U_25835", "Что банк думает о курсе рубля?") is None


def test_chat_endpoint_answers_slices_without_key(tmp_path):
    import importlib
    import os

    from fastapi.testclient import TestClient
    os.environ.update(APP_CACHE_DIR=str(tmp_path), APP_TODAY="2026-10-20", RATE_LIMIT_PER_MIN="100")
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as m
    m = importlib.reload(m)
    c = TestClient(m.app)
    r = c.post("/api/cards/i-rate-auto/chat", json={"company_id": "U_25835", "q": "Покупатели по отраслям"})
    assert r.status_code == 200 and r.json()["chart"]["points"]
    # без модели свободный вопрос не отклоняется: ответ из материалов карточки, с пометкой
    r = c.post("/api/cards/i-rate-auto/chat", json={"company_id": "U_25835", "q": "Что банк думает о курсе?"})
    assert r.status_code == 200 and r.json()["a"][1]["p"] == "bank" and "материалов разбора" in r.json()["a"][-1]["t"]
    r = c.post("/api/cards/i-rate-auto/chat", json={"company_id": "U_25835", "q": "С другой стороны"})
    assert any(s["p"] == "ai" and "Возможн" not in s["t"] for s in r.json()["a"][1:-1]), "возможности и риски карточки"
