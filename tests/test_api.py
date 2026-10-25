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
    assert "base-uri 'none'" in r.headers["Content-Security-Policy"] and r.headers["X-Frame-Options"] == "DENY"


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
    assert "Краткий разбор" in c["blocks"]["basis"]["coverage"]
    assert client.get("/api/metrics").json()["agent_runs"]["total"] == 0, "разбор правилами не считается запуском агента"
    origins = client.get("/api/companies/U_25835").json()["origins"]
    assert origins[c["id"]] == "rules"


def test_input_validation(client):
    assert client.post("/api/analyze", json={"company_id": "U_25835", "title": "x"}).status_code == 422
    assert client.post("/api/cards/i-rate-auto/feedback",
                       json={"company_id": "U_25835", "role": "admin", "vote": "up"}).status_code == 422


def test_feedback_only_for_existing_cards(client):
    """Оценка меняет порядок ленты — по несуществующей компании или карточке её не принимаем."""
    assert client.post("/api/cards/i-rate-auto/feedback", json={"company_id": "U_0", "role": "cfo", "vote": "up"}).status_code == 404
    assert client.post("/api/cards/nope/feedback", json={"company_id": "U_25835", "role": "cfo", "vote": "up"}).status_code == 404


def test_news_refresh_is_for_bank_staff_only(client):
    """Сбор новостей ходит в сеть и тратит общий ключ — без токена сотрудника не запускается."""
    assert client.post("/api/news/refresh").status_code in (401, 503)


def test_feedback_counts_in_metrics(client):
    r = client.post("/api/cards/i-rate-auto/feedback", json={"company_id": "U_25835", "role": "cfo", "vote": "up"})
    assert r.status_code == 200
    assert client.get("/api/metrics").json()["feedback"]["useful"] == 1


def test_rate_limit_on_costly_endpoints(client):
    body = {"company_id": "U_25835", "title": "Новость для проверки лимита", "text": ""}
    codes = [client.post("/api/analyze", json=body).status_code for _ in range(5)]
    assert 429 in codes


def test_any_company_gets_briefs_without_model(client):
    """Открыл компанию — в ленте сразу разборы лучших событий (без модели, с пометкой), а не пустой экран."""
    # опт топливом: у опта металлами U_200890 единственное значимое событие уже разобрано ИИ (data/cards), а
    # «запас золота ЦБ» и «цифровое золото», дававшие ему краткие разборы, отбор больше не пропускает
    d = client.get("/api/companies/U_57706").json()
    rules = [c for c in d["insights"] if d["origins"][c["id"]] == "rules"]
    assert rules, "для компании с отобранными событиями должен быть краткий разбор"
    c = rules[0]
    assert all(s["p"] != "ai" for s in c["key_points"]) and c["takeaway"]["cfo"]


def test_brief_is_instant_and_keeps_agent_card(client):
    """Своя новость: краткий разбор сразу (без модели); готовый разбор ИИ по тому же событию не затирается."""
    body = {"company_id": "U_25835", "title": "ЦБ снизил ключевую ставку",
            "text": "Совет директоров Банка России снизил ключевую ставку с 16% до 14% годовых. Регулятор допустил дальнейшее снижение."}
    r = client.post("/api/brief", json=body).json()
    assert r["origin"] == "rules" and r["card"]["blocks"]["calc"]
    import backend.app as m
    m.store.put_card("U_25835", r["card"] | {"why_short": "разбор ИИ"}, "agent")
    again = client.post("/api/brief", json=body).json()
    assert again["origin"] == "agent" and again["card"]["why_short"] == "разбор ИИ"


def test_events_order_depends_on_role(client):
    cfo = [e["id"] for e in client.get("/api/companies/U_25835/events?role=cfo").json()]
    ceo = [e["id"] for e in client.get("/api/companies/U_25835/events?role=ceo").json()]
    assert sorted(cfo) == sorted(ceo), "тот же отбор, другой порядок под роль"


def test_pasted_news_can_be_upgraded_and_is_marked_as_users(client):
    """Аудит: «Полный разбор ИИ» по вставленной новости падал с «событие не найдено» (404)."""
    import backend.app as app_module
    app_module._hits.clear()  # лимит платных запросов в фикстуре — 3 в минуту, его тратят соседние тесты
    body = {"company_id": "U_177581", "title": "Минстрой продлил льготную ипотеку",
            "text": "Минстрой продлил программу льготной ипотеки на новостройки до конца 2027 года, ставка 8% сохранится."}
    b = client.post("/api/brief", json=body).json()
    assert b["user"] is True and b["event_id"]
    r = client.post(f"/api/companies/U_177581/events/{b['event_id']}/analyze")
    assert r.status_code == 200 and r.json()["card"]["id"] == b["card"]["id"]   # без ключа — краткий разбор, но не 404
    assert b["card"]["id"] in client.get("/api/companies/U_177581").json()["user_cards"]
    assert b["card"]["id"] in client.get("/api/bootstrap").json()["user_cards"]["U_177581"]


def test_company_has_sensitivity_factors_and_hits(client):
    d = client.get("/api/companies/U_25835").json()
    ids = [f["id"] for f in d["company"]["factors"]]
    assert ids[0] == "rate" and "buyers" in ids
    assert d["factor_hits"] and all(set(h) <= set(ids) for h in d["factor_hits"].values())
    assert set(client.get("/api/bootstrap").json()["factor_hits"]) >= {"U_25835", "U_177581", "U_200890"}
    r = client.post("/api/cards/i-rate-auto/feedback", json={"company_id": "U_25835", "role": "accountant", "vote": "up"})
    assert r.status_code == 200, "главный бухгалтер — допустимая роль"


def test_event_dropped_from_feed_is_rebuilt_from_its_card(client):
    """Краткий разбор остался в кеше, а событие выпало из потока (поток пересобран) — полный разбор всё равно запускается."""
    import backend.app as app_module

    card = {"id": "ev-gone000000-U_25835", "title": "АвтоВАЗ сократил дилерскую сеть в Москве", "published": "2026-10-20",
            "sources": [{"name": "Интерфакс", "kind": "news", "date": "2026-10-20", "url": "https://example.ru/a"}],
            "blocks": {"what": [{"t": "Дилеров стало меньше: с 15 до 6.", "p": "fact", "src": 0}]}}
    app_module.store.put_card("U_25835", card, "rules")
    ev = app_module._event("U_25835", "ev-gone000000")
    assert ev["id"] == "ev-gone000000" and "с 15 до 6" in ev["text"] and ev["url"] == "https://example.ru/a"


def test_top_events_get_ai_analysis_queued_in_background(client, monkeypatch):
    """Главное — без нажатия: быстрые выводы лучших событий ставятся в очередь на разбор ИИ, каждый — один раз."""
    import queue

    import backend.app as app_module

    q = queue.Queue()
    monkeypatch.setattr(app_module, "_pre_queue", q)
    monkeypatch.setattr(type(app_module.llm), "online", property(lambda self: True))
    cid = "U_25835"
    app_module._ensure_briefs(cid)
    rules = {c["id"] for c, o in app_module.store.cards(cid) if o == "rules"}
    first = q.qsize()
    app_module._precompute(cid)
    assert 0 < first <= app_module.PRECOMPUTE_TOP and q.qsize() == first  # повторное открытие не ставит заново
    queued = {f"{ev['id']}-{c}" for ev, c in list(q.queue)}
    assert queued <= rules


def test_post_ceiling_and_counter_cleanup(client):
    """Потолок на любые POST с одного адреса; счётчики адресов без запросов за минуту забываются."""
    from collections import defaultdict, deque

    import backend.app as a
    hits = defaultdict(deque)
    assert not any(a._over(hits, "x", 3, 0.0) for _ in range(3)) and a._over(hits, "x", 3, 1.0)
    assert not a._over(hits, "x", 3, 62.0), "через минуту лимит снова доступен"
    a._posts["203.0.113.9"] = deque([0.0])
    a._last_sweep[0] = 0.0
    a._sweep(1000.0)
    assert "203.0.113.9" not in a._posts
