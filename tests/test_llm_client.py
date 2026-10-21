"""Клиент моделей: белый список, формат запроса, кеш, учёт стоимости, дневной лимит, офлайн."""
import json

import httpx
import pytest

from agent.llm import LLM, LLMUnavailable


def make(tmp_path, handler, budget=10.0, key="test-key"):
    return LLM(tmp_path, base_url="https://llm.example/v1", api_key=key, daily_budget_rub=budget,
               http=httpx.Client(transport=httpx.MockTransport(handler)))


def ok_handler(calls):
    def h(req: httpx.Request):
        calls.append(json.loads(req.content))
        assert req.url.path == "/v1/chat/completions" and req.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 500}})
    return h


def test_not_allowed_model_rejected_before_network(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="не входит"):
        make(tmp_path, ok_handler(calls)).complete("gpt", [{"role": "user", "content": "hi"}])
    assert calls == []


def test_request_cost_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PRICE_GLM", "100,200")       # ₽ за 1 млн токенов
    monkeypatch.setenv("LLM_MODEL_GLM", "glm-5.3-flash")
    calls = []
    llm = make(tmp_path, ok_handler(calls))
    msgs = [{"role": "user", "content": "привет"}]
    c1 = llm.complete("glm", msgs, json_mode=True)
    assert calls[0]["model"] == "glm-5.3-flash" and calls[0]["response_format"] == {"type": "json_object"}
    assert c1.cost_rub == pytest.approx((1000 * 100 + 500 * 200) / 1e6) and not c1.cached
    c2 = llm.complete("glm", msgs, json_mode=True)
    assert c2.cached and c2.cost_rub == 0 and len(calls) == 1, "повтор не должен идти в сеть и стоить денег"
    assert llm.spent_today() == pytest.approx(0.2)


def test_daily_budget_blocks_new_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PRICE_QWEN", "1000000,0")    # каждый вызов «стоит» 1000 ₽
    llm = make(tmp_path, ok_handler([]), budget=1.0)
    llm.complete("qwen", [{"role": "user", "content": "1"}])
    with pytest.raises(LLMUnavailable, match="лимит"):
        llm.complete("qwen", [{"role": "user", "content": "2"}])


def test_offline_serves_cache_only(tmp_path):
    calls = []
    online = make(tmp_path, ok_handler(calls))
    online.complete("gemma", [{"role": "user", "content": "x"}])
    offline = make(tmp_path, ok_handler(calls), key="")
    assert offline.complete("gemma", [{"role": "user", "content": "x"}]).cached
    with pytest.raises(LLMUnavailable, match="ключ"):
        offline.complete("gemma", [{"role": "user", "content": "новый запрос"}])


def test_provider_errors_become_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    llm = make(tmp_path, lambda req: httpx.Response(503, json={}))
    with pytest.raises(LLMUnavailable, match="не ответил"):
        llm.complete("deepseek", [{"role": "user", "content": "x"}])


def test_provider_cost_used_when_present(tmp_path):
    def h(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0355}})
    c = make(tmp_path, h).complete("glm", [{"role": "user", "content": "x"}])
    assert c.cost_rub == pytest.approx(0.0355)


def test_budget_exceeded_is_not_retried(tmp_path):
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "Budget has been exceeded! Current cost: 500.1"}})
    with pytest.raises(LLMUnavailable, match="бюджет"):
        make(tmp_path, h).complete("gemma", [{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_reasoning_ate_all_tokens(tmp_path):
    def h(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 6000, "cost": 0.5}})
    llm = make(tmp_path, h)
    with pytest.raises(LLMUnavailable, match="рассуждения"):
        llm.complete("qwen", [{"role": "user", "content": "x"}])
    assert llm.spent_today() == pytest.approx(0.5), "расход учитывается даже при пустом ответе"
