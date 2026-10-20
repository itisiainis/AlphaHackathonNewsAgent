"""Клиент к разрешённым моделям кейса: белый список, кеш ответов, учёт стоимости и дневной лимит.

Что делает: единая точка вызова LLM для всего решения. Других путей к моделям в коде нет.

Как устроено:
- Белый список (ALLOWED) — только модели из приложения 3 кейса. Вызов модели не из списка
  падает до сетевого запроса: так запрет на другие LLM гарантирован кодом.
- Провайдер — OpenAI-совместимый эндпоинт (POST {base}/chat/completions) через httpx, без SDK
  и без агентных фреймворков. Адрес, ключ и фактические идентификаторы моделей у провайдера
  задаются переменными окружения (.env.example).
- Кеш: ответ на одинаковый запрос (модель + сообщения + параметры) хранится на диске и повторно
  не оплачивается. Это главный способ уложиться в 500 ₽/день при повторных показах.
- Стоимость: токены из ответа × тариф модели (₽ за 1 млн токенов, из окружения). Сумма за день
  копится в файле; при превышении LLM_DAILY_BUDGET_RUB новые платные вызовы отклоняются.
- Без ключа клиент работает в режиме offline: кешированные ответы отдаются, новые — нет
  (LLMUnavailable). Остальная система — сбор, отбор, поиск, валидация — работает без модели.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

# Ключ — роль модели в нашем решении; значение — имя по кейсу и переменная с ID у провайдера
ALLOWED = {
    "gemma": ("Gemma 4 26B A4B", "LLM_MODEL_GEMMA"),
    "glm": ("GLM-5.3-Flash", "LLM_MODEL_GLM"),
    "qwen": ("Qwen3.6 35B A3B", "LLM_MODEL_QWEN"),
    "deepseek": ("DeepSeek-V4.1-Flash", "LLM_MODEL_DEEPSEEK"),
    "qwen_vl": ("Qwen3-VL-30B-A3B-Instruct", "LLM_MODEL_QWEN_VL"),
    "kimi": ("Kimi-K2.6", "LLM_MODEL_KIMI"),
}

# Какая модель на какую задачу — по рекомендациям приложения 3
ROUTING = {
    "triage": "gemma",      # классификация и извлечение — невысокая сложность
    "draft": "glm",         # глубокий анализ и рассуждения
    "check": "deepseek",    # поиск взаимосвязей, проверка — другая модель, не та, что писала
    "chat": "qwen",         # универсальная
    "vision": "qwen_vl",    # сканы и графики
}


class LLMUnavailable(RuntimeError):
    """Модель недоступна: нет ключа, исчерпан лимит или ошибка провайдера."""


@dataclass
class Completion:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_rub: float
    ms: int
    cached: bool


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class LLM:
    def __init__(self, cache_dir: Path, base_url: str | None = None, api_key: str | None = None,
                 daily_budget_rub: float | None = None, http: httpx.Client | None = None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.budget = daily_budget_rub if daily_budget_rub is not None else _env_float("LLM_DAILY_BUDGET_RUB", 450)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = http or httpx.Client(timeout=120)

    @property
    def online(self) -> bool:
        return bool(self.base_url and self.api_key)

    # ---------- учёт расходов ----------
    def _spend_path(self) -> Path:
        return self.cache_dir / f"spend-{date.today().isoformat()}.json"

    def spent_today(self) -> float:
        p = self._spend_path()
        return json.loads(p.read_text())["rub"] if p.exists() else 0.0

    def _add_spend(self, rub: float) -> None:
        p = self._spend_path()
        p.write_text(json.dumps({"rub": round(self.spent_today() + rub, 4)}))

    @staticmethod
    def price(role: str) -> tuple[float, float]:
        """Тариф ₽ за 1 млн входных / выходных токенов: LLM_PRICE_<РОЛЬ>=in,out."""
        raw = os.environ.get(f"LLM_PRICE_{role.upper()}", "")
        try:
            a, b = (float(x) for x in raw.split(","))
            return a, b
        except ValueError:
            return 0.0, 0.0

    # ---------- вызов ----------
    def complete(self, role: str, messages: list[dict], max_tokens: int = 2000,
                 temperature: float = 0.2, json_mode: bool = False) -> Completion:
        if role not in ALLOWED:
            raise ValueError(f"модель «{role}» не входит в список разрешённых кейсом")
        model_id = os.environ.get(ALLOWED[role][1], "") or ALLOWED[role][0]
        body = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            c = json.loads(cache_file.read_text(encoding="utf-8"))
            return Completion(c["text"], c["model"], c["tokens_in"], c["tokens_out"], 0.0, 0, True)

        if not self.online:
            raise LLMUnavailable("ключ модели не задан (LLM_BASE_URL / LLM_API_KEY) — доступны только сохранённые ответы")
        if self.spent_today() >= self.budget:
            raise LLMUnavailable(f"дневной лимит {self.budget:.0f} ₽ исчерпан")

        t0 = time.monotonic()
        last = None
        for attempt in range(3):
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=body,
                                   headers={"Authorization": f"Bearer {self.api_key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("повторяемая ошибка", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                break
            except (httpx.HTTPError, ValueError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise LLMUnavailable(f"провайдер модели не ответил: {last}")

        text = (data["choices"][0]["message"].get("content") or "").strip()
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        pin, pout = self.price(role)
        cost = (tin * pin + tout * pout) / 1_000_000
        self._add_spend(cost)
        cache_file.write_text(json.dumps({"text": text, "model": model_id, "tokens_in": tin, "tokens_out": tout},
                                         ensure_ascii=False), encoding="utf-8")
        return Completion(text, model_id, tin, tout, cost, int((time.monotonic() - t0) * 1000), False)


def extract_json(text: str):
    """Достаёт JSON-объект или массив из ответа модели (терпимо к ```json и пояснениям вокруг)."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("в ответе модели нет JSON")
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    if end <= start:
        raise ValueError("в ответе модели незакрытый JSON")
    return json.loads(text[start:end + 1])
