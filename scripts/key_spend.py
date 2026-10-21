"""Сколько уже потрачено по общему ключу команды за сегодня (данные прокси организаторов).

Запуск: python -m scripts.key_spend   (ключ и адрес берутся из окружения / .env)
Ключ в вывод не попадает. Нужен перед любой пачкой вызовов: бюджет 500 ₽/день общий на всю команду.
"""
import os
import sys
from pathlib import Path

import httpx


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def key_spend() -> dict:
    base = os.environ["LLM_BASE_URL"].rstrip("/").removesuffix("/v1")
    r = httpx.get(f"{base}/key/info", headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}, timeout=20)
    r.raise_for_status()
    info = r.json().get("info", {})
    return {"spend": round(info.get("spend", 0), 2), "max_budget": info.get("max_budget"),
            "reset_at": info.get("budget_reset_at")}


if __name__ == "__main__":
    load_env(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("LLM_API_KEY"):
        sys.exit("ключ не задан")
    print(key_spend())
