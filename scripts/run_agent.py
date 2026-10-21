"""Разбор событий агентом на реальных моделях из консоли — для проверки качества и предрасчёта ленты.

Примеры:
  python -m scripts.run_agent U_25835                 # лучшее по релевантности событие для компании
  python -m scripts.run_agent U_25835 --top 3         # три лучших события
  python -m scripts.run_agent U_25835 --title "ЦБ сохранил ставку 14%" --text "..."

Защита общего ключа команды: перед запуском проверяется, сколько уже потрачено за день по всему ключу,
и при превышении SHARED_STOP_RUB запуск отменяется. Личный лимит — LLM_DAILY_BUDGET_RUB (agent/llm.py).
Готовые карточки сохраняются в data/cards/<компания>/<id>.json — их подхватывает бэкенд как предрасчёт.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from agent.llm import LLM
from agent.pipeline import Pipeline, event_from_text
from scripts.key_spend import key_spend, load_env

ROOT = Path(__file__).resolve().parents[1]
SHARED_STOP_RUB = 300  # если команда за день уже потратила столько — не запускаемся


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("company")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--title")
    ap.add_argument("--text", default="")
    ap.add_argument("--today")
    ap.add_argument("--save", action="store_true", help="сохранить прошедшие проверку карточки в data/cards")
    a = ap.parse_args()

    load_env(ROOT / ".env")
    spend = key_spend()
    print(f"Ключ команды: потрачено сегодня {spend['spend']} из {spend['max_budget']} ₽")
    if spend["spend"] >= SHARED_STOP_RUB:
        print("Остановлено: общий бюджет команды почти исчерпан")
        return 2

    llm = LLM(ROOT / "data" / "cache" / "llm")
    pipe = Pipeline(llm, today=date.fromisoformat(a.today) if a.today else None)
    if a.title:
        events = [event_from_text(a.title, a.text)]
    else:
        events = [e for e, _ in pipe.ranked(a.company, a.top)]
    for ev in events:
        t0 = time.monotonic()
        res, m = pipe.analyze(ev, a.company)
        sec = time.monotonic() - t0
        cost = sum(t.get("cost", 0) for t in res.trace)
        print(f"\n=== {ev['title'][:100]}\n    признаки: {m.matched} | {sec:.0f} с | {cost:.3f} ₽ | ок: {res.card is not None}")
        for t in res.trace:
            print(f"    · {t['kind']:<8} {t['title'][:60]:<60} {t.get('ms', 0) / 1000:5.1f} с {t.get('cost', 0):.3f} ₽  {t['detail'][:90]}")
        if res.card:
            c = res.card
            print("    ВЫВОД:", json.dumps(c["takeaway"], ensure_ascii=False))
            for k in ("what", "why", "risks", "opportunities", "actions"):
                for s in c["blocks"][k]:
                    print(f"      [{k}/{s['p']}{'/' + str(s['src']) if 'src' in s else ''}] {s['t']}")
            print("    БАНК:", c["blocks"]["bank"]["quote"][:200], "|", c["blocks"]["bank"]["doc"][:60])
            print("    ОСНОВАНИЕ:", c["blocks"]["basis"]["coverage"][:200])
            if a.save:
                out = ROOT / "data" / "cards" / a.company / f"{c['id']}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            print("    ОШИБКИ:", res.report.errors[:5], res.error)
    print(f"\nЛичный расход за сегодня: {llm.spent_today():.2f} ₽ из {llm.budget:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
