"""Качество готовых разборов агента (data/cards): «нельзя ли этот вывод скопировать другой компании».

Запуск: python -m scripts.eval_cards
Для каждой карточки: есть ли в главном выводе данные именно этой компании, позиция банка с датой, конкретное действие,
и каким другим компаниям из набора вывод подошёл бы без правки. Без модели и без сети (agent/quality.py).
"""
import json
import sys
from pathlib import Path

from agent import quality

ROOT = Path(__file__).resolve().parents[1]


def evaluate(cards_dir: Path = ROOT / "data" / "cards") -> list[dict]:
    files = sorted(cards_dir.glob("*/*.json"))
    companies = sorted({f.parent.name for f in files})
    out = []
    for f in files:
        card = json.loads(f.read_text(encoding="utf-8"))
        r = quality.report(card, f.parent.name, companies)
        r["ok"] = bool(r["company_numbers"]) and r["bank_dated"] and r["action"] and not r["copyable_to"]
        out.append({"card": f"{f.parent.name}/{f.stem}", **r})
    return out


if __name__ == "__main__":
    rows = evaluate()
    for r in rows:
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['card']}: приметы компании {r['company_numbers'] or '—'}; "
              f"банк с датой: {'да' if r['bank_dated'] else 'нет'}; действие: {'да' if r['action'] else 'нет'}; "
              f"копируется: {', '.join(r['copyable_to']) or 'нет'}")
    print(f"\nПрошли: {sum(r['ok'] for r in rows)} из {len(rows)}")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)
