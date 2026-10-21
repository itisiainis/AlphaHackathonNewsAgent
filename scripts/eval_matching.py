"""Замер отбора событий на размеченном наборе data/eval/relevance.json (критерий 12).

    python -m scripts.eval_matching          # точность, полнота, F1 и список ошибок

Без LLM и сети: проверяется слой отбора правилами — тот, что решает, на какие пары тратить бюджет модели.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agent import profiles
from agent.matching import match
from agent.pipeline import event_from_text

ROOT = Path(__file__).resolve().parents[1]


def evaluate(path: Path = ROOT / "data" / "eval" / "relevance.json") -> dict:
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    tp = fp = fn = tn = 0
    errors = []
    for it in items:
        ev = event_from_text(it["title"], it["text"])
        for cid, want in [(c, True) for c in it["relevant"]] + [(c, False) for c in it["irrelevant"]]:
            p = profiles.get(cid)
            m = match(ev, profiles.features(p), p["bank_products"])
            got = m.selected
            tp, fp, fn, tn = tp + (got and want), fp + (got and not want), fn + (want and not got), tn + (not got and not want)
            if got != want:
                errors.append(f"{'лишнее' if got else 'пропуск'}: {cid} × «{it['title']}» (оценка {m.score}; {', '.join(m.matched) or 'нет признаков'})")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pairs": tp + fp + fn + tn, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "errors": errors}


if __name__ == "__main__":
    r = evaluate()
    print(f"пар: {r['pairs']} · точность {r['precision']} · полнота {r['recall']} · F1 {r['f1']}")
    for e in r["errors"]:
        print("  " + e)
    sys.exit(0)
