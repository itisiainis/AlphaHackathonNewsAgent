"""Замер отбора событий на размеченных наборах (критерий 12).

    python -m scripts.eval_matching          # data/eval/relevance.json: точность, полнота, F1 и список ошибок
    python -m scripts.eval_matching --novel  # data/eval/novel_news.json: новые формулировки, доля пройденных проверок

Без LLM и сети: проверяется слой отбора правилами — тот, что решает, на какие пары тратить бюджет модели.
relevance.json — пары «новость × компания», на которых правила настраивали. novel_news.json — новые
формулировки («ставка НДС», «Руденко», медь на LME, раздел сайта вместо статьи): темы, отбор и, по флагу
calc, допустимые расчёты.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agent import profiles
from agent.matching import match
from agent.pipeline import event_from_text

ROOT = Path(__file__).resolve().parents[1]
NOVEL = ROOT / "data" / "eval" / "novel_news.json"


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


def check_item(it: dict, calc: bool = False) -> tuple[int, list[str]]:
    """Проверки одного пункта novel_news.json: (сколько проверок, список провалов)."""
    if it.get("html_fixture"):  # страница, которую чтение ссылки должно отклонить как «не статью»
        from agent.ingest.article import NotArticle, _extract, check_article

        _, text, _, kind = _extract((ROOT / it["html_fixture"]).read_text(encoding="utf-8"))
        try:
            check_article(text, kind)
        except NotArticle:
            return 1, [] if it.get("not_article") else [f"{it['id']}: страница отклонена как «не статья»"]
        return 1, [f"{it['id']}: страница принята как статья"] if it.get("not_article") else []
    title, _, body = it["text"].partition("\n")
    ev = event_from_text(title, body or title)
    n, errors = 0, []
    for t in it.get("expect_topics", []):
        n += 1
        if t not in ev["topics"]:
            errors.append(f"{it['id']}: нет темы «{t}» (темы: {', '.join(ev['topics']) or 'нет'})")
    for t in it.get("not_topics", []):
        n += 1
        if t in ev["topics"]:
            errors.append(f"{it['id']}: лишняя тема «{t}»")
    for cid, want in [(c, True) for c in it.get("select", [])] + [(c, False) for c in it.get("not_select", [])]:
        n += 1
        p = profiles.get(cid)
        m = match(ev, profiles.features(p), p["bank_products"])
        if m.selected != want:
            errors.append(f"{it['id']}: {'лишнее' if m.selected else 'пропуск'} {cid} (оценка {m.score}; "
                          f"{', '.join(m.matched) or 'нет признаков'})")
    allowed = it.get("allowed_calc_titles")
    if calc and allowed is not None:  # расчёты кодом: только допустимые для этой новости (включает W2-T4)
        from agent.impact import compute

        for cid in it.get("select", []):
            n += 1
            bad = [c["title"] for c in compute(ev, profiles.get(cid)) if not any(c["title"].startswith(a) for a in allowed)]
            if bad:
                errors.append(f"{it['id']}: {cid} — недопустимый расчёт: {'; '.join(bad)}")
    return n, errors


def evaluate_novel(path: Path = NOVEL, calc: bool = False) -> dict:
    """Доля пройденных проверок на новых формулировках — всего и по происхождению пунктов (audit / team-blind)."""
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    total = passed = 0
    by_origin: dict[str, list[int]] = {}
    errors = []
    for it in items:
        n, errs = check_item(it, calc)
        total, passed = total + n, passed + n - len(errs)
        o = by_origin.setdefault(it.get("origin", "audit"), [0, 0])
        o[0], o[1] = o[0] + n - len(errs), o[1] + n
        errors += errs
    return {"items": len(items), "checks": total, "passed": passed, "score": round(passed / total, 3) if total else 1.0,
            "by_origin": {k: {"passed": v[0], "checks": v[1], "score": round(v[0] / v[1], 3) if v[1] else 1.0}
                          for k, v in by_origin.items()},
            "errors": errors}


if __name__ == "__main__":
    if "--novel" in sys.argv:
        r = evaluate_novel(calc="--calc" in sys.argv)
        print(f"новые формулировки: пунктов {r['items']} · проверок {r['checks']} · пройдено {r['passed']} · балл {r['score']}")
        for o, v in r["by_origin"].items():
            print(f"  {o}: {v['passed']} из {v['checks']} ({v['score']})")
    else:
        r = evaluate()
        print(f"пар: {r['pairs']} · точность {r['precision']} · полнота {r['recall']} · F1 {r['f1']}")
    for e in r["errors"]:
        print("  " + e)
    sys.exit(0)
