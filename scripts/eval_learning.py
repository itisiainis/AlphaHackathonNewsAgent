"""Воспроизведение замеров обучения на опыте (docs/experiments/learning.md). Без модели и без сети.

    python -m scripts.eval_learning stories    # память разборов: «тот же сюжет», старое и новое правило
    python -m scripts.eval_learning routine    # шаблоны в потоке за 21 день
    python -m scripts.eval_learning feedback   # симуляция оценок клиентов на реальных лентах (≈1 мин)
    python -m scripts.eval_learning lessons    # уроки проверки на реальных замечаниях
    python -m scripts.eval_learning all
    python -m scripts.eval_learning fetch 21   # (нужна сеть) собрать свежую историю вместо data/eval/history21.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

from agent.stories import MONTHS, conflict, jaccard, same_story, template_days, words

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"


def _old_words(title: str) -> set[str]:
    """Сравнение до замера: основы слов без чисел и месяцев (дни недели оставались)."""
    t = re.sub(r"\d+|" + MONTHS, " ", title.lower().replace("ё", "е"))
    return {w[:6] for w in re.findall(r"[а-яa-z]{3,}", t) if w not in {"это", "как", "что", "для", "при", "его", "она", "они",
                                                                        "года", "году", "все", "уже", "или"}}


# ---------- 1. память разборов ----------
def stories() -> None:
    for name in ("stories_dev", "stories_test"):
        pairs = json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))["pairs"]
        old = [p for p in pairs if jaccard(_old_words(p["a"]), _old_words(p["b"])) >= 0.6]
        new = [p for p in pairs if same_story(p["a"], p["b"], source_a=p["sa"], source_b=p["sb"], day_a=p["da"], day_b=p["db"])]
        same = sum(p["same"] for p in pairs)
        for label, sel in (("только сходство ≥ 0,6", old), ("сходство + запреты", new)):
            wrong = sum(not p["same"] for p in sel)
            bound = f", точность ≥ {1 - 3 / len(sel):.0%} с 95% уверенностью" if sel and not wrong else ""
            print(f"{name}: {label}: совпадений {len(sel)}, ошибочных {wrong} ({wrong / len(sel):.0%}){bound}; "
                  f"найдено {sum(p['same'] for p in sel)} из {same} верных пар выборки")
        reasons = collections.Counter(conflict(p["a"], p["b"], p["sa"], p["sb"], p["da"], p["db"]) for p in old if not p["same"])
        print(f"   почему прежние совпадения ошибочны: {dict(reasons)}")


# ---------- 3. шаблоны ----------
def _history() -> dict:
    return json.loads((EVAL / "history21.json").read_text(encoding="utf-8"))


def routine() -> None:
    h = _history()
    by_source = collections.defaultdict(list)
    for src, day, title in h["titles"]:
        by_source[src].append((day, title))
    flagged = [(src, title) for src, day, title in h["titles"] if template_days(title, by_source[src])]
    groups = collections.Counter((src, " ".join(sorted(words(t)))) for src, t in flagged)
    example = {(src, " ".join(sorted(words(t)))): t for src, t in flagged}
    print(f"постов {len(h['titles'])}, помечено шаблоном {len(flagged)} ({len(flagged) / len(h['titles']):.1%}), групп {len(groups)}")
    for key, n in groups.most_common():
        print(f"   {n:3d} {key[0]:18s} {example[key][:80]}")


# ---------- 2. оценки клиентов ----------
def feedback(seeds: int = 20) -> None:
    """Симуляция: реальные ленты компаний (115 профилей × события за 21 день после отбора), синтетические клиенты.
    Истинная полезность события = балл отбора + Σ по темам (вес роли + отклонение компании ~ N(0; 0,5)) / 2 —
    в том же масштабе, что формула ленты; в сценарии «формула откалибрована неверно» темы весят вчетверо больше
    (балл / 2 + Σ), и выигрывает то, что поднимает все темы сразу;
    в сценарии «противоположные вкусы» у компаний без кредита ставка и долг на 2 ниже, у остальных — на 2 выше,
    а оценивают только компании без кредита. Клиент видит 5 верхних событий и оценивает одно:
    «относительно» — лучше ли медианы показанных; «абсолютно» — лучше ли медианы всей своей ленты; 15% ошибок.
    Качество — NDCG@3 (на главной 3 события). Стратегии: без обучения; сумма голосов (прежняя формула);
    подъём над средней долей «полезно» (формула в agent/learning.py)."""
    from agent import profiles
    from agent.matching import rank

    roles = {r["id"]: r["weights"] for r in json.loads((ROOT / "frontend" / "mock_data.json").read_text(encoding="utf-8"))["roles"]}
    events = _history()["events"]
    topics = sorted({t for e in events for t in e["topics"]})
    feeds = {}
    for cid in profiles.all_ids():
        p = profiles.get(cid)
        f = [(e["id"], tuple(e["topics"]), m.score) for e, m in rank(events, profiles.features(p), p["bank_products"])][:40]
        if len(f) >= 8:
            feeds[cid] = f
    credit = {c: bool({"кредит", "лизинг"} & set(profiles.get(c)["bank_products"])) for c in feeds}
    M, K, FLIP, role = 3.0, 3.0, 0.15, "cfo"
    STRATS = ("без обучения", "сумма голосов", "подъём", "подъём, значимый")
    print(f"компаний с лентой: {len(feeds)} (с кредитом или лизингом {sum(credit.values())})")

    def ndcg3(order, util):
        gains = sorted(util.values(), reverse=True)
        lo = gains[-1]
        dcg = sum((util[i] - lo) / math.log2(k + 2) for k, i in enumerate(order[:3]))
        idcg = sum((g - lo) / math.log2(k + 2) for k, g in enumerate(gains[:3]))
        return dcg / idcg if idcg else 1.0

    def significant(v, p0, n_min=5, z_min=2.0):
        n = sum(v)
        return n >= n_min and abs(v[0] / n - p0) >= z_min * math.sqrt(p0 * (1 - p0) / n)

    def delta(strategy, cid, votes, totals):
        out = {}
        pooled_tot = [sum(v[k] for v in totals.values()) for k in (0, 1)]
        p_role = (pooled_tot[0] + 1) / (sum(pooled_tot) + 2)
        for t in topics:
            own = votes.get(cid, {}).get(t, [0, 0])
            pooled = [sum(v[t][k] for v in votes.values()) for k in (0, 1)]
            if strategy == "без обучения":
                out[t] = 0.0
            elif strategy == "сумма голосов":
                out[t] = M * (pooled[0] - pooled[1]) / (sum(pooled) + K) + M * (own[0] - own[1]) / (sum(own) + K)
            else:
                lift = (pooled[0] + K * p_role) / (sum(pooled) + K) - p_role
                if strategy == "подъём, значимый" and not significant(pooled, p_role):
                    lift = 0.0
                if cid in totals:
                    p_own = (totals[cid][0] + 1) / (sum(totals[cid]) + 2)
                    own_lift = (own[0] + K * (p_own + lift)) / (sum(own) + K) - p_own
                    if strategy == "подъём, значимый" and not significant(own, p_own):
                        own_lift = lift
                    out[t] = 2 * M * own_lift
                else:
                    out[t] = 2 * M * lift
        return out

    def run(seed, scenario, model, scale):
        rnd = random.Random(seed)  # noqa: S311 — симуляция, не криптография
        prefs = {}
        for c in feeds:
            prefs[c] = {t: roles[role].get(t, 0) + rnd.gauss(0, 0.5) for t in topics}
            if scenario == "противоположные вкусы":
                for t in ("ставка", "долг"):
                    prefs[c][t] += 2 if credit[c] else -2
        util = {c: {i: (s + sum(prefs[c][t] for t in ts) / 2 if scale == "верная" else s / 2 + sum(prefs[c][t] for t in ts))
                    for i, ts, s in feeds[c]} for c in feeds}
        pool = [c for c in feeds if not credit[c]] if scenario == "противоположные вкусы" else sorted(feeds)
        voters = rnd.sample(pool, min(15, len(pool)))
        others = [c for c in feeds if c not in voters and (scenario != "противоположные вкусы" or credit[c])]
        absolute = {c: statistics.median(util[c].values()) for c in feeds}
        res = {}
        for strat in STRATS:
            r2 = random.Random(seed + 1)  # noqa: S311
            votes = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
            totals = collections.defaultdict(lambda: [0, 0])

            def order(c, strat=strat, votes=votes, totals=totals):
                d = delta(strat, c, votes, totals)
                w = {t: roles[role].get(t, 0) + d[t] for t in topics}

                def key(item):
                    return -(item[2] + sum(w.get(t, 0) for t in item[1]) / 2)
                return [item[0] for item in sorted(feeds[c], key=key)]
            for _ in range(30):
                for c in voters:
                    top = order(c)[:5]
                    i = r2.choice(top)
                    thr = sorted(util[c][j] for j in top)[2] if model == "относительно" else absolute[c]
                    up = (util[c][i] > thr) != (r2.random() < FLIP)
                    totals[c][0 if up else 1] += 1
                    for t in dict((x[0], x[1]) for x in feeds[c])[i]:
                        votes[c][t][0 if up else 1] += 1
            res[strat] = (statistics.mean(ndcg3(order(c), util[c]) for c in voters),
                          statistics.mean(ndcg3(order(c), util[c]) for c in others))
        return res

    for scale in ("верная", "неверная"):
        for scenario in ("случайные клиенты", "противоположные вкусы"):
            for model in ("относительно", "абсолютно"):
                runs = [run(seed, scenario, model, scale) for seed in range(seeds)]
                print(f"[калибровка {scale}; {scenario}; оценка {model}] NDCG@3, Δ к «без обучения» ± 95% ДИ, {seeds} прогонов:")
                for strat in STRATS:
                    parts = []
                    for k, who in ((0, "оценивавшие"), (1, "остальные")):
                        vals = [r[strat][k] for r in runs]
                        diff = [r[strat][k] - r["без обучения"][k] for r in runs]
                        ci = 1.96 * statistics.stdev(diff) / len(diff) ** 0.5 if strat != "без обучения" else 0
                        parts.append(f"{who} {statistics.mean(vals):.3f} (Δ {statistics.mean(diff):+.3f} ± {ci:.3f})")
                    print(f"   {strat:14s} " + " | ".join(parts))


# ---------- 4. уроки ----------
LESSON_TRUTH = {  # хеш текста замечания → категории (разметка вручную по полному тексту)
    "6f7784f0": {"num"}, "f29dfd92": {"cite", "bank"}, "9493cc5b": {"num", "cite"}, "d9a1c377": set(),
    "cffbec38": {"num"}, "8816d375": {"num", "cite"}, "97cb204b": {"num", "cite"}, "bb8f4421": {"cite"},
    "904df4ec": {"fmt", "prov", "num"}, "19bacbb1": set(), "8a63736a": {"num"}, "dddbb7a7": {"cite"},
    "09392a6c": {"num"}, "6b153c89": {"num"}, "2ffcf73b": {"cite"}, "01da5c8c": {"fmt", "cite"},
    "b32d350c": {"cite"}, "d9b740c5": {"cite", "bank"}, "11ecae4d": {"num", "cite"}, "2405c915": set()}
OLD_LESSONS = [("num", r"не найдены в источнике|убрано: «"), ("jargon", r"код отрасли|ОКВЭД|оквэд"), ("jargon", r"ДКП|аббревиатур"),
               ("jargon", r"контрагент|входящий поток|поток»"), ("retell", r"повторяет заголовок|пересказ"),
               ("bank", r"позиция банка не совпала|ссылается на источник вида"), ("cite", r"без источника"),
               ("fmt", r"не прошла проверку кодом")]


def lessons() -> None:
    from agent.learning import lesson_categories

    msgs = []
    for f in sorted((ROOT / "data" / "cards").glob("*/*.json")):
        msgs += [t["detail"] for t in json.loads(f.read_text(encoding="utf-8")).get("trace", []) if t.get("kind") == "check"]
    db = ROOT / "data" / "cache" / "app.db"
    if db.exists():
        msgs += [x for (e,) in sqlite3.connect(db).execute("SELECT errors FROM runs") for x in json.loads(e or "[]")]
    labelled = [m for m in msgs if hashlib.md5(m.encode(), usedforsecurity=False).hexdigest()[:8] in LESSON_TRUTH]
    print(f"замечаний с разметкой: {len(labelled)} из {len(msgs)}")
    for label, classify in (("прежние правила", lambda m: {c for c, rx in OLD_LESSONS if re.search(rx, m, re.I)}),
                            ("новые правила", lesson_categories)):
        tp = fp = fn = 0
        for m in labelled:
            truth, got = LESSON_TRUTH[hashlib.md5(m.encode(), usedforsecurity=False).hexdigest()[:8]], classify(m)
            tp, fp, fn = tp + len(got & truth), fp + len(got - truth), fn + len(truth - got)
        print(f"   {label}: точность {tp / (tp + fp):.2f}, полнота {tp / (tp + fn):.2f}")


def fetch(days: int) -> None:
    import asyncio

    import httpx

    from agent.ingest import webcollector as wc
    from agent.ingest.collect import UA, _title, to_events

    wc.MAX_PAGES, wc.MAX_PER_CHANNEL = 80, 3000

    async def main():
        posts = []
        async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True, timeout=30) as http:
            for s in json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))["telegram"]:
                try:
                    posts += [(p, s) for p in await wc.fetch_recent_web(http, s["channel"], days * 24)]
                except Exception as e:  # канал недоступен — без него
                    print("пропущен", s["name"], e)
        return posts

    posts = asyncio.run(main())
    out = {"titles": [[s["name"], p.date.date().isoformat(), _title(p)] for p, s in posts],
           "events": [{k: e[k] for k in ("id", "title", "published", "source", "topics", "sectors")} | {"text": e["text"][:400]}
                      for e in to_events(posts)]}
    (EVAL / f"history{days}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"постов {len(out['titles'])}, событий {len(out['events'])}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "fetch":
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 21)
    for name, fn in (("stories", stories), ("routine", routine), ("lessons", lessons), ("feedback", feedback)):
        if cmd in (name, "all"):
            print(f"\n===== {name} =====")
            fn()
