"""Новые формулировки (data/eval/novel_news.json): правила отбора не ломаются на новостях, которых при настройке не было.

Пункты из аудита (origin: audit) проверяются строго — каждый отдельным тестом. Пункты, которые команда пишет вслепую
(origin: team-blind), — замер обобщения: по ним требуется общий балл не ниже порога, а не каждый пункт.
Расчёты (allowed_calc_titles) пока не проверяются — их включает W2-T4 после правок калькулятора.
"""
import json

import pytest

from scripts.eval_matching import NOVEL, check_item, evaluate_novel

ITEMS = json.loads(NOVEL.read_text(encoding="utf-8"))["items"]
BLIND_MIN = 0.7


def test_set_format():
    keys = {"text", "expect_topics", "not_topics", "select", "not_select", "allowed_calc_titles", "origin"}
    assert len(ITEMS) >= 30
    assert all(keys <= set(it) and it["origin"] in ("audit", "team-blind") for it in ITEMS)
    assert len({it["id"] for it in ITEMS}) == len(ITEMS)


@pytest.mark.parametrize("it", [it for it in ITEMS if it["origin"] == "audit"], ids=lambda it: it["id"])
def test_audit_item(it):
    n, errors = check_item(it)
    assert n and not errors, errors


def test_blind_items_score():
    r = evaluate_novel()
    blind = r["by_origin"].get("team-blind")
    if blind is None:
        pytest.skip("пунктов team-blind пока нет")
    assert blind["score"] >= BLIND_MIN, r["errors"]


@pytest.mark.skip(reason="допустимые расчёты проверяет W2-T4 после правок калькулятора (agent/impact.py)")
def test_calc_titles():
    r = evaluate_novel(calc=True)
    assert r["score"] == 1.0, r["errors"]
