"""Цифры в ответах чата сверяет код: доля поступлений не превращается в долю выручки, выдуманное число убирается."""
import json
from datetime import date
from types import SimpleNamespace

import pytest

from agent import chat

AUTO = "U_25835"  # 46% поступлений — автодилеры; отрасль известна для 51%; крупнейший получатель платежей — 27%
CARD = {"title": "ЦБ сохранил ставку 14%", "takeaway": {"cfo": "Ставка 14% до конца года"}, "key_points": [],
        "sources": [{"name": "Банк России", "kind": "news", "date": "2026-10-23", "url": ""}],
        "blocks": {"what": [{"t": "Банк России сохранил ставку 14%", "p": "fact", "src": 0}], "why": [], "risks": [],
                   "opportunities": [], "actions": [], "bank": {"quote": "", "doc": "", "date": "2026-10-23", "stale": False}}}


def ask(statements):
    llm = SimpleNamespace(complete=lambda *a, **k: SimpleNamespace(text=json.dumps({"a": statements}), cost_rub=0, model="stub"))
    bank = SimpleNamespace(search=lambda *a, **k: [])
    return chat.answer(llm, bank, CARD, AUTO, "Подробнее", date(2026, 10, 23))


def test_flow_share_called_revenue_share_is_removed():
    r = ask([{"t": "51% выручки приходится на автодилеров", "p": "ai"},
             {"t": "27% выручки от одного клиента", "p": "ai"},
             {"t": "46% ваших поступлений — от автодилеров, ставка 14% давит на их закупки", "p": "ai"}])
    texts = [s["t"] for s in r["a"]]
    assert texts[0].startswith("46% ваших поступлений") and len(r["_dropped"]) == 2 and "Часть ответа убрана" in texts[-1]


def test_invented_number_is_removed_but_counts_and_years_stay():
    r = ask([{"t": "Убытки составят 37% маржи", "p": "ai"}, {"t": "Сделайте 3 шага до 2027 года", "p": "ai"}])
    assert [s["t"] for s in r["a"]][0] == "Сделайте 3 шага до 2027 года" and "37" in r["_dropped"][0]


def test_answer_with_nothing_verified_falls_back():
    with pytest.raises(ValueError):
        ask([{"t": "Убытки составят 37% маржи", "p": "ai"}])


def test_figures_passport_explains_meaning():
    from agent import profiles
    f = chat.figures(profiles.get(AUTO))
    assert "46% поступлений через банк — автодилеры и автосервисы" in f and "покрытие данных" in f and "27% платежей" in f
