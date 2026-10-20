"""Валидатор карточки: моки проходят, нарушения правил честности ловятся."""
import copy
import json
from datetime import date
from pathlib import Path

import pytest

from agent.contract import validate_card
from agent.contract.validator import numbers_in

MOCK = json.loads((Path(__file__).parents[1] / "frontend" / "mock_data.json").read_text(encoding="utf-8"))
CARDS = [c for cards in MOCK["insights"].values() for c in cards]
TODAY = date(2026, 10, 20)


def card(cid="i-rate-auto"):
    return copy.deepcopy(next(c for c in CARDS if c["id"] == cid))


@pytest.mark.parametrize("c", CARDS, ids=[c["id"] for c in CARDS])
def test_mock_cards_pass(c):
    rep = validate_card(c, today=TODAY)
    assert rep.ok, rep.errors


def test_missing_role_takeaway():
    c = card()
    del c["takeaway"]["treasurer"]
    assert not validate_card(c, today=TODAY).ok


def test_fact_without_source_is_error():
    c = card()
    c["blocks"]["what"][0].pop("src")
    rep = validate_card(c, today=TODAY)
    assert any("без источника" in e for e in rep.errors)


def test_src_out_of_range():
    c = card()
    c["key_points"][1]["src"] = 99
    assert any("src=99" in e for e in validate_card(c, today=TODAY).errors)


def test_bank_statement_must_cite_bank_source():
    c = card()
    c["key_points"][1]["src"] = 0  # источник 0 — пресс-релиз ЦБ, kind=news
    assert any("позиция банка" in e for e in validate_card(c, today=TODAY).errors)


def test_old_bank_material_must_be_stale():
    c = card("i-steel")
    c["blocks"]["bank"]["stale"] = False
    assert any("stale=false" in e for e in validate_card(c, today=TODAY).errors)


def test_invented_number_is_caught():
    c = card()
    texts = {0: "11 сентября ЦБ оставил ключевую ставку без изменений на уровне 14,0%."}
    assert validate_card(c, today=TODAY, source_texts=texts).ok
    c["blocks"]["what"][0]["t"] = "11 сентября ЦБ оставил ключевую ставку на уровне 15%."
    rep = validate_card(c, today=TODAY, source_texts=texts)
    assert any("15" in e for e in rep.errors)


def test_number_normalization():
    assert numbers_in("ставка 14,0% и 2 300 млрд") == {"14", "2300"}
    assert numbers_in("1,5 и 1.50") == {"1.5"}


def test_unknown_topic_rejected():
    c = card()
    c["topics"] = ["погода"]
    assert not validate_card(c, today=TODAY).ok


def test_all_profiles_build_views():
    """Все 115 профилей превращаются в вид для интерфейса (в данных бывают null)."""
    from agent import profiles
    for cid in profiles.all_ids():
        v = profiles.view(profiles.get(cid))
        assert v["id"] == cid and isinstance(v["data_quality"], list)
        profiles.features(profiles.get(cid))
