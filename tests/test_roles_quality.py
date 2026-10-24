"""Единый конфиг ролей (data/roles.json) и проверка «нельзя ли вывод скопировать другой компании» (agent/quality.py)."""
from agent import channels, prompts, quality, roles

AUTO, DEVEL = "U_25835", "U_177581"   # опт автозапчастей (46% поступлений — автодилеры) и девелопер


def test_one_config_drives_channels_prompt_and_zones():
    ids = roles.ids()
    assert "accountant" in ids and set(channels.ORDER) == set(ids) == set(prompts.ROLES_RU)
    zone_ids = {z["id"] for z in roles.zones()}
    assert all(set(r["zones"]) <= zone_ids and r["actions"] and r["weights"] for r in roles.roles())
    assert "accountant — налоги" in roles.prompt_block() and "channels" not in roles.public()[0]


def test_card_zones_follow_topics():
    assert roles.zones_of(["ставка", "курс"]) == ["debt", "ved"]
    assert roles.zones_of(["налоги"]) == ["tax"] and roles.zones_of([]) == []


def card(take: str) -> dict:
    return {"takeaway": {"cfo": take}, "blocks": {"bank": {"quote": "", "date": "2026-10-01", "stale": False}, "actions": []}}


def test_generic_takeaway_is_copyable_to_any_company():
    c = card("Ставка останется высокой — закладывайте дорогой кредит в бюджет")
    assert quality.copyable_to(c, AUTO, [AUTO, DEVEL]) == [DEVEL] and quality.style_issue(c, AUTO)


def test_takeaway_with_company_marks_is_not_copyable():
    c = card("46% ваших поступлений — от автодилеров: дорогие автокредиты сократят их закупки у вас")
    assert quality.copyable_to(c, AUTO, [AUTO, DEVEL]) == [] and not quality.style_issue(c, AUTO)
