"""Калькулятор эффекта: шоки из текста, формулы, ложные срабатывания, расчёт в карточке агента."""
import copy

import pytest

from agent import impact, profiles
from agent.contract import validate_card

AUTO = profiles.get("U_25835")     # опт автозапчастей: кредит + лизинг, ВЭД, 46% входящего потока — авто (45)
DEVEL = profiles.get("U_177581")   # девелопер: только лизинг, без ВЭД
METALS = profiles.get("U_200890")  # опт металлов: только РКО


def calcs(text, p):
    return impact.compute({"title": "", "text": text}, p)


@pytest.mark.parametrize("text,level,delta", [
    ("Банк России снизил ключевую ставку с 16% до 14% годовых.", 14, -2),
    ("Совет директоров Банка России повысил ключевую ставку на 200 б.п., до 18% годовых.", 18, 2),
    ("Банк России сохранил ключевую ставку на уровне 14% годовых.", 14, 0),
    ("ЦБ понизил ключевую ставку на 0,5 процентного пункта.", None, -0.5),
])
def test_rate_shock(text, level, delta):
    r = impact.rate_shock(text)
    assert r.get("level") == level and r.get("delta") == pytest.approx(delta)


def test_rate_cut_lowers_debt_cost_per_billion():
    c = calcs("Банк России снизил ключевую ставку с 16% до 14% годовых.", AUTO)[0]
    assert c["title"] == "Стоимость долга" and c["sign"] == "opportunity"
    assert c["value"].startswith("−20 млн ₽ в год") and c["scale"]["per_bn"] == pytest.approx(-20e6)
    assert {i["src"] for i in c["inputs"]} == {"news", "profile"}


def test_no_debt_no_debt_calc():
    assert calcs("Банк России снизил ключевую ставку с 16% до 14% годовых.", METALS) == []


@pytest.mark.parametrize("text,sign", [("Рубль ослаб к юаню на 4,5% за неделю.", 4.5),
                                       ("Курс доллара вырос на 3% и достиг 95 рублей.", 3),
                                       ("Рубль укрепился к доллару на 2%.", -2)])
def test_fx_direction(text, sign):
    assert impact.fx_shock(text)["change_pct"] == sign


@pytest.mark.parametrize("text", ["Выручка компании выросла на 12% до 1,2 млрд рублей.",
                                  "Просроченная задолженность застройщиков выросла на 43% с начала года."])
def test_amounts_in_rubles_are_not_fx_or_demand(text):
    assert impact.fx_shock(text) is None and impact.sector_shocks(text) == []


def test_customer_sector_demand_drop_hits_revenue_by_flow_share():
    out = calcs("Продажи новых легковых автомобилей в России в сентябре упали на 12% год к году.", AUTO)
    assert len(out) == 1, "одна оценка выручки на один шок"
    c = out[0]
    assert "покупателей" in c["title"] and "46%" in c["formula"]
    assert c["value"] == "−1,2 млрд ₽ в год"  # 22,3 млрд × 46% × −12%
    assert any("известна" in a for a in c["assumptions"]), "покрытие данных названо в допущениях"


def test_foreign_market_not_applied_without_trade_with_that_country():
    assert calcs("Продажи недвижимости в Дубае упали на 17%.", DEVEL) == []


def test_own_industry_price_is_upper_bound_with_low_strength():
    c = calcs("Цены на металлопрокат в России снизились на 8% за месяц.", METALS)[0]
    assert c["strength"] == "low" and "верхняя оценка" in c["value"]


def test_duty_for_importer():
    c = calcs("Правительство с 1 декабря повышает утильсбор на легковые автомобили на 20%.", AUTO)[0]
    assert c["title"] == "Утильсбор" and c["value"].startswith("+200 млн ₽")


def test_profile_text_grounds_profile_inputs():
    from agent.contract.validator import numbers_in
    text = impact.profile_text(AUTO)
    for c in calcs("Продажи новых легковых автомобилей в России упали на 12%.", AUTO):
        for i in c["inputs"]:
            if i["src"] == "profile":
                assert numbers_in(i["t"]) <= numbers_in(text), i["t"]


def test_card_calc_block_validates_and_catches_invented_input():
    from tests.test_agent_loop import EVENT, ScriptedLLM, run
    cut = EVENT | {"text": "Совет директоров Банка России 11 сентября снизил ключевую ставку с 16% до 14% годовых."}
    res = run(ScriptedLLM(), cut)
    card = res.card
    calc = card["blocks"]["calc"][0]
    kinds = {card["sources"][i["src"]]["kind"] for i in calc["inputs"]}
    assert kinds == {"news", "stat"}, "входные данные ссылаются на новость и профиль компании"
    assert any(t["title"] == "calculate_impact()" for t in card["trace"]), "расчёт виден в трассе"
    bad = copy.deepcopy(card)
    bad["blocks"]["calc"][0]["inputs"][0]["t"] = "Ставка снижена до 9,5%"
    texts = {0: cut["title"] + "\n" + cut["text"]}
    assert not validate_card(bad, source_texts=texts).ok


def test_exposure_without_percent_in_news():
    out = calcs("Автодилеры сообщили о затоваривании складов и снижении закупок у дистрибьюторов автозапчастей.", AUTO)
    c = out[0]
    assert c["title"].startswith("Выручка под влиянием") and c["value"] == "≈ 10,2 млрд ₽ в год"  # 22,3 × 46%
    assert c["sign"] == "neutral" and "не прогноз потерь" in c["assumptions"][0]


def test_hold_rate_gives_no_calc():
    assert calcs("Банк России сохранил ключевую ставку на уровне 14% годовых.", AUTO) == []
