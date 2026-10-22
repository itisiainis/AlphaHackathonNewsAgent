"""Калькулятор эффекта: шоки из текста, формулы, ложные срабатывания, расчёт в карточке агента."""
import copy

import pytest

from agent import impact, profiles
from agent.contract import validate_card

AUTO = profiles.get("U_25835")     # опт автозапчастей: кредит + лизинг, ВЭД, 46% поступлений — автодилеры (45)
DEVEL = profiles.get("U_177581")   # девелопер: только лизинг, без ВЭД
METALS = profiles.get("U_200890")  # опт металлов: только РКО, поставщиков в выборке нет
DEALER = profiles.get("U_44245")   # розница легковых авто: 58% поступлений — автодилеры (45)
FUEL = profiles.get("U_19717")     # АЗС (47.30): топливо — закупка
OIL = profiles.get("U_57706")      # опт топлива, ВЭД
CARDBOARD = profiles.get("U_150736")  # картон (17.21): целлюлоза — закупка
CAFE = profiles.get("U_212593")    # рестораны: в выборке банка лишь 3 плательщика


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


def test_own_industry_price_scales_revenue_only_for_annual_change():
    # было: изменение цены за месяц переносилось на годовую выручку как «до −4,8 млрд ₽ — верхняя оценка»
    assert calcs("Цены на металлопрокат в России снизились на 8% за месяц.", METALS) == []
    c = calcs("Цены на металлопрокат в России снизились на 8% г/г.", METALS)[0]
    assert c["title"] == "Выручка, если ваши цены изменятся как у рынка" and c["value"] == "−4,8 млрд ₽ в год"
    assert c["strength"] == "low" and c["sign"] == "risk" and "верхняя оценка" not in c["value"]


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


# ---------- экономика, а не совпадение слов (случаи аудита: scratchpad/audit/insight-quality/calc_test.py) ----------
def titles(out):
    return [c["title"] for c in out]


def revenue(out):
    return [c for c in out if c["title"].startswith("Выручка") or "выручка" in c["title"]]


@pytest.mark.parametrize("text", [
    "Большинство аналитиков ожидают, что ЦБ снизит ключевую ставку на 50 б.п. на заседании 23 октября.",
    "Аналитики ожидают снижения ключевой ставки на 50 б.п.",
    "Если инфляция ускорится, ЦБ может повысить ключевую ставку на 100 б.п.",
])
def test_forecast_is_scenario_not_fact(text):
    out = calcs(text, AUTO)
    assert len(out) == 1
    c = out[0]
    assert c["title"].startswith("Если ставку ") and c["title"].endswith(", стоимость долга")
    assert c["sign"] == "neutral" and c["strength"] == "low" and impact.SCENARIO in c["assumptions"]


@pytest.mark.parametrize("text", ["Банк России снизил ключевую ставку на 50 б.п., до 13,5% годовых.",
                                  "ЦБ снизил ключевую ставку на 50 б.п. и может продолжить снижение.",
                                  "Банк России, как и ожидалось, снизил ключевую ставку на 50 б.п."])
def test_fact_with_forecast_words_elsewhere_stays_fact(text):
    c = calcs(text, AUTO)[0]
    assert c["title"] == "Стоимость долга" and c["sign"] == "opportunity" and c["value"].startswith("−5 млн ₽")


def test_sales_forecast_is_scenario():
    c = calcs("Аналитики ожидают, что продажи автомобилей в 2026 году упадут на 15%.", AUTO)[0]
    assert c["title"].startswith("Если рынок сократится на 15%, выручка от покупателей")
    assert c["sign"] == "neutral" and c["strength"] == "low"


@pytest.mark.parametrize("text,change", [
    ("Рубль ослаб к доллару на 3% за неделю, курс превысил 85 рублей.", 3),
    ("Курс юаня на Мосбирже снизился на 2% на фоне продаж экспортёров.", -2),
    ("Курс доллара вырос на 1,5% после снижения цен на нефть на 4%.", 1.5),
    ("Курс доллара вырос до 95 рублей, рубль ослаб на 10%", 10),
    ("Рубль укрепился к доллару на 2%, а юань подорожал на 1%.", -2),  # части расходятся — верим части про рубль
])
def test_fx_is_a_level_and_ruble_clause_wins(text, change):
    out = calcs(text, AUTO)
    assert titles(out) == ["Валютные расчёты"], "курс — уровень: недельное движение тоже считается; нефть второй раз не считается"
    c = out[0]
    assert c["formula"] == f"1 млрд ₽ × {impact._pc(change)}" and c["sign"] == ("risk" if change > 0 else "opportunity")


def test_monthly_change_is_not_scaled_to_annual_revenue():
    assert calcs("В августе продажи легковых автомобилей снизились на 7% к июлю.", DEALER) == []
    out = calcs("Выдачи ипотеки в сентябре сократились на 25%", DEVEL)
    assert not any("в год" in c["value"] for c in out)
    assert calcs("Цены на арматуру выросли на 5% за месяц.", METALS) == []


def test_month_year_on_year_is_annualised_with_caveat():
    out = calcs("Продажи новых легковых автомобилей в сентябре выросли на 12% год к году, сообщил «Автостат».", DEALER)
    assert len(revenue(out)) == 1, "одна оценка выручки на событие"
    c = out[0]
    assert c["title"].startswith("Выручка от покупателей") and c["sign"] == "opportunity" and impact.MONTH_YOY in c["assumptions"]
    c = calcs("Выдачи ипотеки в сентябре сократились на 25% в годовом выражении.", DEVEL)[0]
    assert c["title"] == "Выручка, если ваши продажи изменятся как у рынка" and c["strength"] == "low"
    assert impact.MONTH_YOY in c["assumptions"] and c["sign"] == "risk"


def test_own_price_since_start_of_year():
    c = calcs("Цены на новостройки в Москве выросли на 3% с начала года.", DEVEL)[0]
    assert c["title"] == "Выручка, если ваши цены изменятся как у рынка" and c["sign"] == "opportunity" and c["strength"] == "low"


@pytest.mark.parametrize("text", ["Биржевые цены на бензин АИ-95 выросли на 8% с начала месяца.",
                                  "Биржевые цены на бензин выросли на 8%"])
def test_fuel_price_is_purchase_cost_for_fuel_retailer(text):
    out = calcs(text, FUEL)
    assert titles(out) == ["Закупки топлива"] and out[0]["sign"] == "risk"
    assert out[0]["value"].startswith("+80 млн ₽ на каждый 1 млрд ₽ закупок топлива")
    assert not any(c["sign"] == "opportunity" or "+1,5 млрд" in c["value"] for c in out)


@pytest.mark.parametrize("text", ["Биржевые цены на бензин выросли на 8%, розничные — на 1,2%.",
                                  "Биржевые цены на бензин выросли на 8%. Розничные цены на АЗС выросли на 1,2%."])
def test_wholesale_and_retail_move_gives_one_margin_calc(text):
    out = calcs(text, FUEL)
    assert titles(out) == ["Маржа на топливе"]
    assert out[0]["value"].startswith("−68 млн ₽") and out[0]["sign"] == "risk"
    assert out[0]["formula"] == "1 млрд ₽ × (+1,2% − 8%)"


def test_fuel_price_forecast_is_purchase_scenario():
    c = calcs("Аналитики ожидают, что биржевые цены на бензин вырастут на 10%.", FUEL)[0]
    assert c["title"] == "Если цены вырастут на 10%, закупки топлива" and c["sign"] == "neutral" and c["strength"] == "low"


def test_retail_fuel_price_alone_is_revenue():
    c = calcs("Розничные цены на бензин на АЗС выросли на 1,2% с начала года.", FUEL)[0]
    assert c["title"] == "Выручка, если ваши цены изменятся как у рынка" and c["sign"] == "opportunity"


def test_pulp_price_is_purchase_cost_for_cardboard_maker():
    out = calcs("Целлюлоза подешевела на 6%.", CARDBOARD)
    assert titles(out) == ["Закупки целлюлозы"] and out[0]["sign"] == "opportunity"
    assert not any(c["sign"] == "risk" or "951" in c["value"] for c in out)
    assert calcs("Производство целлюлозы выросло на 5%.", CARDBOARD) == [], "объём сырья — не спрос на картон"


def test_metal_price_is_purchase_cost_for_builder():
    c = calcs("Цены на арматуру выросли на 5% за месяц.", DEVEL)[0]
    assert c["title"] == "Закупки металла" and c["sign"] == "risk", "уровень цены закупки: период не важен"


def test_exchange_price_is_benchmark_not_foreign_market():
    out = calcs("Медь на LME подорожала на 8%", METALS)
    assert titles(out) == ["Выручка, если ваши цены изменятся как у рынка"] and out[0]["strength"] == "low"


@pytest.mark.parametrize("text,fee", [("Минпромторг повысил утильсбор на импортные автомобили на 30%", "+300 млн ₽"),
                                      ("Минпромторг повысил утильсбор на импортные автомобили на 25%", "+250 млн ₽")])
def test_util_fee_is_a_cost_not_a_sales_change(text, fee):
    out = calcs(text, AUTO)
    assert titles(out) == ["Утильсбор"] and out[0]["sign"] == "risk" and out[0]["value"].startswith(fee)
    assert not any(c["title"].startswith("Выручка") for c in out)
    assert not any(x in c["value"] for c in out for x in ("+3,1 млрд", "+5,6 млрд"))


def test_importnye_is_not_import_volume():
    assert impact.sector_shocks("Минпромторг повысил утильсбор на импортные автомобили на 30%.") == []


def test_util_fee_up_verb_vyrastet():
    out = calcs("С 1 декабря утильсбор на легковые автомобили вырастет на 20%.", DEALER)
    assert titles(out) == ["Утильсбор"] and out[0]["value"].startswith("+200 млн ₽") and out[0]["sign"] == "risk"


def test_duties_export_import_level_and_cancel():
    c = calcs("Экспортная пошлина на нефтепродукты выросла с 5% до 7%.", OIL)[0]
    assert c["title"] == "Пошлина на экспорт" and c["sign"] == "risk" and "вывоза" in c["value"]
    assert calcs("Правительство повысит экспортную пошлину на нефть до 15%.", OIL) == [], "«до 15%» — уровень, не изменение"
    c = calcs("Правительство снизило импортную пошлину на автозапчасти с 15% до 10%.", AUTO)[0]
    assert c["title"] == "Пошлина на импорт" and c["sign"] == "opportunity" and c["value"].startswith("−50 млн ₽")
    c = calcs("Правительство отменило ввозную пошлину на автозапчасти 5%.", AUTO)[0]
    assert c["title"] == "Пошлина на импорт" and c["formula"] == "1 млрд ₽ × −5%" and c["sign"] == "opportunity"


def test_profit_tax_change():
    c = calcs("Ставка налога на прибыль вырастет с 25% до 27% с 2027 года.", AUTO)[0]
    assert c["title"] == "Налог на прибыль" and c["value"].startswith("+20 млн ₽") and c["sign"] == "risk"


@pytest.mark.parametrize("title,eff,sign", [("Стоимость долга", 1, "risk"), ("Закупки топлива", -1, "opportunity"),
                                            ("Пошлина на экспорт", 1, "risk"), ("Утильсбор", 1, "risk"),
                                            ("Налог на прибыль", -1, "opportunity"), ("Выручка от покупателей — x", -1, "risk"),
                                            ("Доход по депозитам", -1, "risk"), ("Маржа на топливе", 1, "opportunity")])
def test_sign_follows_economics(title, eff, sign):
    assert impact._sign(title, eff) == sign


def test_auto_sales_seed_uses_customer_share_once():
    """Карточка-пример i-auto-sales: −7% за август к июлю и в годовом выражении — одна оценка по доле покупателей."""
    out = impact.compute({"title": "Продажи новых легковых авто в августе упали на 7%",
                          "text": "В июле продажи легковых авто были около локального пика — 124 тыс. шт., но в годовом "
                                  "выражении рост уже нулевой (после +19% г/г во 2К26). В августе объём продаж снизился "
                                  "на 7% к июлю и также в годовом выражении."}, AUTO)
    assert len(out) == 1
    c = out[0]
    assert c["formula"] == "22,3 млрд ₽ × 46% × −7%" and c["value"] == "−717 млн ₽ в год"  # ≈ −0,72 млрд ₽
    assert impact.MONTH_YOY in c["assumptions"] and not any("верхняя оценка" in x["value"] for x in out)


def test_car_production_reaches_dealers_through_auto_chain():
    c = calcs("Производство легковых автомобилей в России в сентябре сократилось на 20% год к году.", AUTO)[0]
    assert c["title"].startswith("Выручка от покупателей") and "46%" in c["formula"]


def test_small_payment_sample_gives_no_flow_share_calc():
    for text in ["«Красное & Белое» пересматривает механизм принятия поставок продукции",
                 "Оптовые компании сообщили о снижении отгрузок ресторанам."]:
        out = calcs(text, CAFE)
        assert not any("под влиянием" in c["title"] or "покупател" in c["title"] for c in out)
        assert not any("24,1 млрд" in c["value"] or "24,2 млрд" in c["value"] for c in out)
    big = copy.deepcopy(CAFE)
    big["customers"]["n"] = 30  # та же доля при выборке побольше — масштаб показываем
    assert titles(calcs("Оптовые компании сообщили о снижении отгрузок ресторанам.", big)) == [
        "Выручка под влиянием: покупатели — оптовые компании"]
    text = impact.profile_text(CAFE)
    assert "по платежам через банк видно лишь 3 плательщиков — оценку не даём" in text
    assert "Группа: 3 юрлица; доля юрлиц группы по отраслям" in text
    assert "Покупатели — плательщиков в выборке банка: 744" in impact.profile_text(AUTO)


@pytest.mark.parametrize("text", ["Минфин предложил повысить НДС до 22%", "Ставки по ипотеке снизились на 2 п.п. вслед за решением ЦБ.",
                                  "Ставки аренды складов в Подмосковье выросли на 10%.", "ФРС снизила ставку на 25 б.п.",
                                  "Купонная ставка по облигациям составит 18% годовых."])
def test_not_key_rate(text):
    assert impact.rate_shock(text) is None


def test_key_rate_still_found():
    r = impact.rate_shock("Банк России снизил ключевую ставку с 16% до 14% годовых")
    assert r["delta"] == pytest.approx(-2) and r["level"] == 14 and not r["modal"]


def test_rate_sentence_is_not_a_sector_shock():
    out = calcs("Банк России снизил ключевую ставку на 100 б.п. Ставки по ипотеке снизились на 2 п.п.", DEVEL)
    assert titles(out) == ["Стоимость долга"]


# ---------- случаи из независимой проверки волны 1 ----------
def test_stock_exchange_news_is_not_a_commodity_price():
    """«бирж» совпадало с «Мосбиржа»: падение акций ММК давало металлотрейдеру «выручка −3 млрд ₽»."""
    text = "Акции ММК на Мосбирже упали на 5% после публикации отчётности металлургов."
    assert calcs(text, METALS) == [] and calcs(text, DEVEL) == []


def test_bond_yields_after_hold_do_not_turn_hold_into_cut():
    r = impact.rate_shock("Банк России сохранил ключевую ставку на уровне 16,5%. "
                          "После решения ЦБ по ставке доходность ОФЗ снизилась на 30 б.п.")
    assert r["delta"] == 0 and r["level"] == 16.5


def test_market_expectation_after_hold_is_not_a_decision():
    r = impact.rate_shock("Банк России сохранил ключевую ставку 16,5%. "
                          "Участники рынка закладывают снижение ключевой ставки на 100 б.п. в декабре.")
    assert r["delta"] == 0
    one = calcs("ЦБ сохранил ключевую ставку на уровне 16,5%, но рынок ждёт снижения на 100 б.п. в декабре.", AUTO)
    assert all(c["title"].startswith("Если") and c["sign"] == "neutral" for c in one)  # ожидание — сценарий, не факт


def test_retail_sales_volume_is_not_a_retail_price():
    out = calcs("Биржевые цены на бензин выросли на 8%, а продажи топлива на АЗС упали на 3% г/г.", FUEL)
    assert not any(c["title"].startswith("Маржа") for c in out)
    assert any(c["title"].startswith("Закупки") and c["sign"] == "risk" for c in out)


def test_foreign_market_named_in_lead_is_not_applied():
    """Цены в Германии (страна — в первой фразе) не меняют закупки российской АЗС."""
    out = impact.compute({"title": "В Германии бензин и дизель подорожали из-за налога",
                          "text": "Цены на бензин и дизель выросли на 37% с начала года."}, FUEL)
    assert out == []


def test_lme_price_is_global_benchmark():
    assert calcs("Медь на LME подорожала на 8%.", METALS)


def test_other_rate_in_the_same_sentence_is_not_the_key_rate_change():
    """«…до 15%, ставки по ипотеке снизятся на 2 п.п.» — 2 п.п. про ипотеку; изменение ключевой ставки не названо."""
    r = impact.rate_shock("Банк России снизил ключевую ставку до 15%, ставки по ипотеке снизятся на 2 п.п.")
    assert r["level"] == 15 and "delta" not in r
    assert calcs("Банк России снизил ключевую ставку до 15%, ставки по ипотеке снизятся на 2 п.п.", AUTO) == []
    r = impact.rate_shock("Банк России, как и ожидалось, снизил ключевую ставку на 50 б.п., до 16,5%.")
    assert r["delta"] == pytest.approx(-0.5) and r["level"] == 16.5 and not r["modal"]


def test_price_premium_is_not_a_metal_price():
    """Премия к цене LME для покупателей одной страны — надбавка, а не цена металла (было: «закупки −350 млн»)."""
    text = "Премия на алюминий на LME для японских покупателей упала на 35%."
    assert calcs(text, DEVEL) == [] and calcs(text, METALS) == []


def test_each_part_of_a_compound_sentence_keeps_its_own_number():
    out = calcs("Цены на новостройки в Москве выросли на 3% г/г, а стоимость арматуры — на 5%.", DEVEL)
    metal = next(c for c in out if c["title"] == "Закупки металла")
    assert metal["scale"]["per_bn"] == pytest.approx(50_000_000)  # 5% — арматура, а не 3% новостроек
    # отрасль названа в первой части, число — в обеих; «неожиданный» — не прогноз (история 21 дня)
    own = calcs("Вторичный рынок недвижимости Москвы демонстрирует неожиданный импульс: за сентябрь спрос на жилье "
                "подскочил на 5%, а общее число сделок за первые три квартала года увеличилось на 3%.", DEVEL)
    assert [(c["title"], c["formula"]) for c in own] == [
        ("Выручка, если ваши продажи изменятся как у рынка", "42,4 млрд ₽ × +5%")]


def test_wording_is_not_mistaken_for_price_or_missed_as_fall():
    # «по оценке» — не «цена», а оговорка прогноза; «упасть» — падение
    out = calcs("Продажи автомобилей в 2027 году сократятся на 15%, по оценке АЕБ.", AUTO)
    assert out and all(c["title"].startswith("Если рынок сократится на 15%") and c["sign"] == "neutral" for c in out)
    out = calcs("Продажи автомобилей могут упасть на 15%.", AUTO)
    assert out and all(c["title"].startswith("Если рынок сократится на 15%") for c in out)
    assert not any("размер эффекта в новости не указан" in " ".join(c["assumptions"]) for c in out)


def test_export_volume_and_foreign_markets_do_not_become_your_numbers():
    # «Центральную Азию» — не «цена»: объём экспорта дизеля не меняет закупки АЗС
    assert calcs("Российский экспорт дизеля в Центральную Азию вырос на 19% м/м.", FUEL) == []
    # экспорт машин из Китая — не продажи покупателей российского опта автозапчастей
    assert calcs("Экспорт автомобилей из Китая вырос на 20% г/г.", AUTO) == []


def test_scenario_titles_have_one_condition_and_assumptions_have_no_jargon():
    out = calcs("Аналитики ожидают, что цены на новостройки вырастут на 4%.", DEVEL)
    assert [c["title"] for c in out] == ["Если цены вырастут на 4%, выручка при ценах как у рынка"]
    texts = [s for p, t in ((AUTO, "Рубль ослаб к юаню на 5%."), (AUTO, "Продажи легковых автомобилей выросли на 12%."),
                            (OIL, "Экспортная пошлина на нефтепродукты вырастет на 10%."))
             for c in calcs(t, p) for s in c["assumptions"] + [i["t"] for i in c["inputs"]]]
    assert texts and not [s for s in texts if "ВЭД" in s or "эластичн" in s]
