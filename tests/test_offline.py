"""Разбор правилами без модели: только факты, данные компании, позиция банка по теме и расчёт."""
from agent import profiles
from agent.matching import match
from agent.offline import rules_card
from agent.pipeline import event_from_text
from tests.test_agent_loop import BANK, NEWS, TODAY


def card_for(title, text, cid="U_25835"):
    ev = event_from_text(title, text, published="2026-10-20")
    p = profiles.get(cid)
    return rules_card(ev, cid, match(ev, profiles.features(p), p["bank_products"]), BANK, NEWS, TODAY)


def test_rate_news_quotes_bank_on_same_topic_and_calculates():
    card, trace = card_for("ЦБ снизил ключевую ставку до 14%",
                           "Совет директоров Банка России снизил ключевую ставку с 16% до 14% годовых. Следующее заседание — 23 октября.")
    assert card is not None, trace[-2]["detail"]
    assert "14,0%" in card["blocks"]["bank"]["quote"] and card["sources"][1]["kind"] == "bank"
    assert card["blocks"]["calc"] and all(s["p"] != "ai" for s in card["key_points"])
    assert [t["kind"] for t in trace][-1] == "done" and all(t["cost"] == 0 for t in trace)


def test_unrelated_bank_fragment_is_not_quoted():
    card, _ = card_for("Рубль ослаб к юаню на 4,5% за неделю", "Курс юаня превысил 12 рублей, рубль ослаб на 4,5% за неделю.")
    assert card["blocks"]["bank"]["quote"] == "" and "не найдено" in card["blocks"]["bank"]["doc"]
    assert card["blocks"]["calc"][0]["title"] == "Валютные расчёты"
