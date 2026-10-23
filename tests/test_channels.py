"""Каналы влияния и факторы чувствительности: вывод краткого разбора строит код по данным банка, без модели."""
import re
from datetime import date
from pathlib import Path

import pytest

from agent import channels, profiles, sensitivity
from agent.contract import validate_card
from agent.impact import compute
from agent.kb import add_posts, load_bank_index
from agent.matching import match, tax_change
from agent.offline import CHANNELS_STEP, rules_card
from agent.pipeline import event_from_text, load_events, news_index

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 10, 21)
AUTO, METALS, FUEL = "U_25835", "U_200890", "U_19717"
LABEL = re.compile(r"^(?:Касается вас:|Событие в (?:вашей )?отрасли)")


def ev(title, text, published="2026-10-21"):
    return event_from_text(title, text, published=published)


def chans(event, cid, meetings=None):
    p = profiles.get(cid)
    m = match(event, profiles.features(p), p["bank_products"])
    return channels.build(event, m, p, compute(event, p), TODAY, meetings), m


HOLD = ev("Банк России сохранил ключевую ставку на уровне 14% годовых",
          "Совет директоров Банка России сохранил ключевую ставку на уровне 14% годовых.", published="2026-10-23")
CUT = ev("Банк России снизил ключевую ставку с 14% до 13,5% годовых",
         "Совет директоров Банка России снизил ключевую ставку с 14% до 13,5% годовых.", published="2026-10-23")
VAT = ev("Минфин предложил повысить НДС до 22%",
         "Минфин предложил повысить ставку НДС с 20% до 22% с 1 января 2027 года, чтобы сократить дефицит бюджета.")
CARS = ev("Продажи легковых автомобилей упали на 12%", "Продажи новых легковых автомобилей в России в сентябре упали на 12% год к году.")
MEETINGS = [date(2026, 10, 23), date(2026, 12, 18)]


# ---------- календарь ЦБ — из материалов банка ----------
def test_cb_meetings_come_from_bank_materials():
    bank = load_bank_index(ROOT / "data" / "bank" / "docs.json")
    assert {date(2026, 10, 23), date(2026, 12, 18)} <= set(channels.cb_meetings(bank.items))
    # следующее заседание — после самой новости: решение 23 октября ведёт к декабрьскому
    assert channels.next_meeting(MEETINGS, date(2026, 10, 21)) == date(2026, 10, 23)
    take = channels.takeaways(chans(HOLD, AUTO, MEETINGS)[0])["cfo"]
    assert "18 декабря" in take and "23 октября" not in take


def test_rate_decision_is_a_decision_not_a_forecast():
    assert channels.rate_decision(HOLD) == "hold" and channels.rate_decision(CUT) == "cut"
    assert channels.rate_decision(ev("Аналитики ждут сохранения ключевой ставки",
                                     "Аналитики ждут, что Банк России сохранит ключевую ставку на уровне 14%.")) == ""


# ---------- выводы под роль ----------
def test_rate_hold_for_a_borrower_is_a_risk_with_bank_view_and_next_meeting():
    chs, _ = chans(HOLD, AUTO, MEETINGS)
    t = channels.takeaways(chs)
    assert channels.card_type(chs, []) == "risk"
    assert "кредита и лизинга не снизится" in t["cfo"] and "прогноз Альфа-Банка — 14,0% на конец 2026" in t["cfo"]
    assert t["accountant"].startswith("Проценты по кредиту и лизингу в расходах")
    assert len({t[r] for r in channels.ROLES}) == 4, "у каждой роли свой вывод"


def test_rate_cut_shows_money_and_a_justified_action():
    chs, _ = chans(CUT, AUTO, MEETINGS)
    t, st = channels.takeaways(chs), channels.statements(chs)
    assert "экономия 5 млн ₽ в год на каждый 1 млрд ₽" in t["cfo"]
    assert st["actions"][0]["cta"]["kind"] == "manager" and st["actions"][0]["p"] == "rule"
    assert channels.card_type(chs, []) == "opportunity"


def test_company_without_debt_gets_an_indirect_effect_not_a_loan_story():
    t = channels.takeaways(chans(HOLD, METALS, MEETINGS)[0])
    assert all(x.startswith("Кредитов в Альфа-Банке у компании нет") for x in t.values())


def test_buyers_shock_reaches_liquidity_and_accounting():
    chs, _ = chans(CARS, AUTO)
    t, st = channels.takeaways(chs), channels.statements(chs)
    assert "46% ваших поступлений" in t["ceo"] and "потери 1,2 млрд ₽" in t["ceo"]
    assert "дебиторскую задолженность" in t["treasurer"] and "резерв по сомнительным долгам" in t["accountant"]
    assert st["risks"] and st["actions"][0]["cta"]["kind"] == "product"


def test_vat_change_is_selected_for_every_company_and_explained_per_role():
    assert tax_change(VAT["title"], VAT["text"])
    assert not tax_change("Льготный НДС на устрицы нужно отменить", "Льготный НДС на устрицы нужно отменить.")
    chs, m = chans(VAT, METALS)
    assert m.selected and any(r.get("kind") == "tax" for r in m.reasons)
    t = channels.takeaways(chs)
    assert "1,7%" in t["cfo"] and "с 1 января 2027" in t["cfo"] and t["cfo"].startswith("Если решение примут")
    assert "переходящие договоры" in t["accountant"]


def test_market_news_without_figures_says_so_and_names_what_changes():
    news = ev("ФАС возбудила 70 антимонопольных дел по рынку топлива", "ФАС с начала года возбудила 70 дел по рынку топлива.")
    take = channels.takeaways(chans(news, FUEL)[0])["ceo"]
    assert take.startswith("Изменения регулирования") and "«Нефтепереработка»" in take


# ---------- краткий разбор целиком: на ленте 21.10 — без ярлыков, проверка кодом пройдена ----------
@pytest.fixture(scope="module")
def world():
    events = load_events(ROOT / "data" / "news" / "events.json")
    return events, add_posts(load_bank_index(ROOT / "data" / "bank" / "docs.json"), events), news_index(events)


def test_briefs_on_the_feed_are_conclusions_not_labels(world):
    events, bank, news = world
    cards = []
    for cid in (AUTO, METALS, FUEL, "U_177581", "U_57706"):
        p = profiles.get(cid)
        for e in events:
            m = match(e, profiles.features(p), p["bank_products"])
            if m.selected:
                card, trace = rules_card(e, cid, m, bank, news, TODAY)
                assert card, trace[-2]["detail"]
                cards.append(card)
    assert len(cards) >= 15
    for c in cards:
        assert validate_card(c, today=TODAY).ok
        assert any(st["title"] == CHANNELS_STEP for st in c["trace"])
        assert set(c["takeaway"]) == {"cfo", "ceo", "treasurer", "accountant"}
        assert not any(s["p"] == "ai" for b in ("risks", "opportunities", "actions") for s in c["blocks"][b])
    labels = [c for c in cards if any(LABEL.match(t) for t in c["takeaway"].values())]
    assert len(labels) <= len(cards) // 20, [c["takeaway"]["cfo"] for c in labels]


def test_rule_statements_pass_style_checks():
    """Выводы по правилам проверяются, как выводы ИИ: без кодов отраслей и жаргона."""
    card = {"takeaway": {"cfo": "x", "ceo": "x", "treasurer": "x"}, "why_short": "x", "title": "t", "key_points": [],
            "blocks": {"what": [], "why": [], "risks": [{"t": "Риск для отрасли (45)", "p": "rule"}], "opportunities": [],
                       "actions": [], "bank": {"quote": "", "doc": "", "date": "2026-10-21", "stale": False}}}
    from agent.contract.validator import style_issues
    assert style_issues(card)


# ---------- факторы чувствительности ----------
def test_factors_come_from_bank_data_with_honest_coverage():
    auto = {f["id"]: f for f in sensitivity.factors(profiles.get(AUTO))}
    assert auto["rate"]["title"] == "Процентный риск" and auto["rate"]["level"] == "high"
    assert auto["buyers"]["detail"].startswith("46% поступлений")
    metals = {f["id"]: f for f in sensitivity.factors(profiles.get(METALS))}
    assert "rate" not in metals and metals["concentration"]["detail"] == (
        "Крупнейший плательщик — 23% поступлений через Альфа-Банк, пять крупнейших — 57%")
    for cid in profiles.all_ids():
        fs = sensitivity.factors(profiles.get(cid))
        assert fs and fs[-1]["id"] == "market" or any(f["id"] == "market" for f in fs)
        assert not any(re.search(r"\(\d{2}\)|ОКВЭД", f["title"] + f["detail"]) for f in fs)


def test_hits_and_diverse_briefs():
    fs = sensitivity.factors(profiles.get(FUEL))
    card = {"matched": ["кредит", "лизинг"], "blocks": {"calc": []}, "topics": ["ставка"], "type": "risk"}
    assert sensitivity.hits(card, fs) == ["rate"]
    assert sensitivity.hits({"topics": ["курс"], "blocks": {}}, fs) == ["fx"]
    # три новости о рынке топлива не вытесняют решение по ставке: брифы — по разным факторам
    events = load_events(ROOT / "data" / "news" / "events.json")
    p = profiles.get(FUEL)
    pairs = sorted([(e, m) for e in events if (m := match(e, profiles.features(p), p["bank_products"])).selected],
                   key=lambda x: -x[1].score)
    picked = sensitivity.diverse(pairs, 3)
    assert len({sensitivity.main_factor(m.reasons) for _, m in picked}) >= 2
