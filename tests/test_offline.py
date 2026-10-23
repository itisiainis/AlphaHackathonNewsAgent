"""Разбор правилами без модели: только факты, данные компании, позиция банка по теме и расчёт."""
import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from agent import profiles
from agent.kb import add_posts, load_bank_index
from agent.loop import verbatim
from agent.matching import match
from agent.offline import NO_BANK_VIEW, POSITION_RE, SENT_RE, _bank_quote, _facts, _stems, bank_position, position_issue, rules_card
from agent.pipeline import event_from_text, load_events, news_index
from agent.tools import ToolSession
from tests.test_agent_loop import BANK, NEWS, TODAY

ROOT = Path(__file__).parents[1]
DOCS = {d["id"]: d for d in json.loads((ROOT / "data" / "bank" / "docs.json").read_text(encoding="utf-8"))}
AUDIT_DAY = date(2026, 10, 21)
SCHEDULE = "До конца года состоится два заседания ЦБ"  # процедурная фраза Бонд Брифа, 59 из 66 «позиций» до правки


def card_for(title, text, cid="U_25835"):
    ev = event_from_text(title, text, published="2026-10-20")
    p = profiles.get(cid)
    return rules_card(ev, cid, match(ev, profiles.features(p), p["bank_products"]), BANK, NEWS, TODAY)


def chunk(doc_id: str, needle: str) -> dict:
    return next(ch for ch in DOCS[doc_id]["chunks"] if needle in ch["text"])


def test_rate_hold_news_quotes_bank_position_verbatim():
    card, trace = card_for("ЦБ сохранил ключевую ставку 14%",
                           "Совет директоров Банка России сохранил ключевую ставку на уровне 14% годовых. Следующее заседание — 23 октября.")
    assert card is not None, trace[-2]["detail"]
    quote = card["blocks"]["bank"]["quote"]
    assert "14,0%" in quote and quote in BANK.items[0]["text"] and card["sources"][1]["kind"] == "bank"
    assert all(POSITION_RE.search(s) for s in SENT_RE.split(quote))
    assert [t["kind"] for t in trace][-1] == "done" and all(t["cost"] == 0 for t in trace)


def test_cut_news_does_not_show_hold_expectation_as_bank_position():
    """Аудит: под «снизил с 16% до 14%» позицией банка стояло «ЦБ ожидаемо сохранил ставку 14,0%». Ожидание паузы
    после снижения противоречит событию — честнее плашка; расчёт эффекта при этом остаётся."""
    card, trace = card_for("ЦБ снизил ключевую ставку до 14%",
                           "Совет директоров Банка России снизил ключевую ставку с 16% до 14% годовых. Следующее заседание — 23 октября.")
    assert card is not None, trace[-2]["detail"]
    assert card["blocks"]["bank"] == {"quote": "", "doc": NO_BANK_VIEW, "date": TODAY.isoformat(), "stale": False}
    assert card["blocks"]["calc"] and all(s["p"] != "ai" for s in card["key_points"])


def test_unrelated_bank_fragment_is_not_quoted():
    card, _ = card_for("Рубль ослаб к юаню на 4,5% за неделю", "Курс юаня превысил 12 рублей, рубль ослаб на 4,5% за неделю.")
    assert card["blocks"]["bank"]["quote"] == "" and card["blocks"]["bank"]["doc"] == NO_BANK_VIEW
    assert card["blocks"]["calc"][0]["title"] == "Валютные расчёты"


# ---------- фильтр позиции банка на настоящих фрагментах аналитики (data/bank/docs.json) ----------
EV_PRESS = {"title": "23 октября в 15:00 состоится пресс-конференция по итогам заседания Совета директоров "
                     "по денежно-кредитной политике", "text": ""}


def test_bond_brief_schedule_is_not_a_position():
    ch = chunk("bank-87dc1752", SCHEDULE)
    stems = _stems(EV_PRESS["title"] + " ключевая ставка")
    assert _bank_quote(ch["text"], stems, title=DOCS["bank-87dc1752"]["title"], event=EV_PRESS,
                       today=AUDIT_DAY, doc_date="2026-09-14") == ""


def test_rates_review_gives_current_position_verbatim():
    ch = chunk("bank-92a96805", "мы ожидаем ее сохранения")
    q = _bank_quote(ch["text"], _stems(EV_PRESS["title"] + " ключевая ставка"), title=DOCS["bank-92a96805"]["title"],
                    event=EV_PRESS, today=AUDIT_DAY, doc_date="2026-09-11")
    assert q.startswith("Мы ожидаем сохранения паузы на заседании 23 октября") and "14,0% до конца года." in q
    assert re.sub(r"\s+", " ", q) in re.sub(r"\s+", " ", ch["text"]), "цитата — дословный кусок фрагмента"


def test_ocr_garbage_and_fused_headings_are_not_quoted():
    """Аудит: U_34911 — «2 2 5 2 х 2 Е 8 é» позицией банка; U_200890 — «Какая поддержка котировок … сохраняется
    Диверсификация резервов…» (заголовок, сросшийся с текстом)."""
    ofz = chunk("bank-6c3a4882", "2 2 5 2 х 2 Е 8")
    gold = chunk("bank-4159576a", "Какая поддержка котировок")
    for doc_id, ch in (("bank-6c3a4882", ofz), ("bank-4159576a", gold)):
        q = _bank_quote(ch["text"], _stems(ch["text"]), title=DOCS[doc_id]["title"], today=AUDIT_DAY, doc_date=DOCS[doc_id]["date"])
        assert "2 2 5 2" not in q and "Какая поддержка" not in q and "Диверсификация" not in q


def test_position_issue_rules():
    ok = "Мы ожидаем, что в ближайшие месяцы показатели финансов компаний останутся на низких уровнях."
    assert position_issue(ok) == ""
    assert position_issue("В пятницу 11 сентября ЦБ ожидаемо сохранил ключевую ставку на уровне 14,0%, однако "
                          "инвесторы остались разочарованы.") == "нет позиции банка", "«ожидаемо» — не «ожидаем»"
    assert position_issue("Ранее мы писали о том, что считаем октябрьское заседание крайне важным для рынка.") == "нет позиции банка"
    assert position_issue("Мы ожидаем роста.") == "длина" and position_issue(ok[:-1] + " " + "и далее " * 60 + ".") == "длина"
    assert position_issue("Мы ожидаем 2 2 5 2 х 2 Е 8 é Е é é 22 5 2s 2 g é é é a2 a роста цен на сталь.") == "мусор распознавания"
    assert position_issue("Потребление на пике Мы полагаем, что динамика потребления сейчас находится на пике.",
                          title="Потребление на пике") == "сросшийся заголовок"
    assert position_issue("Цены на золото сохраняются Диверсификация резервов, на наш взгляд, остаётся опорой рынка.") \
        == "сросшийся заголовок"
    assert position_issue("мы ожидаем ее сохранения на текущем значении 14,0% до конца года.") == "обрывок предложения"


def test_contradicting_and_expired_expectations_are_dropped():
    hold = "Если раньше мы ориентировались на ставку 13,5%, то теперь мы ожидаем ее сохранения на текущем значении 14,0%."
    assert position_issue(hold, direction="cut") == "решение ЦБ противоречит событию"
    assert position_issue(hold, direction="hold") == ""
    july = "Мы ожидаем, что 24 июля регулятор сохранит ставку неизменной на уровне 14,25%, прогноз будет пересмотрен."
    assert position_issue(july, today=AUDIT_DAY, doc_date="2026-07-22") == "ожидание к прошедшей дате"
    assert position_issue(july, today=date(2026, 7, 23), doc_date="2026-07-22") == ""
    assert position_issue(july.replace("24 июля", "24 июля 2026 г."), today=AUDIT_DAY, doc_date="2026-07-22") \
        == "ожидание к прошедшей дате"
    cut = "Мы ожидаем снижения ключевой ставки до конца года при сохранении проинфляционных рисков в экономике."
    assert position_issue(cut, direction="cut") == "", "«сохранение рисков» — не ожидание паузы"


def test_figure_references_stripped_and_superseded_forecast_not_quoted():
    """1П26 (31.07): «…(cm Рис.7)» убирается; «ставка 13,5% на конец года» пересмотрена 11.09 — не цитируется."""
    ch = chunk("bank-84899553", "заложить в прогноз")
    q = _bank_quote(ch["text"], _stems("инфляция прогноз ключевая ставка снижение"), title=DOCS["bank-84899553"]["title"],
                    today=AUDIT_DAY, doc_date="2026-07-31")
    assert q and "Рис" not in q and "13,5%" not in q and "7,5%" in q


# ---------- пример новостей (data/news/events.sample.json) × компании разных отраслей, без модели ----------
# Полный прогон 115 компаний × поток новостей — скрипт аудита; здесь — все события примера для компаний, чьи
# отрасли поднимают разные документы банка (поиск учитывает отрасли компании)
COMPANIES = ("U_25835", "U_200890", "U_34911", "U_19717")


@lru_cache(maxsize=1)
def sample_world():
    events = load_events(ROOT / "data" / "news" / "events.sample.json")
    bank = add_posts(load_bank_index(ROOT / "data" / "bank" / "docs.json"), events)
    return events, bank, news_index(events)


def brief(ev: dict, cid: str) -> dict:
    events, bank, news = sample_world()
    p = profiles.get(cid)
    card, trace = rules_card(ev, cid, match(ev, profiles.features(p), p["bank_products"]), bank, news, AUDIT_DAY)
    assert card is not None, trace[-2]["detail"]
    return card


def bank_texts(card: dict) -> list[str]:
    return [card["blocks"]["bank"]["quote"]] + [k["t"] for k in card["key_points"] if k["p"] == "bank"]


def test_every_brief_bank_quote_is_a_real_position():
    events, bank, news = sample_world()
    quotes = []
    for cid in COMPANIES:
        for ev in events:
            ses = ToolSession(cid, ev, bank, news, AUDIT_DAY)
            src, quote, _, _ = bank_position(ses, ev, _facts(ev), AUDIT_DAY)
            if quote:
                assert verbatim(quote, src.text), quote
                quotes.append(quote)
    assert not [q for q in quotes if SCHEDULE in q]
    assert all(POSITION_RE.search(s) for q in quotes for s in SENT_RE.split(q))
    assert any("14,0%" in q for q in quotes), "по ставке остаётся действующая позиция банка (11.09)"


def test_audit_pairs_have_no_garbage_bank_text():
    events, *_ = sample_world()
    by_id = {e["id"]: e for e in events}
    assert not any("2 2 5 2 х 2 Е 8 é" in t for t in bank_texts(brief(by_id["ev-dd91cc5dc1"], "U_34911")))
    assert not any("Какая поддержка котировок" in t for t in bank_texts(brief(by_id["ev-6b18a69be3"], "U_200890")))


def test_pasted_rate_cut_has_no_hold_quote():
    ev = event_from_text("Банк России снизил ключевую ставку с 16% до 14% годовых",
                         "Банк России снизил ключевую ставку с 16% до 14% годовых.", published="2026-10-21")
    for cid in ("U_25835", "U_200890"):
        texts = [t for t in bank_texts(brief(ev, cid)) if t]
        assert not any(re.search(r"ожидаемо сохранил|сохранения паузы|ее сохранения", t) for t in texts), texts
        assert all(POSITION_RE.search(t) for t in texts)
