"""Отбор событий под компанию: ложные срабатывания, найденные на реальной ленте 21.10, не возвращаются."""
import pytest

from agent import okved, profiles
from agent.matching import match
from agent.pipeline import event_from_text

AUTO, DEVEL, METALS, CLOTHES = "U_25835", "U_177581", "U_200890", "U_1595"


def score(cid, title, text=""):
    p = profiles.get(cid)
    return match(event_from_text(title, text), profiles.features(p), p["bank_products"])


@pytest.mark.parametrize("text,is_rate", [
    ("Банк России снизил ключевую ставку до 14%", True),
    ("Совет директоров рассмотрит вопрос о ключевой ставке", True),
    ("Банк России второй день сохраняет курс доллара в размере 82 рублей", False),
    ("Ключевым фактором роста стала осознанная ставка на качество бизнеса", False),
])
def test_rate_topic_needs_rate_phrase(text, is_rate):
    assert ("ставка" in okved.topics_in(text)) is is_rate


def test_retail_investors_and_import_substitution_are_not_sectors():
    assert "47" not in okved.sectors_in("В Швейцарии одобрен ETF для розничных инвесторов")
    assert "курс" not in okved.topics_in("Обсудили импортозамещение нефтегазового оборудования")


def test_rate_news_hits_company_with_debt():
    m = score(AUTO, "ЦБ снизил ключевую ставку", "Банк России снизил ключевую ставку с 16% до 14% годовых.")
    assert m.selected and "кредит" in m.matched


def test_foreign_market_is_down_weighted():
    dom = score(DEVEL, "Продажи новостроек в Москве упали", "Девелоперы сообщили о падении продаж новостроек и ипотеки в Москве.")
    abroad = score(DEVEL, "Продажи новостроек в Дубае упали", "Девелоперы сообщили о падении продаж новостроек и ипотеки в Дубае.")
    assert abroad.score < dom.score * 0.5 and any(r["feature"] == "зарубежный рынок" for r in abroad.reasons)


def test_foreign_market_kept_when_company_trades_there():
    m = score(AUTO, "Китай снизил экспортные пошлины на автозапчасти", "Китай снижает пошлины на экспорт автозапчастей и курс юаня.")
    assert not any(r["feature"] == "зарубежный рынок" for r in m.reasons)


def test_digest_post_is_down_weighted():
    items = "\n".join(f"🔴 Новость {k}: ключевая ставка, курс юаня и продажи автомобилей." for k in range(4))
    one = score(AUTO, "Ключевая ставка и курс юаня", "ЦБ сохранил ключевую ставку, курс юаня вырос.")
    digest = score(AUTO, "Важное к началу сессии", items)
    assert any(r["feature"] == "подборка новостей" for r in digest.reasons) and digest.score < one.score + 3


def test_generic_trade_class_weighs_less_than_product_industry():
    feats = profiles.features(profiles.get(METALS))  # опт металлами 46.72
    assert feats["own_sectors"]["46"] < feats["own_sectors"]["24"]
    clothes = profiles.features(profiles.get(CLOTHES))  # розница одеждой 47.71
    assert clothes["own_sectors"]["47"] < clothes["own_sectors"]["14"]


def test_native_ad_and_crime_news_filtered():
    from agent.ingest.collect import AD_RE
    assert AD_RE.search("Продают блоками и этажами. Стоимость лота — от 310 тыс. руб./кв.м.")
    crime = score("U_3263", "Бывший управленец РЖД получил 11 лет колонии за взятку", "Суд вынес приговор по уголовному делу.")
    market = score("U_3263", "РЖД снизила погрузку на 8%", "Погрузка на сети РЖД в сентябре снизилась на 8%.")
    assert crime.score < market.score and any(r["feature"] == "криминальная хроника" for r in crime.reasons)
