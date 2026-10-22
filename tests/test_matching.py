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
    assert crime.score < market.score and any(r["feature"] == "суд или уголовное дело" for r in crime.reasons)


@pytest.mark.parametrize("text,sector", [
    ("Детали взлома стали известны вчера", "24"), ("Холдинг расширяет медиабизнес", "24"),
    ("Цены на бумажную нефть выросли", "17"), ("Правительство утвердило дорожную карту по цифровому рублю", "42"),
    ("Платёжная инфраструктура банков работает штатно", "42"), ("США отправляют авианосец в Средиземное море", "50"),
])
def test_sector_false_friends(text, sector):
    assert sector not in okved.sectors_in(text)


@pytest.mark.parametrize("text,sector", [
    ("Цены на медь обновили максимум", "24"), ("Производители картона подняли цены", "17"),
    ("Авиакомпании сокращают рейсы", "51"), ("Страховщики ужесточают тарифы", "65"), ("Сеть клиник открыла центр", "86"),
])
def test_sector_dictionary_covers_more_industries(text, sector):
    assert sector in okved.sectors_in(text)


def test_labelled_relevance_set_quality():
    from scripts.eval_matching import evaluate
    r = evaluate()
    assert r["pairs"] >= 40 and r["f1"] >= 0.9, r["errors"]


@pytest.mark.parametrize("text,is_fx", [
    ("Рубль ослаб к юаню на 4,5% за неделю", True), ("Курс доллара превысил 95 рублей", True),
    ("Плата составит 1 рубль в месяц", False), ("Латвия нарастила экспорт российского зерна", False),
])
def test_fx_topic_needs_currency_phrase(text, is_fx):
    assert ("курс" in okved.topics_in(text)) is is_fx


def test_daily_official_rate_is_routine():
    m = score(AUTO, "ЦБ установил официальные курсы валют на 21 октября", "Курс юаня на 21 октября — 11,3 рубля.")
    assert not m.selected


# ---------- точность отбора на новых формулировках (аудит 21.10, B08/B09) ----------
OIL, SOFT, CAFE = "U_57706", "U_44792", "U_212593"


def stream() -> dict:
    from agent.pipeline import load_events
    return {e["id"]: e for e in load_events()}


def test_rudenko_gold_reserves_and_digital_gold_are_not_metals():
    """«Руденко» совпадал с «рудой», запас золота ЦБ и «цифровое золото» — с металлургией (опт металлами)."""
    p = profiles.get(METALS)
    ev = stream()
    for eid in ("ev-3ac1177db7", "ev-bee1cddbf6", "ev-131a0d372a", "ev-6b18a69be3"):
        m = match(ev[eid], profiles.features(p), p["bank_products"])
        assert not any("руд" in r["feature"] or "металлург" in r["feature"] for r in m.reasons), (eid, m.reasons)
    assert "07" in okved.sectors_in("Добыча железной руды выросла, рудники нарастили выпуск")


@pytest.mark.parametrize("text,sector", [
    ("Ростелеком проложит оптоволоконные линии", "46"), ("Ситуация на рынке складывается в пользу покупателей", "52"),
    ("Apple показала складной iPhone", "52"), ("Органы опеки проверят детские дома", "06"),
    ("Коммерческий директор банка рассказал о рекламе", "68"), ("Золото подорожало до рекорда", "24"),
])
def test_stoplist_and_word_boundaries(text, sector):
    assert sector not in okved.sectors_in(text)


@pytest.mark.parametrize("text,is_rate", [
    ("Минфин предложил повысить НДС до 22%", False),
    ("Минфин предложил повысить ставку НДС с 20% до 22%", False),
    ("Ставки утильсбора на легковые автомобили вырастут на 30%", False),
    ("Ставки аренды складов выросли на 25%", False),
    ("Минфин установил ставку купона по ОФЗ на уровне 15%", False),
    ("Ставки по ипотеке снизились до 21%", False),
    ("ФРС снизила ставку на 0,25 п.п.", False),
    ("ЦБ Турции снизил ключевую ставку до 40%", False),
    ("ЦБ сохранил ставку", True),
    ("Банк России снизил ставку до 13,5%", True),
    ("Ключевая ставка останется высокой до весны", True),
    ("Банк России снизил ключевую ставку с 16% до 14% годовых", True),
    ("Повышение НДС может подтолкнуть ЦБ к сохранению жёсткой ставки", True),
])
def test_key_rate_is_not_tax_rent_coupon_or_foreign_rate(text, is_rate):
    assert ("ставка" in okved.topics_in(text)) is is_rate


def test_vat_news_has_no_rate_topic_and_no_debt_reason():
    m = score(AUTO, "Минфин предложил повысить НДС до 22%", "Минфин предложил повысить ставку НДС с 20% до 22% с 2027 года.")
    assert not any(r.get("kind") == "debt" for r in m.reasons)


def test_bank_economist_post_is_not_a_digest():
    """Прогноз главного экономиста банка доводами списком («🟥 …») — не подборка: вес не режется."""
    ev = stream()["ev-928fb42abc"] | {"bank_voice": True}
    p = profiles.get(AUTO)
    m = match(ev, profiles.features(p), p["bank_products"])
    assert m.score >= 2.5 and not any(r["feature"] == "подборка новостей" for r in m.reasons)
    # тот же текст в канале не банка — обычная подборка; обзор канала банка без экономиста — тоже
    other = match(ev | {"bank_voice": False, "source": "Другой канал"}, profiles.features(p), p["bank_products"])
    assert any(r["feature"] == "подборка новостей" for r in other.reasons)
    digest = stream()["ev-d16efad254"]
    assert any(r["feature"] == "подборка новостей" for r in match(digest, profiles.features(p), p["bank_products"]).reasons)


def test_synthetic_bank_voice_event_skips_digest_factor():
    text = "\n".join(["Главный экономист Альфа-Банка считает, что ставка не изменится."] +
                     [f"🟥 Довод {k}: инфляция и спрос, ключевая ставка останется 14%." for k in range(4)])
    ev = event_from_text("Прогноз по ключевой ставке", text) | {"bank_voice": True, "source": "Канал банка"}
    p = profiles.get(AUTO)
    assert not any(r["feature"] == "подборка новостей" for r in match(ev, profiles.features(p), p["bank_products"]).reasons)


def test_sanctions_on_own_goods_weigh_more():
    """Опт топливом и «22-й пакет санкций против нефти РФ»: раньше «к сведению», теперь — значимое событие."""
    p = profiles.get(OIL)
    m = match(stream()["ev-1852318aaf"], profiles.features(p), p["bank_products"])
    assert m.score >= 6 and any(r.get("kind") == "sanctions" and r["why"] == "ограничения касаются вашего товара"
                                for r in m.reasons)


def test_tariffs_elsewhere_are_not_sanctions_for_trade_partner():
    """«Американские пошлины… падение продаж в Китае» (Volkswagen) — не ограничения в торговле России с Китаем."""
    m = score("U_150736", "Volkswagen хочет заморозить зарплату", "Меры — на фоне американских пошлин и падения продаж в Китае.")
    assert not any(r.get("kind") == "sanctions" for r in m.reasons)
    m = score("U_150736", "Китай ввёл пошлины на российский картон", "Пекин ввёл пошлины 20% на ввоз картона из России.")
    assert any(r.get("kind") == "sanctions" for r in m.reasons) and m.selected


def test_exchange_price_is_a_world_benchmark_not_foreign_market():
    m = score(METALS, "Медь на LME подорожала на 8%", "Цены на медь на Лондонской бирже металлов (LME) выросли на 8% за неделю.")
    assert m.selected and not any(r["feature"] == "зарубежный рынок" for r in m.reasons)


def test_cb_decision_reaches_every_company_with_honest_reason():
    m = score(SOFT, "ЦБ сохранил ставку на уровне 14%", "Банк России сохранил ключевую ставку на уровне 14% годовых.")
    assert m.selected and any(r["why"] == "кредитов в Альфа-Банке у вас нет — влияние через покупателей и курс"
                              for r in m.reasons)
    # прогноз и чужой центральный банк — не решение Банка России
    assert not score(SOFT, "Ключевая ставка останется высокой", "Аналитики ждут, что ЦБ сохранит ключевую ставку 14%.").selected
    assert not score(SOFT, "ФРС снизила ставку", "ФРС снизила ставку на 0,25 п.п.").selected


def test_rate_demand_channel_uses_own_and_goods_sectors_only():
    """Опт топливом с недвижимостью в группе: «ваши покупатели берут ипотеку» — неправда, а автодилеры — правда."""
    news = ("ЦБ снизил ключевую ставку", "Банк России снизил ключевую ставку с 16% до 14% годовых.")
    assert not any(r["feature"] == "спрос клиентов в кредит" for r in score(OIL, *news).reasons)
    assert any(r["feature"] == "спрос клиентов в кредит" for r in score(AUTO, *news).reasons)


def test_wholesale_goods_by_product_code():
    assert okved.related_sectors("46.46") == ["21"] and okved.related_sectors("46.41") == ["13"]
    assert okved.related_sectors("46.42") == ["14"]
    assert not any("47" in okved.related_sectors(c) for c in ("46.41", "46.42", "46.43", "46.46", "46.49"))


def test_flow_share_needs_ten_counterparties():
    """Ресторан: 90% поступлений «от оптовых компаний» — это 3 плательщика в выборке, а не отрасль покупателей."""
    p = profiles.get(CAFE)
    assert p["customers"]["n"] < 10
    m = score(CAFE, "Оптовые компании подняли цены на 10%", "Оптовые компании и дистрибуторы подняли цены на 10%.")
    assert not any(r.get("kind") == "customers" for r in m.reasons)
    m = score(DEVEL, "Производители стройматериалов подняли цены на цемент", "Цемент подорожал на 12%.")
    assert any(r.get("kind") == "customers" for r in m.reasons)  # у девелопера 30 плательщиков в выборке


def test_sector_mentioned_in_passing_weighs_less():
    """«Маск ограничил доступ к Starshield… из-за ударов по НПЗ» — не новость нефтепереработки."""
    ev = stream()["ev-a7e6afcb0d"]
    for cid in (OIL, "U_19717"):
        p = profiles.get(cid)
        assert not match(ev, profiles.features(p), p["bank_products"]).selected
    m = score(OIL, "Атаки дронов на НПЗ остановили 17% мощностей нефтепереработки",
              "Из-за атак беспилотников выпуск бензина сократился.")
    assert m.selected


def test_thin_feeds_get_macro_news():
    """Меньше трёх отобранных событий — «главное в экономике»: ставка, курс, инфляция, без повторов ленты."""
    from datetime import date

    from agent.matching import macro_events, rank

    events = list(stream().values())
    empty_in_audit = ("U_150736", "U_111822", "U_94979", "U_106783", "U_170906", "U_175851", "U_183405", "U_202619",
                      "U_225628", "U_276153", "U_46183", "U_49201", "U_86035")
    shown = {}  # компания → сколько событий она видит: отобранные + главное в экономике
    for cid in profiles.all_ids():
        p = profiles.get(cid)
        feed = rank(events, profiles.features(p), p["bank_products"])
        shown[cid] = len(feed)
        if len(feed) < 3:
            items = macro_events(events, date(2026, 10, 21), exclude={e["id"] for e, _ in feed})
            assert 1 <= len(items) <= 3, cid
            assert not {e["id"] for e, _, _ in items} & {e["id"] for e, _ in feed}
            assert all(k in ("ставка", "курс", "инфляция") and why for _, k, why in items)
            shown[cid] += len(items)
    assert all(shown[cid] >= 1 for cid in empty_in_audit)  # ни одной пустой ленты из аудита
    # дайджесты, ставка ФРС и новости старше трёх дней не берутся
    ids = {e["id"] for e, _, _ in macro_events(events, date(2026, 10, 21), limit=10)}
    assert "ev-d16efad254" not in ids and "ev-5cd64b54e9" not in ids
    assert not macro_events(events, date(2026, 11, 30))


def test_macro_route(tmp_path):
    import importlib
    import os

    from fastapi.testclient import TestClient
    os.environ.update(APP_CACHE_DIR=str(tmp_path), APP_TODAY="2026-10-21")
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as m
    m = importlib.reload(m)
    c = TestClient(m.app)
    thin = c.get("/api/companies/U_150736/macro").json()
    assert thin["needed"] and thin["selected"] == 0 and thin["items"]
    it = thin["items"][0]
    assert {"id", "title", "published", "source", "url", "topics", "macro", "why", "analyzed"} <= set(it)
    full = c.get("/api/companies/U_57706/macro").json()
    assert full["selected"] >= 3 and not full["needed"] and full["items"] == []
    assert c.get("/api/companies/U_0/macro").status_code == 404
