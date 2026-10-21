"""Обучение на опыте: шаблоны в потоке, веса тем по оценкам, память разборов, уроки, изоляция компаний.
Примеры заголовков — реальные пары из замера docs/experiments/learning.md."""
from datetime import date

import pytest

from agent.learning import FEEDBACK_MAX, Learner, lesson_categories
from agent.stories import conflict, same_story, template_days
from backend.store import Store


def card(cid, eid, title, topics=("ставка",), published="2026-10-19", check="", source="Интерфакс"):
    return {"id": f"{eid}-{cid}", "title": title, "topics": list(topics), "published": published,
            "sources": [{"name": source, "kind": "news", "date": published}],
            "takeaway": {"cfo": "вывод " + title}, "trace": [{"kind": "check", "detail": check}] if check else []}


@pytest.fixture
def learner(tmp_path):
    return Learner(Store(tmp_path / "app.db"))


# ---------- тот же сюжет: ошибки прежнего правила, найденные замером ----------
@pytest.mark.parametrize(("a", "b"), [
    ("ЦБ установил официальные курсы валют на 21 октября", "ЦБ установил официальные курсы валют на 20 октября"),
    ("На 10:10 мск у индекса Мосбиржи +0,34%", "На 10:10 мск у индекса Мосбиржи −0,07%"),
    ("Трамп поздравил Путина с днем рождения — Кремль.", "Трамп не поздравил Путина с Днём Рождения — Песков."),
    ("Brent подешевела до $102,93 за баррель", "Brent подорожала до $105,58 за баррель"),
    ("Аэропорт Внуково обслуживает рейсы по согласованию", "Аэропорт Шереметьево обслуживает рейсы по согласованию"),
    ("Рынок акций РФ открылся в основную сессию подъемом индексов МосБиржи и РТС на 3,2%",
     "Рынок акций РФ открылся в основную сессию снижением индексов МосБиржи и РТС на 0,3%"),
])
def test_different_events_are_not_same_story(a, b):
    assert not same_story(a, b)


@pytest.mark.parametrize(("a", "b"), [
    ("Бензин с начала года подорожал на 21,37% – данные Росстата", "Бензин с начала года подорожал на 21,37% — Росстат."),
    ("Евросоюз ввел санкции против российского девелопера \"Самолет\" и его гендиректора Анны Акиньшиной",
     "ЕС ввёл санкции против девелопера «Самолёт» и его гендиректора Анны Акиньшиной."),
    ("ЦБ РФ сохранит ставку на уровне 14% на заседании 23 октября — аналитики, опрос ТАСС",
     "ЦБ РФ сохранит ставку на уровне 14% на заседании 23 октября — аналитики, опрос РБК"),
])
def test_same_event_from_different_outlets_is_same_story(a, b):
    assert same_story(a, b)


def test_same_source_next_day_is_new_publication():
    t = "Минфин разместил ОФЗ на аукционе"
    assert conflict(t, t, "Минфин России", "Минфин России", "2026-10-14", "2026-10-21") == "тот же источник, другой день"
    assert conflict(t, t, "Минфин России", "Интерфакс", "2026-10-14", "2026-10-14") is None


# ---------- шаблоны в потоке ----------
def test_template_needs_three_days_and_varying_details():
    h = [("2026-10-19", "ЦБ установил официальные курсы валют на 20 октября"),
         ("2026-10-20", "ЦБ установил официальные курсы валют на 21 октября")]
    assert template_days("ЦБ установил официальные курсы валют на 22 октября", h) == 0
    h.append(("2026-10-21", "ЦБ установил официальные курсы валют на 22 октября"))
    assert template_days("ЦБ установил официальные курсы валют на 22 октября", h) == 3


def test_repeated_coverage_of_important_event_is_not_template():
    """Опросы аналитиков перед заседанием ЦБ выходят несколько дней с теми же цифрами — это не рутина."""
    t = "ЦБ РФ сохранит ставку на уровне 14% на заседании 23 октября — аналитики, опрос {}"
    h = [("2026-10-16", t.format("Ведомости")), ("2026-10-19", t.format("RTRS")), ("2026-10-21", t.format("ТАСС"))]
    assert template_days(t.format("ТАСС"), h) == 0


def test_learner_marks_template_and_switch_turns_it_off(learner):
    evs = [{"source": "Интерфакс", "published": f"2026-10-{d}", "title": f"Что произошло за день: {w}, {d} октября"}
           for d, w in (("19", "понедельник"), ("20", "вторник"), ("21", "среда"))]
    learner.observe(evs)
    assert learner.routine_days(evs[-1]) == 3 and learner.routine()[0]["source"] == "Интерфакс"
    learner.enabled = False
    assert learner.routine_days(evs[-1]) == 0


# ---------- оценки клиентов ----------
def vote(store, cid, n, topics, v, start=0):
    for i in range(start, start + n):
        store.put_card(cid, card(cid, f"e{i}", f"Новость {i}", topics), "agent")
        store.put_feedback(cid, f"e{i}-{cid}", "cfo", v)


def test_feedback_lift_moves_topics_and_ignores_omnipresent_topic(learner):
    vote(learner.store, "A", 6, ("ставка", "отрасль"), "up")
    vote(learner.store, "A", 6, ("спрос", "отрасль"), "down", start=6)
    w = learner.topic_weights()
    own = w["companies"]["A"]["cfo"]
    assert 0 < own["ставка"] < FEEDBACK_MAX and own["спрос"] < 0
    assert own["отрасль"] == 0                 # тема в каждой оценённой новости не набирает веса
    assert "ceo" not in w["roles"]


def test_few_votes_do_not_move_weights(learner):
    """Защита от шума: по 1–2 оценкам вес не меняется (в симуляции так лента не ухудшается)."""
    vote(learner.store, "A", 2, ("ставка",), "up")
    vote(learner.store, "A", 2, ("спрос",), "down", start=2)
    w = learner.topic_weights()
    assert all(d == 0 for d in w["companies"]["A"]["cfo"].values())


def test_company_without_votes_gets_role_experience_only(learner):
    vote(learner.store, "A", 6, ("ставка",), "up")
    vote(learner.store, "A", 6, ("спрос",), "down", start=6)
    w = learner.topic_weights()
    assert "B" not in w["companies"] and w["roles"]["cfo"]["ставка"] > 0


# ---------- память разборов ----------
def test_memory_reuses_same_story_for_same_company_only(learner):
    s = learner.store
    s.put_card("A", card("A", "e1", "ЦБ сохранил ключевую ставку на уровне 14% годовых"), "agent")
    same = {"title": "Банк России сохранил ключевую ставку на уровне 14% годовых", "source": "РБК", "published": "2026-10-20"}
    assert learner.remembered("A", same, date(2026, 10, 21))["id"] == "e1-A"
    assert learner.remembered("B", same, date(2026, 10, 21)) is None          # чужой разбор не отдаём
    assert learner.remembered("A", same, date(2026, 11, 30)) is None          # устарел
    s.moderate("A", "e1-A", True)
    assert learner.remembered("A", same, date(2026, 10, 21)) is None          # скрытый банком разбор не раздаём


def test_memory_does_not_serve_yesterdays_template(learner):
    """Главная ошибка прежнего правила: вчерашний «курс на 20 октября» выдавался на новость про 21-е."""
    learner.store.put_card("A", card("A", "e1", "ЦБ установил официальные курсы валют на 20 октября"), "agent")
    ev = {"title": "ЦБ установил официальные курсы валют на 21 октября", "source": "Интерфакс", "published": "2026-10-20"}
    assert learner.remembered("A", ev, date(2026, 10, 21)) is None


def test_precedents_are_isolated_per_company(learner):
    s = learner.store
    s.put_card("A", card("A", "e1", "ЦБ сохранил ставку"), "agent")
    s.put_card("B", card("B", "e1", "ЦБ сохранил ставку"), "agent")
    s.put_card("A", card("A", "e3", "ЦБ сохранил ставку"), "rules")  # краткий разбор — не опыт
    p = learner.precedents("A", {"id": "e9", "title": "ЦБ снизил ставку до 13%", "topics": ["ставка"]})
    assert len(p) == 1 and p[0]["takeaway"] == "вывод ЦБ сохранил ставку"


# ---------- уроки проверки ----------
def test_lesson_categories_on_real_findings():
    assert lesson_categories("Все проверки пройдены Исправлено до проверки: ссылка S4→S2.") == {"cite"}
    # «ДКП» внутри процитированного черновика — не замечание о жаргоне
    assert lesson_categories("Исправлено до проверки: убрано: «Проинфляционные риски и жёсткая ДКП (ЦБ…».") == {"num"}
    assert lesson_categories("ошибок: 1. key_points: [] is too short") == {"fmt"}
    assert lesson_categories("правка не прошла проверку кодом — оставлен прежний черновик") == set()


def test_lessons_activate_after_repeats(learner):
    s = learner.store
    s.log_run("A", "e1", False, ["цифры не найдены в источнике: 12%"], 0, 0, 0, 1)
    assert not learner.lessons()[0]["active"] and learner.active_lessons() == []
    s.put_card("A", card("A", "e2", "x", check="Исправлено до проверки: убрано: «15%…»"), "agent")
    assert learner.active_lessons() == ["Цифры и даты — только из источников: если числа нет в тексте источника, не пиши его."]


def test_summary_has_trend_and_counters(learner):
    learner.store.log_run("A", "e1", True, [], 0.4, 30000, 5, 0)
    s = learner.summary()
    assert s["trend"][0]["runs"] == 1 and s["memory"] == 0 and s["enabled"] is True and s["useful_share"] is None


# ---------- склейка дублей при сборе ----------
def test_collector_keeps_template_days_apart_and_merges_true_duplicates():
    from datetime import datetime, timezone

    from agent.ingest.collect import to_events
    from agent.ingest.post import Post

    def post(n, text, day, channel):
        return (Post(channel, channel, n, text, datetime(2026, 10, day, 12, tzinfo=timezone.utc)), {"name": channel})
    evs = to_events([
        post(1, "ЦБ установил официальные курсы валют на 20 октября: доллар 82,9 рубля", 19, "interfax"),
        post(2, "ЦБ установил официальные курсы валют на 21 октября: доллар 82,4 рубля", 20, "interfax"),
        post(3, "Бензин с начала года подорожал на 21,37% – данные Росстата", 20, "interfax"),
        post(4, "Бензин с начала года подорожал на 21,37% — Росстат.", 20, "markettwits"),
    ])
    titles = [e["title"] for e in evs]
    assert sum("курсы валют" in t for t in titles) == 2                  # выпуски шаблона — разные события
    assert sum("Бензин" in t for t in titles) == 1                        # одна новость из двух каналов — одно событие
    assert next(e for e in evs if "Бензин" in e["title"])["also"][0]["name"] == "markettwits"
