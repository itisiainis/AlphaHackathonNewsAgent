"""Обучение на опыте: рутина в потоке, веса тем по оценкам, память разборов, уроки, изоляция компаний."""
from datetime import date

import pytest

from agent.learning import FEEDBACK_MAX, Learner, signature
from backend.store import Store


def card(cid, eid, title, topics=("ставка",), published="2026-10-19", check=""):
    return {"id": f"{eid}-{cid}", "title": title, "topics": list(topics), "published": published,
            "takeaway": {"cfo": "вывод " + title}, "trace": [{"kind": "check", "detail": check}] if check else []}


@pytest.fixture
def learner(tmp_path):
    return Learner(Store(tmp_path / "app.db"))


def test_signature_ignores_dates_and_numbers():
    assert signature("ЦБ установил официальные курсы валют на 21 октября") == signature("ЦБ установил официальные курсы валют на 22 октября")
    assert signature("ЦБ сохранил ставку 14%") != signature("Минфин разместил ОФЗ")


def test_routine_needs_several_days(learner):
    t = "ЦБ установил официальные курсы валют на {} октября"
    learner.observe([{"title": t.format(20), "published": "2026-10-20T10:00"}, {"title": t.format(21), "published": "2026-10-21T10:00"}])
    assert not learner.is_routine(t.format(23))
    learner.observe([{"title": t.format(22), "published": "2026-10-22T10:00"}])
    assert learner.is_routine(t.format(23))
    learner.enabled = False
    assert not learner.is_routine(t.format(23))


def test_feedback_moves_topic_weights_smoothly(learner):
    s = learner.store
    s.put_card("A", card("A", "e1", "Ставка", ("ставка",)), "agent")
    s.put_card("A", card("A", "e2", "Спрос", ("спрос",)), "agent")
    s.put_feedback("A", "e1-A", "cfo", "up")
    s.put_feedback("A", "e2-A", "cfo", "down")
    w = learner.topic_weights()
    assert 0 < w["roles"]["cfo"]["ставка"] < FEEDBACK_MAX and w["roles"]["cfo"]["спрос"] < 0
    assert w["companies"]["A"]["cfo"]["ставка"] == w["roles"]["cfo"]["ставка"]
    assert "ceo" not in w["roles"]


def test_memory_reuses_same_story_for_same_company_only(learner):
    s = learner.store
    s.put_card("A", card("A", "e1", "ЦБ сохранил ключевую ставку на уровне 14% годовых"), "agent")
    same = "Банк России сохранил ключевую ставку на уровне 14% годовых"
    assert learner.remembered("A", same, date(2026, 10, 21))["id"] == "e1-A"
    assert learner.remembered("B", same, date(2026, 10, 21)) is None          # чужой разбор не отдаём
    assert learner.remembered("A", same, date(2026, 11, 30)) is None          # устарел
    assert learner.remembered("A", "Минфин разместил ОФЗ на 50 млрд", date(2026, 10, 21)) is None
    s.moderate("A", "e1-A", True)
    assert learner.remembered("A", same, date(2026, 10, 21)) is None          # скрытый банком разбор не раздаём


def test_precedents_are_isolated_per_company(learner):
    s = learner.store
    s.put_card("A", card("A", "e1", "ЦБ сохранил ставку"), "agent")
    s.put_card("B", card("B", "e1", "ЦБ сохранил ставку"), "agent")
    s.put_card("A", card("A", "e3", "ЦБ сохранил ставку"), "rules")  # краткий разбор — не опыт
    p = learner.precedents("A", {"id": "e9", "title": "ЦБ снизил ставку", "topics": ["ставка"]})
    assert len(p) == 1 and p[0]["takeaway"] == "вывод ЦБ сохранил ставку"


def test_lessons_activate_after_repeats(learner):
    s = learner.store
    s.log_run("A", "e1", False, ["цифры не найдены в источнике: 12%"], 0, 0, 0, 1)
    assert not learner.lessons()[0]["active"] and learner.active_lessons() == []
    s.put_card("A", card("A", "e2", "x", check="убрано: «15%» — нет в источнике"), "agent")
    assert learner.active_lessons() == ["Цифры — только из источников: если числа нет в тексте источника, не пиши его."]


def test_summary_has_trend_and_counters(learner):
    learner.store.log_run("A", "e1", True, [], 0.4, 30000, 5, 0)
    s = learner.summary()
    assert s["trend"][0]["runs"] == 1 and s["memory"] == 0 and s["enabled"] is True
