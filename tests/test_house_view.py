"""Взгляд банка: действующий прогноз побеждает пересмотренный — в данных, поиске, задании агенту, трассе и проверке."""
import json
import re
import socket
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agent import house_view, kb, okved, profiles
from agent.contract import validate_card
from agent.ingest.collect import to_events
from agent.ingest.post import Post
from agent.llm import Completion
from agent.loop import Agent
from agent.matching import match
from agent.prompts import DRAFT_SYSTEM, house_view_block
from agent.tools import ToolSession
from scripts import build_house_view as build

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 10, 21)
DOCS = {d["id"]: d for d in json.loads((ROOT / "data" / "bank" / "docs.json").read_text(encoding="utf-8"))}
ENTRIES = house_view.load()
EVENTS_PATH = next(p for p in (ROOT / "data" / "news" / "events.json", ROOT / "data" / "news" / "events.sample.json")
                   if p.exists())
EVENTS = {e["id"]: e for e in json.loads(EVENTS_PATH.read_text(encoding="utf-8"))}
SAMPLE = {e["id"]: e for e in json.loads((ROOT / "data" / "news" / "events.sample.json").read_text(encoding="utf-8"))}
CARD = ROOT / "data" / "cards" / "U_25835" / "ev-e31c37f5eb-U_25835.json"
QUERY = "ключевая ставка до конца года прогноз"
DIVERGENCE = "источники расходятся: ставка на конец 2026 — 13,5% (31.07) и 14,0% (11.09); берём более свежий"


def bank_index():
    return kb.load_bank_index(ROOT / "data" / "bank" / "docs.json")


# ---------- данные: сборка файла ----------

def test_build_is_deterministic_and_offline(monkeypatch):
    def no_network(*a, **kw):
        raise AssertionError("сборка взгляда банка не должна ходить в сеть")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    first, second = build.build(), build.build()
    assert json.dumps(first, ensure_ascii=False) == json.dumps(second, ensure_ascii=False)
    # файл в репозитории собран из текущих документов (посты зависят от свежего сбора — сверяются цитатой ниже)
    assert [e for e in first if e["page"]] == [e for e in ENTRIES if e["page"]]
    src = (ROOT / "scripts" / "build_house_view.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:from|import)\s+(?:agent\.llm|httpx|requests|urllib)", src, re.M)


def test_every_quote_is_verbatim_and_states_its_value():
    for e in ENTRIES:
        if e["page"]:
            where = DOCS[e["doc_id"]]["pages"][e["page"] - 1]
        else:
            post = EVENTS.get(e["doc_id"]) or SAMPLE[e["doc_id"]]
            where = post["text"]
        assert house_view.norm(e["quote"]) in house_view.norm(where), (e["doc_id"], e["page"], e["quote"][:80])
        assert build.states_value(e), (e["indicator"], e["value"], e["quote"][:80])
        assert e["date"] == (DOCS[e["doc_id"]]["date"] if e["page"] else EVENTS.get(e["doc_id"], SAMPLE.get(e["doc_id"]))["published"])


def test_table_rows_give_the_value_of_their_year_column():
    """Строка таблицы: значение — из столбца своего года (шапка 2019 … 2026П 2027П), а не любое число строки."""
    for e in (e for e in ENTRIES if e["page"] and any(re.match(lbl, e["quote"]) for lbl in build.TABLE_ROWS.values())):
        page = DOCS[e["doc_id"]]["pages"][e["page"] - 1]
        years = re.findall(r"20\d\d", build.TABLE_HEAD.search(page).group(0))
        label = build.TABLE_ROWS[e["indicator"]]
        cells = re.sub(label, "", e["quote"]).split()
        assert len(cells) == len(years)
        assert float(cells[years.index(e["horizon"][4:])].rstrip("%").replace(",", ".")) == e["value"]


def test_garbled_ocr_rows_are_not_parsed():
    """Строка юаня в обзоре 1П26 испорчена распознаванием («10,1 fa Rea 11,7») — из неё прогноза нет."""
    assert not [e for e in ENTRIES if e["doc_id"] == "bank-84899553" and e["indicator"] == "cny_rub"]


def test_quote_is_the_part_of_the_sentence_with_the_forecast():
    gdp = next(e for e in ENTRIES if (e["indicator"], e["horizon"], e["doc_id"]) == ("gdp", "end-2026", "bank-807ee992"))
    assert gdp["quote"].startswith("мы понизили наш прогноз") and "0,1%" in gdp["quote"]


def test_required_key_rate_entries_and_supersession():
    old = [e for e in ENTRIES if (e["indicator"], e["horizon"], e["value"]) == ("key_rate", "end-2026", 13.5)]
    assert {e["doc_id"] for e in old} >= {"bank-84899553"} and all(e["superseded_by"] == "bank-92a96805" for e in old)
    new = [e for e in ENTRIES if (e["indicator"], e["horizon"], e["value"]) == ("key_rate", "end-2026", 14.0)]
    assert new and all(e["superseded_by"] is None and e["doc_id"] == "bank-92a96805" for e in new)
    # 87 и 87,2 руб./$ — одно значение с округлением, не пересмотр; 95 → 93 руб./$ на конец 2027 — пересмотр
    assert all(e["superseded_by"] is None for e in ENTRIES if (e["indicator"], e["horizon"]) == ("usd_rub", "end-2026"))
    assert [e["superseded_by"] for e in ENTRIES if (e["indicator"], e["horizon"], e["value"]) == ("usd_rub", "end-2027", 95.0)] \
        == ["bank-92a96805"]


def test_bank_voice_post_entry():
    e = next(e for e in ENTRIES if (e["indicator"], e["horizon"]) == ("key_rate", "2027-04"))
    assert (e["doc_id"], e["value"], e["date"]) == ("ev-928fb42abc", 14.0, "2026-10-20")
    assert e["doc_title"] == "Альфа-Инвестиции, пост от 20.10.2026"
    assert e["quote"] in (EVENTS.get("ev-928fb42abc") or SAMPLE["ev-928fb42abc"])["text"]


# ---------- правило: какой прогноз действует ----------

def test_current_superseded_and_trail():
    cur = house_view.current("key_rate", "end-2026", TODAY)
    assert cur["value"] == 14.0 and cur["doc_id"] == "bank-92a96805"
    assert 13.5 in {e["value"] for e in house_view.superseded("key_rate", "end-2026", TODAY)}
    assert house_view.trail("key_rate", "end-2026", TODAY) == "31 июля: 13,5% → 11 сентября: 14,0% (действует)"
    assert house_view.trail("key_rate", "end-2027", TODAY) == ""  # прогноз не менялся


def test_current_respects_the_date_and_staleness():
    assert house_view.current("key_rate", "end-2026", date(2026, 8, 1))["value"] == 13.5  # 11 сентября ещё не наступило
    assert house_view.current("key_rate", "end-2026", date(2027, 9, 12)) is None  # всё старше 12 месяцев
    assert house_view.STALE_DAYS == 365


def test_as_context_lists_current_view_and_revisions():
    ctx = house_view.as_context(TODAY)
    assert ctx.startswith("Текущий взгляд Альфа-Банка")
    line = next(x for x in ctx.splitlines() if x.startswith("- Ключевая ставка на конец 2026"))
    assert "14,0%" in line and "11.09.2026" in line and "прежний прогноз 13,5% (31.07, 05.08) пересмотрен" in line


@pytest.mark.parametrize("text, flagged", [
    ("Альфа-Банк ждёт ключевую ставку 13,5% на конец 2026 г.", True),
    ("На конец года мы ожидаем ключевую ставку на уровне 13,5%", True),
    ("Если раньше мы ориентировались на ставку 13,5% на конец 2026 г., то теперь ожидаем 14,0% до конца года.", False),
    ("Ставки по депозитам — около 13,5% годовых.", False),
    ("Ипотечная ставка 13,5% на конец 2026 г. — уже рыночная.", False),
    ("Мы ожидаем ставку 13,5% на конец 2026 г., при этом ставки по депозитам снизятся.", True),
    ("Ставки по кредитам вслед за ключевой ставкой — 13,5% на конец 2026 г.", True),
    ("В 2025 г. ставка снижалась до 13,5%.", False),
    ("Регулятор повысит прогноз диапазона ключевой ставки на 2027 год до 11,5–13,5%.", False),
    ("Банк ждёт ставку 14,0% до конца года.", False),
])
def test_revisions_in(text, flagged):
    assert bool(house_view.revisions_in(text, TODAY)) is flagged


# ---------- поиск по аналитике ----------

def test_kb_ranks_current_forecast_above_revised_one():
    hits = bank_index().search(QUERY, top_k=4, today=TODAY)
    ids = [h.doc_id for h in hits]
    assert ids.index("bank-92a96805") < ids.index("bank-84899553")
    new = hits[ids.index("bank-92a96805")]
    assert "14,0% до конца года" in new.text and not new.revised
    old = hits[ids.index("bank-84899553")]
    assert old.revised == "пересмотрено 11.09: банк ждёт 14,0% (ставка на конец 2026)"


def test_kb_adds_current_forecast_when_only_old_one_matches():
    """Запрос нашёл только старый прогноз — действующий добавляется в выдачу перед ним; длина выдачи прежняя."""
    q = "Снижение нормы сбережений для поддержания потребления"
    plain = kb.BM25Index(bank_index().items).search(q, top_k=4, today=TODAY)
    assert "bank-92a96805" not in {h.doc_id for h in plain}
    hits = bank_index().search(q, top_k=4, today=TODAY)
    ids = [h.doc_id for h in hits]
    assert len(hits) == 4 and "bank-92a96805" in ids
    assert ids.index("bank-92a96805") < min(i for i, h in enumerate(hits) if h.revised)


def test_kb_news_index_is_untouched():
    """Новости и посты не банка не помечаются: пересмотр — свойство прогнозов банка."""
    idx = kb.BM25Index([{"doc_id": "n1", "title": "Аналитики ждут ставку 13,5% на конец 2026 г.", "date": "2026-10-01",
                         "page": None, "text": "Аналитики ждут ставку 13,5% на конец 2026 г.", "kind": "news"}],
                       house_view.load())
    assert not idx.search("ставка 13,5%", today=TODAY)[0].revised


# ---------- голос банка: посты экономистов ----------

def test_only_alfa_investments_is_bank_voice():
    src = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    flagged = [s.get("channel") or s["url"] for kind in ("rss", "telegram") for s in src[kind] if s.get("bank_voice")]
    assert flagged == ["alfa_investments"]
    assert kb.bank_voice_sources() == {"Альфа-Инвестиции"}


def test_collect_marks_bank_channel_posts():
    when = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)
    text = "Главный экономист Альфа-Банка: ЦБ сохранит ключевую ставку 14% до апреля 2027 года\nПодробности в обзоре."
    other = "Аналитики рынка: ЦБ на заседании в пятницу снизит ключевую ставку до 13,5% годовых\nКомментарий."
    evs = to_events([(Post("alfa_investments", "Альфа-Инвестиции", 1, text, when),
                      {"name": "Альфа-Инвестиции", "kind": "news", "bank_voice": True}),
                     (Post("markettwits", "MarketTwits", 2, other, when), {"name": "MarketTwits", "kind": "news"})])
    by_src = {e["source"]: e for e in evs}
    assert by_src["Альфа-Инвестиции"].get("bank_voice") is True
    assert "bank_voice" not in by_src["MarketTwits"]


def test_economist_post_becomes_bank_fragment():
    post = SAMPLE["ev-928fb42abc"]
    frag = kb.post_fragment(post)
    assert frag["title"] == "Альфа-Инвестиции, пост от 20.10.2026" and frag["date"] == "2026-10-20"
    assert frag["kind"] == "bank" and frag["url"] == "https://t.me/alfa_investments/8481"
    ad = SAMPLE["ev-ea193d7d2c"]  # пост того же канала без экономиста банка — не позиция банка
    assert ad["source"] == "Альфа-Инвестиции" and kb.post_fragment(ad) is None
    stranger = post | {"id": "ev-x", "source": "MarketTwits"}  # чужой канал пересказал экономиста — тоже нет
    assert kb.post_fragment(stranger) is None
    idx = kb.add_posts(bank_index(), list(SAMPLE.values()))
    again = kb.add_posts(idx, list(SAMPLE.values()))
    assert len(idx.items) == len(bank_index().items) + 1 == len(again.items)
    one = kb.add_fragment(bank_index(), frag)
    assert one.items[-1]["doc_id"] == "ev-928fb42abc" and one.house


def test_pipeline_puts_economist_posts_into_bank_index():
    from agent.pipeline import Pipeline

    pipe = Pipeline(llm=None, today=TODAY)
    assert "ev-928fb42abc" in {i["doc_id"] for i in pipe.bank.items}
    assert len(pipe.bank_docs.items) == len(pipe.bank.items) - sum(i["doc_id"].startswith("ev-") for i in pipe.bank.items)


# ---------- инструмент поиска и задание агенту ----------

def session(event_id="ev-e31c37f5eb"):
    meta = kb.load_bank_meta(ROOT / "data" / "bank" / "docs.json")
    return ToolSession("U_25835", SAMPLE[event_id], bank_index(), kb.BM25Index([]), TODAY, meta)


def test_tool_marks_revised_fragment_and_decides_divergence():
    ses = session()
    res, entry = ses.call("search_bank_analytics", {"query": QUERY})
    assert entry["ok"] and res["divergence"] == DIVERGENCE
    old = next(f for f in res["fragments"] if f["doc"].startswith("Макростатистика 1П26"))
    assert old["text"].startswith("[ПЕРЕСМОТРЕНО 11.09 → 14,0% (ставка на конец 2026)] ")
    src = next(s for s in ses.sources if s.label == old["label"])
    assert src.revised and not src.text.startswith("[")  # текст источника — дословный: по нему сверяются цифры
    assert ToolSession.note(res) == " · " + DIVERGENCE
    res2, _ = ses.call("search_bank_analytics", {"query": "ключевая ставка пауза риторика"})
    assert "divergence" not in res2 and ToolSession.note(res2) == ""


def test_draft_prompt_has_house_view_block_and_rule():
    assert "без пометки ПЕРЕСМОТРЕНО" in DRAFT_SYSTEM and "Текущий взгляд Альфа-Банка" in DRAFT_SYSTEM
    ses = session()
    ses.call("search_bank_analytics", {"query": QUERY})
    block = house_view_block(TODAY, ses.sources)
    label = next(s.label for s in ses.sources if s.doc.startswith("Ключевая ставка ЦБ РФ") and "14,0% до конца года" in s.text)
    line = next(x for x in block.splitlines() if x.startswith("- Ключевая ставка на конец 2026"))
    assert f"источник {label}" in line and "13,5%" in line
    assert "источник" not in next(x for x in block.splitlines() if x.startswith("- Рост ВВП за 2027"))
    DRAFT_SYSTEM.format(roles="…")  # в правиле нет фигурных скобок, ломающих format


class ScriptedLLM:
    """Цикл ищет позицию банка запросом, который находит и старый, и новый прогноз."""

    def __init__(self):
        self.systems = []
        self.actions = [{"thought": "план", "plan": ["новость", "позиция банка"], "action": "search_bank_analytics",
                         "args": {"query": QUERY}},
                        {"thought": "достаточно", "action": "final"}]

    def complete(self, role, messages, **kw):
        system = messages[0]["content"]
        self.systems.append(system)
        if "проверяющий" in system:
            out = {"ok": True, "issues": [], "research_query": ""}
        elif "составь карточку" in system:
            out = {"type": "info", "impact": "low", "topics": ["ставка"], "title": "Пресс-конференция ЦБ 23 октября",
                   "takeaway": {"cfo": "Долг не подешевеет до конца года", "ceo": "Деньги останутся дорогими",
                                "treasurer": "Платежи по кредиту не снизятся"},
                   "why_short": "У компании кредит и лизинг",
                   "key_points": [{"t": "У компании кредит и лизинг", "p": "ai"}, {"t": "Ставка решает стоимость долга", "p": "ai"}],
                   "blocks": {"what": [], "why": [], "risks": [], "opportunities": [], "actions": [],
                              "bank": {"quote": "", "src": None}}}
        else:
            out = self.actions.pop(0)
        return Completion(json.dumps(out, ensure_ascii=False), role, 100, 50, 0.01, 5, False)


def test_agent_trace_shows_the_decision():
    ev = SAMPLE["ev-e31c37f5eb"]
    ev = ev | {"topics": okved.topics_in(ev["text"], title=ev["title"])}
    p = profiles.get("U_25835")
    llm = ScriptedLLM()
    res = Agent(llm, bank_index(), kb.BM25Index([]), today=TODAY).run(ev, "U_25835", match(ev, profiles.features(p), p["bank_products"]))
    step = next(t for t in res.trace if t["title"].startswith("search_bank_analytics"))
    assert "берём более свежий" in step["detail"] and DIVERGENCE in step["detail"]
    draft_system = next(s for s in llm.systems if "составь карточку" in s)
    assert "Текущий взгляд Альфа-Банка" in draft_system and "ПЕРЕСМОТРЕНО" in draft_system


# ---------- проверка карточки ----------

def test_validator_flags_revised_bank_position_only_when_asked():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert validate_card(card, today=TODAY).ok  # по умолчанию — как раньше: карточка грузится при старте
    rep = validate_card(card, today=TODAY, check_revised=True)
    hits = [e for e in rep.errors if "пересмотрена" in e]
    assert hits and all("с 11.09 банк ждёт 14,0%" in e for e in hits)
    assert any(e.startswith("key_points[1]") for e in hits) and any(e.startswith("blocks.bank.quote") for e in hits)
    # вывод ИИ со старым числом — предупреждение, не ошибка
    assert any(w.startswith("blocks.opportunities[0]") and "пересмотрена" in w for w in rep.warnings)


def test_validator_keeps_news_that_states_the_old_value():
    """«ЦБ снизил ставку до 13,5%» в заголовке — событие, а не позиция банка, если это говорит сама новость."""
    card = json.loads(CARD.read_text(encoding="utf-8"))
    card["title"] = "ЦБ снизил ключевую ставку до 13,5%"
    card["takeaway"]["cfo"] = "Ставка 13,5% на конец 2026 г. удешевит ваш долг"
    rep = validate_card(card, today=TODAY, check_revised=True)
    assert any(e.startswith("title:") for e in rep.errors) and any(e.startswith("takeaway.cfo") for e in rep.errors)
    card["blocks"]["what"][0] = {"t": "Банк России снизил ключевую ставку до 13,5% годовых", "p": "fact", "src": 0}
    rep = validate_card(card, today=TODAY, check_revised=True)
    assert not any(e.startswith(("title:", "takeaway.")) for e in rep.errors)
    assert any(e.startswith("key_points[1]") for e in rep.errors)  # позиция банка по-прежнему пересмотрена


def test_validator_accepts_bank_quote_from_the_newer_document():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    card["blocks"]["bank"] = {"quote": "Если раньше мы ориентировались на ставку на уровне 13,5% на конец 2026 г., то теперь мы "
                                       "ожидаем ее сохранения на текущем значении 14,0% до конца года.",
                              "doc": "Ключевая ставка ЦБ РФ", "date": "2026-09-11", "stale": False}
    rep = validate_card(card, today=TODAY, check_revised=True)
    assert not any(e.startswith("blocks.bank.quote") for e in rep.errors)
