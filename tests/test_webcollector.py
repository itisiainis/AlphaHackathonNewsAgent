"""Проверка веб-парсера t.me (без сети) (перенесено из бота): pytest tests/test_webcollector.py"""
import asyncio
from datetime import datetime, timedelta, timezone

from agent.ingest.webcollector import (
    _parse_views,
    fetch_recent_web,
    parse_channel_page,
    resolve_channel_web,
)

NOW = datetime.now(timezone.utc)

PAGE = f"""<html><head><meta property="og:title" content="Тестовый &amp;Канал"></head><body>
<section class="tgme_channel_history js-message_history">
<div class="tgme_widget_message force_userpic js-widget_message" data-post="testchan/100">
  <div class="tgme_widget_message_bubble">
    <div class="tgme_widget_message_footer compact js-message_footer">
      <span class="tgme_widget_message_views">987</span>
      <a class="tgme_widget_message_date" href="https://t.me/testchan/100"><time datetime="{(NOW - timedelta(hours=30)).isoformat()}" class="time">07:00</time></a>
    </div>
  </div>
</div>
<div class="tgme_widget_message force_userpic js-widget_message" data-post="testchan/101">
  <div class="tgme_widget_message_bubble">
    <div class="tgme_widget_message_text js-message_text" dir="auto">Первая новость про банки и ставки регулятора.<br/>Подробности &amp; детали по <a href="https://example.com">ссылке</a>.</div>
    <div class="tgme_widget_message_footer compact js-message_footer">
      <span class="tgme_widget_message_views">1.2K</span>
      <a class="tgme_widget_message_date" href="https://t.me/testchan/101"><time datetime="{(NOW - timedelta(hours=2)).isoformat()}" class="time">08:15</time></a>
    </div>
  </div>
</div>
<div class="tgme_widget_message force_userpic js-widget_message" data-post="testchan/102">
  <div class="tgme_widget_message_bubble">
    <div class="tgme_widget_message_document_title accent_color" dir="auto">Global_Banking_Review_2026.pdf</div>
    <div class="tgme_widget_message_text js-message_text" dir="auto">Свежее исследование банковского рынка</div>
    <div class="tgme_widget_message_footer compact js-message_footer">
      <span class="tgme_widget_message_views">500</span>
      <a class="tgme_widget_message_date" href="https://t.me/testchan/102"><time datetime="{(NOW - timedelta(hours=1)).isoformat()}" class="time">09:15</time></a>
    </div>
  </div>
</div>
</section></body></html>"""

EMPTY_PAGE = '<html><body><section class="tgme_channel_history"></section></body></html>'
NO_PREVIEW_PAGE = '<html><body><div class="tgme_page">Preview channel</div></body></html>'


class FakeResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        pass


class FakeHTTP:
    """Первая страница — PAGE, дальше — пустая история (конец пагинации)."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = 0

    async def get(self, url, params=None):
        self.calls += 1
        return FakeResp(self.pages.pop(0) if self.pages else EMPTY_PAGE)


async def _main():
    # разбор страницы
    title, posts, min_id = parse_channel_page(PAGE, "testchan")
    assert title == "Тестовый &Канал"
    assert min_id == 100
    assert len(posts) == 2, "пост без текста (100) должен быть отброшен"

    p = next(x for x in posts if x.msg_id == 101)
    assert p.views == 1200
    assert "Первая новость" in p.text and "& детали" in p.text and "<" not in p.text
    assert "\n" in p.text  # <br/> стал переносом
    assert p.link == "https://t.me/testchan/101"

    # PDF-вложение: имя файла попадает в текст и doc_name
    d = next(x for x in posts if x.msg_id == 102)
    assert d.doc_name == "Global_Banking_Review_2026.pdf"
    assert "[Приложены файлы: Global_Banking_Review_2026.pdf]" in d.text

    # выборка за 24 часа: пост 30-часовой давности отсекается
    http = FakeHTTP([PAGE])
    recent = await fetch_recent_web(http, "testchan", hours=24)
    assert sorted(x.msg_id for x in recent) == [101, 102]

    # resolve: канал с историей
    http2 = FakeHTTP([PAGE])
    username, t = await resolve_channel_web(http2, "https://t.me/s/testchan")
    assert username == "testchan" and t == "Тестовый &Канал"

    # resolve: веб-просмотр выключен
    http3 = FakeHTTP([NO_PREVIEW_PAGE])
    try:
        await resolve_channel_web(http3, "@closed")
        raise AssertionError("должна быть ошибка про веб-просмотр")
    except ValueError as e:
        assert "веб-просмотр" in str(e)

    # разбор просмотров
    assert _parse_views("987") == 987
    assert _parse_views("1.2K") == 1200
    assert _parse_views("3,4M") == 3400000
    assert _parse_views("мусор") == 0



def test_main():
    asyncio.run(_main())
