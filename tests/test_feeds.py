"""Проверка веб-источников (RSS) без сети (перенесено из бота): pytest tests/test_feeds.py"""
import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from agent.ingest.feeds import discover_feed, domain_of, fetch_feed_posts, normalize_site

NOW = datetime.now(timezone.utc)

RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Forbes Russia</title>
<item>
  <title>Банк X купил финтех Y за $1 млрд</title>
  <link>https://www.forbes.ru/finansy/1</link>
  <pubDate>{format_datetime(NOW - timedelta(hours=2))}</pubDate>
  <description><![CDATA[<p>Крупнейшая &amp; знаковая сделка года на финансовом рынке.</p>]]></description>
</item>
<item>
  <title>Старая новость, которой не место в дайджесте</title>
  <link>https://www.forbes.ru/finansy/2</link>
  <pubDate>{format_datetime(NOW - timedelta(hours=40))}</pubDate>
  <description>Сорокачасовой давности материал.</description>
</item>
<item>
  <title>Новость без даты — пропускается</title>
  <link>https://www.forbes.ru/finansy/3</link>
</item>
</channel></rss>"""

HTML_WITH_LINK = (
    '<html><head><link rel="alternate" type="application/rss+xml" '
    'title="RSS" href="/newrss.xml"></head><body>Forbes</body></html>'
)


class FakeResp:
    def __init__(self, body, status_code=200):
        self.text = body if isinstance(body, str) else body.decode("utf-8")
        self.content = body.encode("utf-8") if isinstance(body, str) else body
        self.status_code = status_code

    def raise_for_status(self):
        pass


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes  # {подстрока URL: тело ответа}

    async def get(self, url, **kw):
        for key, body in self.routes.items():
            if key in url:
                return FakeResp(body)
        return FakeResp("not found", 404)


async def _main():
    assert normalize_site("forbes.ru") == "https://forbes.ru"
    assert domain_of("https://www.forbes.ru/newrss.xml") == "forbes.ru"

    # автопоиск ленты: сайт отдаёт HTML со ссылкой на RSS
    http = FakeHTTP({"newrss.xml": RSS, "forbes.ru": HTML_WITH_LINK})
    feed_url, title = await discover_feed(http, "forbes.ru")
    assert feed_url.endswith("/newrss.xml") and title == "Forbes Russia", (feed_url, title)

    # чтение статей: свежая попадает, старая и без даты — нет
    posts = await fetch_feed_posts(http, feed_url, title, hours=24)
    assert len(posts) == 1, [p.text for p in posts]
    p = posts[0]
    assert p.link == "https://www.forbes.ru/finansy/1"
    assert "Банк X" in p.text and "& знаковая" in p.text and "<p>" not in p.text
    assert p.channel_title == "Forbes Russia" and p.channel == "forbes.ru"

    # прямая ссылка на ленту тоже работает
    feed_url2, _ = await discover_feed(http, "https://www.forbes.ru/newrss.xml")
    assert feed_url2.endswith("/newrss.xml")

    # сайт без ленты -> понятная ошибка
    http2 = FakeHTTP({"nofeed.ru": "<html><body>ничего</body></html>"})
    try:
        await discover_feed(http2, "nofeed.ru")
        raise AssertionError("должна быть ошибка про RSS")
    except ValueError as e:
        assert "RSS" in str(e)



def test_main():
    asyncio.run(_main())
