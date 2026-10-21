"""Веб-источники: новостные сайты (Forbes, РБК и т.п.) через RSS/Atom-ленты."""
import calendar
import html as html_lib
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

from .post import MIN_TEXT_LEN, Post

MAX_ITEMS = 60  # статей из одной ленты за раз
LINK_TAG_RE = re.compile(
    r"<link[^>]+type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*>", re.I
)
HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']", re.I)
COMMON_PATHS = ("/rss", "/feed", "/rss.xml", "/feed.xml", "/newrss.xml", "/rss/all")


def normalize_site(ref: str) -> str:
    ref = ref.strip()
    if not re.match(r"https?://", ref):
        ref = "https://" + ref
    return ref


def domain_of(url: str) -> str:
    return urlparse(normalize_site(url)).netloc.replace("www.", "")


def _parse_feed(content):
    import feedparser

    d = feedparser.parse(content)
    return d if getattr(d, "entries", None) else None


async def _try_feed(http, url: str):
    try:
        r = await http.get(url)
        if r.status_code >= 400:
            return None
        return _parse_feed(r.content)
    except Exception:
        return None


async def discover_feed(http, ref: str) -> tuple[str, str]:
    """Адрес сайта или ленты -> (feed_url, название). ValueError, если лента не нашлась."""
    url = normalize_site(ref)
    d = await _try_feed(http, url)
    if d:
        return url, (d.feed.get("title") or domain_of(url))
    try:
        r = await http.get(url)
        page = r.text
    except Exception as err:
        raise ValueError("сайт недоступен") from err
    candidates = []
    for tag in LINK_TAG_RE.findall(page):
        m = HREF_RE.search(tag)
        if m:
            candidates.append(urljoin(url, m.group(1)))
    candidates += [urljoin(url, p) for p in COMMON_PATHS]
    for cand in candidates[:8]:
        d = await _try_feed(http, cand)
        if d:
            return cand, (d.feed.get("title") or domain_of(url))
    raise ValueError(
        "не нашёл RSS-ленту на сайте — пришлите прямую ссылку на ленту (обычно /rss)"
    )


async def fetch_feed_posts(http, feed_url: str, title: str, hours: int = 24) -> list[Post]:
    """Статьи ленты за последние `hours` часов, приведённые к формату постов."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    r = await http.get(feed_url)
    r.raise_for_status()
    d = _parse_feed(r.content)
    if not d:
        return []
    site = domain_of(feed_url)
    feed_title = title or d.feed.get("title") or site

    posts: list[Post] = []
    for e in d.entries[:MAX_ITEMS]:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if not t:
            continue  # без даты нельзя понять свежесть — пропускаем
        date = datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
        if date < since:
            continue
        link = (e.get("link") or "").strip()
        head = (e.get("title") or "").strip()
        summary = re.sub(r"<[^>]+>", " ", e.get("summary", "") or "")
        summary = html_lib.unescape(" ".join(summary.split()))[:600]
        text = (head + ("\n" + summary if summary else "")).strip()
        if not link or len(text) < MIN_TEXT_LEN:
            continue
        posts.append(
            Post(
                channel=site,
                channel_title=feed_title,
                msg_id=0,
                text=text,
                date=date,
                url=link,
            )
        )
    return posts
