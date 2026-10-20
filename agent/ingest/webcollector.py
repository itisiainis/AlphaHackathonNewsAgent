"""Чтение публичных каналов через веб-просмотр https://t.me/s/… — без ключей Telegram API.

Ограничения режима: работают только публичные каналы с включённым веб-просмотром
(у подавляющего большинства новостных каналов он включён), приватные недоступны.
"""
import html as html_lib
import re
from datetime import datetime, timedelta, timezone

from .post import MAX_PER_CHANNEL, MIN_TEXT_LEN, Post, normalize_ref

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
MAX_PAGES = 6  # максимум страниц истории на канал (по ~20 постов)

MSG_RE = re.compile(r'<div class="tgme_widget_message[^"]*"[^>]*data-post="([^"]+)"')
TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
DOC_RE = re.compile(r'<div class="tgme_widget_message_document_title[^"]*"[^>]*>(.*?)</div>', re.S)
TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')
VIEWS_RE = re.compile(r'<span class="tgme_widget_message_views">([^<]+)</span>')
TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]*)"')
HISTORY_MARK = "tgme_channel_history"


def _parse_views(s: str) -> int:
    s = s.strip().upper().replace(" ", "").replace(",", ".")
    mult = 1
    if s.endswith("K"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return 0


def _clean_text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html_lib.unescape(fragment).strip()


def parse_channel_page(page: str, username: str, title_fallback: str = ""):
    """Разбирает HTML страницы канала. Возвращает (title, [Post], min_msg_id)."""
    m = TITLE_RE.search(page)
    title = html_lib.unescape(m.group(1)) if m else (title_fallback or username)

    posts: list[Post] = []
    min_id = None
    matches = list(MSG_RE.finditer(page))
    for i, mt in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page)
        chunk = page[mt.start() : end]
        try:
            msg_id = int(mt.group(1).rsplit("/", 1)[1])
        except (IndexError, ValueError):
            continue
        min_id = msg_id if min_id is None else min(min_id, msg_id)

        times = TIME_RE.findall(chunk)
        if not times:
            continue
        try:
            date = datetime.fromisoformat(times[-1])  # последний <time> — дата поста
        except ValueError:
            continue
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)

        tx = TEXT_RE.search(chunk)
        text = _clean_text(tx.group(1)) if tx else ""
        docs = [_clean_text(d) for d in DOC_RE.findall(chunk)]
        docs = [d for d in docs if d]
        if docs:
            # содержимое файлов через веб-просмотр недоступно, но хотя бы называем их
            text = (text + "\n[Приложены файлы: " + ", ".join(docs) + "]").strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        vw = VIEWS_RE.search(chunk)
        posts.append(
            Post(
                channel=username.lower(),
                channel_title=title,
                msg_id=msg_id,
                text=text,
                date=date,
                views=_parse_views(vw.group(1)) if vw else 0,
                doc_name=docs[0] if docs else "",
            )
        )
    return title, posts, min_id


async def resolve_channel_web(http, ref: str) -> tuple[str, str]:
    """Проверяет доступность канала через веб-просмотр. Возвращает (username, title)."""
    name = normalize_ref(ref)
    if not name:
        raise ValueError("пустое имя канала")
    r = await http.get(f"https://t.me/s/{name}")
    if r.status_code == 404:
        raise ValueError("канал не найден")
    page = r.text
    if HISTORY_MARK not in page:
        raise ValueError(
            "канал не найден или у него выключен веб-просмотр "
            "(без ключей Telegram API такой канал недоступен)"
        )
    m = TITLE_RE.search(page)
    title = html_lib.unescape(m.group(1)) if m else name
    return name.lower(), title


async def fetch_recent_web(http, username: str, hours: int = 24) -> list[Post]:
    """Посты канала за последние `hours` часов через t.me/s (с пагинацией)."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    url = f"https://t.me/s/{username}"
    collected: dict[int, Post] = {}
    before = None
    title = username

    for _ in range(MAX_PAGES):
        params = {"before": before} if before else None
        r = await http.get(url, params=params)
        r.raise_for_status()
        page = r.text
        if HISTORY_MARK not in page:
            if not collected:
                raise ValueError("у канала выключен веб-просмотр")
            break
        title, posts, min_id = parse_channel_page(page, username, title)
        for p in posts:
            collected[p.msg_id] = p
        if min_id is None or min_id <= 1 or len(collected) >= MAX_PER_CHANNEL:
            break
        oldest = min((p.date for p in posts), default=None)
        if oldest is not None and oldest < since:
            break  # дошли до постов старше нужного окна
        before = min_id

    result = [p for p in collected.values() if p.date >= since]
    result.sort(key=lambda p: p.date, reverse=True)
    return result[:MAX_PER_CHANNEL]
