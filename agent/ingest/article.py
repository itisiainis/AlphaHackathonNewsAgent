"""Полный текст материала по ссылке: статья (HTML), пост Telegram, PDF.

Зачем: RSS отдаёт только анонс (до 600 символов) — для фактов, цифр и причин этого мало
(критерий 03). Перед разбором агентом короткая новость дополняется полным текстом.
Пользователь также может вставить ссылку вместо текста («Разобрать новость»).

Безопасность (ограничение кейса 05, защита от SSRF): ссылку присылает пользователь, поэтому
- только http/https и стандартные порты;
- адрес хоста разрешается заранее, и запрещены внутренние сети: loopback, частные, link-local,
  зарезервированные и multicast-адреса (иначе через сервис можно было бы обратиться к внутренним
  ресурсам сервера);
- редиректы проходят ту же проверку на каждом шаге, не больше MAX_REDIRECTS;
- ограничены размер ответа и время.

Как извлекается основной текст HTML (без сторонних библиотек): парсер стандартной библиотеки
проходит документ, пропускает служебные блоки (script, style, nav, header, footer, aside, form),
собирает абзацы <p>, заголовки и пункты списков. Если на странице есть <article>, берутся абзацы
только из него. Короткие абзацы (меню, подписи, кнопки) отбрасываются порогом длины.
"""
from __future__ import annotations

import html as html_lib
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 4 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT = 15
MAX_CHARS = 8000
MIN_PARAGRAPH = 40
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button", "figure"}
TEXT_TAGS = {"p", "h1", "h2", "h3", "li", "blockquote"}


class UnsafeURL(ValueError):
    """Ссылка ведёт во внутреннюю сеть или использует недопустимую схему."""


@dataclass
class Article:
    url: str
    title: str
    text: str
    published: str = ""
    kind: str = "html"   # html | telegram | pdf


def check_url(url: str) -> str:
    """Проверка ссылки до запроса. Возвращает нормализованный адрес или бросает UnsafeURL."""
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise UnsafeURL("нужна ссылка http(s)://")
    if u.port not in (None, 80, 443):
        raise UnsafeURL("нестандартный порт")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror as e:
        raise UnsafeURL("сайт не найден") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise UnsafeURL("ссылка ведёт во внутреннюю сеть")
    return u.geturl()


def safe_get(url: str, client: httpx.Client | None = None) -> httpx.Response:
    """GET с проверкой каждого адреса в цепочке редиректов и ограничением размера."""
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": UA}, timeout=TIMEOUT, follow_redirects=False)
    try:
        for _ in range(MAX_REDIRECTS + 1):
            url = check_url(url)
            with client.stream("GET", url) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                r.raise_for_status()
                chunks, size = [], 0
                for chunk in r.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("материал слишком большой")
                    chunks.append(chunk)
                # iter_bytes уже распаковал gzip/br — заголовки сжатия убираем, иначе распаковка повторится
                headers = {k: v for k, v in r.headers.items() if k.lower() not in ("content-encoding", "content-length")}
                return httpx.Response(r.status_code, headers=headers, content=b"".join(chunks), request=r.request)
        raise ValueError("слишком много перенаправлений")
    finally:
        if own:
            client.close()


CONTAINERS = {"div", "section", "article", "main", "td", "body"}


class _MainText(HTMLParser):
    """Собирает абзацы вместе с цепочкой блоков-предков, чтобы потом выбрать основной блок страницы."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.stack: list[int] = []            # открытые блоки (номера)
        self.tags: list[str] = []             # теги открытых блоков
        self.n = 0
        self.cur: list[str] | None = None
        self.paras: list[tuple[tuple, str]] = []   # (цепочка предков, текст)
        self.title = ""
        self.meta: dict[str, str] = {}
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            key = a.get("property") or a.get("name") or ""
            if key in ("og:title", "article:published_time", "og:description", "description"):
                self.meta[key] = a.get("content", "") or ""
        if tag == "title":
            self._in_title = True
        if tag in SKIP:
            self.skip += 1
        if tag in CONTAINERS:
            self.n += 1
            self.stack.append(self.n)
            self.tags.append(tag)
        if tag in TEXT_TAGS and not self.skip:
            self.cur = []
        if tag == "br" and self.cur is not None:
            self.cur.append(" ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in SKIP and self.skip:
            self.skip -= 1
        if tag in TEXT_TAGS and self.cur is not None:
            t = re.sub(r"\s+", " ", "".join(self.cur)).strip()
            if len(t) >= MIN_PARAGRAPH or (tag in ("h1", "h2") and len(t) > 10):
                self.paras.append((tuple(self.stack), t))
            self.cur = None
        if tag in CONTAINERS and tag in self.tags:
            # закрываем до ближайшего блока с этим тегом (HTML бывает невалидным)
            while self.tags:
                self.stack.pop()
                if self.tags.pop() == tag:
                    break

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self.cur is not None and not self.skip:
            self.cur.append(data)


JSON_STR_RE = re.compile(r'"((?:[^"\\]|\\.){200,})"')
BOILERPLATE = re.compile(r"Сетевое издание|зарегистрирован\w* Федеральной службой|Роскомнадзор|cookie|"
                         r"Все права защищены|©|политик\w* конфиденциальности|18\+", re.I)
LD_BODY_RE = re.compile(r'"articleBody"\s*:\s*"((?:[^"\\]|\\.)+)"')


def _script_text(page: str) -> str:
    """Текст статьи из JSON на странице (сайты на Vue/React/Next): сначала articleBody из JSON-LD,
    иначе самые длинные строки, похожие на русский связный текст (много пробелов и знаков препинания)."""
    import json as _json

    def unq(raw: str) -> str:
        try:
            v = _json.loads(f'"{raw}"')
        except ValueError:
            return ""
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_lib.unescape(v))).strip()

    m = LD_BODY_RE.search(page)
    if m and len(body := unq(m.group(1))) > 200:
        return body[:MAX_CHARS]
    cands = []
    for pos, raw in enumerate(JSON_STR_RE.findall(page)):
        t = re.sub(r"\bdata-[\w-]+>?", " ", unq(raw))
        t = re.sub(r"\s+", " ", t).strip()
        cyr = len(re.findall(r"[а-яё]", t, re.I))
        if (cyr > 0.5 * len(t) and t.count(" ") > len(t) / 12 and re.search(r"[.,]", t)
                and not BOILERPLATE.search(t)):
            cands.append((pos, t))
    top = sorted(cands, key=lambda x: -len(x[1]))[:6]
    out, seen = [], set()
    for _, t in sorted(top):  # в порядке появления на странице
        if t[:80] not in seen:
            seen.add(t[:80])
            out.append(t)
    return "\n\n".join(out)[:MAX_CHARS]


def _decode(r: httpx.Response) -> str:
    """Кодировка: из заголовка, иначе из <meta charset> (у части сайтов windows-1251 без заголовка)."""
    if "charset=" in r.headers.get("content-type", "").lower():
        return r.text
    head = r.content[:4096].decode("ascii", "ignore")
    m = re.search(r'charset=["\']?([\w-]+)', head, re.I)
    enc = m.group(1) if m else "utf-8"
    try:
        return r.content.decode(enc, "replace")
    except LookupError:
        return r.content.decode("utf-8", "replace")


def extract_html(page: str) -> tuple[str, str, str]:
    """(заголовок, основной текст, дата публикации) из HTML-страницы.

    Выбор основного блока (по мотивам Readability): длина каждого абзаца добавляется его блоку-
    родителю целиком и «деду» наполовину — так текст, где каждый абзац обёрнут в свой <div>,
    всё равно собирается в общем контейнере. Побеждает блок с наибольшей суммой; берутся абзацы,
    лежащие внутри него. Ссылки «читайте также» и ленты анонсов проигрывают сплошному тексту статьи."""
    p = _MainText()
    p.feed(page)
    score: dict[int, float] = {}
    for chain, t in p.paras:
        bonus = len(t) + (t.count(",") * 10)  # в связном тексте больше запятых, чем в заголовках-ссылках
        if chain:
            score[chain[-1]] = score.get(chain[-1], 0) + bonus
        if len(chain) > 1:
            score[chain[-2]] = score.get(chain[-2], 0) + bonus / 2
    best = max(score, key=score.get) if score else None
    paras = [t for chain, t in p.paras if best in chain] if best else []
    seen, uniq = set(), []
    for t in paras:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    text = "\n\n".join(uniq)[:MAX_CHARS]
    if len(text) < 300:  # страница собирается скриптом — текст лежит в JSON внутри страницы
        text = _script_text(page) or text
    if len(text) < 200:
        text = p.meta.get("og:description") or p.meta.get("description") or text
    title = (p.meta.get("og:title") or p.title).strip()
    return html_lib.unescape(title), html_lib.unescape(text), p.meta.get("article:published_time", "")[:10]


TG_POST_RE = re.compile(r"^https?://t\.me/(?:s/)?([A-Za-z0-9_]{4,})/(\d+)")
TG_TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
TG_TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')


def _get(url: str, client) -> httpx.Response:
    try:
        return safe_get(url, client)
    except httpx.HTTPStatusError as e:
        raise ValueError(f"сайт ответил ошибкой {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise ValueError("сайт недоступен") from e


def fetch_article(url: str, client: httpx.Client | None = None) -> Article:
    """Материал по ссылке в виде текста. Бросает UnsafeURL или ValueError с понятным сообщением."""
    m = TG_POST_RE.match(url.strip())
    if m:  # пост Telegram: встраиваемый вид отдаёт текст поста без ключей
        r = _get(f"https://t.me/{m.group(1)}/{m.group(2)}?embed=1", client)
        tx = TG_TEXT_RE.search(r.text)
        if not tx:
            raise ValueError("пост не найден или канал закрыт")
        text = html_lib.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", "\n", tx.group(1)))).strip()
        tm = TG_TIME_RE.search(r.text)
        first = text.split("\n", 1)[0][:200]
        return Article(url, first, text[:MAX_CHARS], tm.group(1)[:10] if tm else "", "telegram")
    r = _get(url, client)
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        import tempfile
        from pathlib import Path

        from .documents import extract_pdf_text
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "doc.pdf"
            path.write_bytes(r.content)
            text = extract_pdf_text(str(path))
        if not text:
            raise ValueError("в PDF нет текстового слоя (скан) — вставьте текст вручную")
        return Article(url, text.split("\n", 1)[0][:200], text[:MAX_CHARS], "", "pdf")
    title, text, published = extract_html(_decode(r))
    if len(text) < 80:
        raise ValueError("не удалось извлечь текст со страницы — вставьте текст вручную")
    return Article(url, title, text, published, "html")
