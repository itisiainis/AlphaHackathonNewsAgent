"""Полный текст по ссылке: основной блок статьи, текст из JSON, Telegram-пост, защита от SSRF — без сети."""
import httpx
import pytest

from agent.ingest import article
from agent.ingest.article import UnsafeURL, check_url, extract_html, fetch_article

PARA = "Банк России сохранил ключевую ставку на уровне 14% годовых, отметив ускорение устойчивой инфляции. "

ARTICLE_PAGE = f"""<html><head><title>Сайт</title><meta property="og:title" content="ЦБ сохранил ставку">
<meta property="article:published_time" content="2026-09-11T13:30:00+03:00"></head><body>
<header><p>Меню сайта: новости, экономика, финансы, спорт и всё остальное подряд</p></header>
<div class="related"><ul><li><a>Читайте также: ещё одна новость про что-то другое совсем</a></li>
<li><a>Читайте также: и ещё одна новость про что-то другое совсем</a></li></ul></div>
<div class="body">
  <div class="p"><p>Первый абзац. {PARA * 2}</p></div>
  <div class="p"><p>Второй абзац. {PARA * 2}</p></div>
  <div class="p"><p>Третий абзац. {PARA * 2}</p></div>
</div>
<footer><p>© Издание, все права защищены, перепечатка запрещена без разрешения</p></footer></body></html>"""


def test_main_block_wins_over_related_links_and_footer():
    title, text, published = extract_html(ARTICLE_PAGE)
    assert title == "ЦБ сохранил ставку" and published == "2026-09-11"
    assert text.count("ключевую ставку") == 6
    assert "Читайте также" not in text and "Меню" not in text and "©" not in text


def test_script_rendered_page_uses_json_text_without_boilerplate():
    body = "Минфин объявил о росте заимствований, компании ожидают удорожания кредитов в 2027 году. " * 4
    page = ('<html><head><meta property="og:title" content="Минфин"></head><body><div id="app"></div>'
            f'<script>window.__STATE__={{"a":"{body}","b":"Сетевое издание зарегистрировано Федеральной службой '
            'по надзору в сфере связи, все права защищены, перепечатка только с разрешения редакции сайта"}}</script></body></html>')
    _, text, _ = extract_html(page)
    assert "Минфин объявил" in text and "Сетевое издание" not in text


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://10.1.2.3/x", "http://169.254.169.254/meta",
                                 "file:///etc/passwd", "ftp://example.org/", "http://[::1]/", "https://8.8.8.8:8443/"])
def test_ssrf_blocked(url):
    with pytest.raises(UnsafeURL):
        check_url(url)


def _client(routes):
    def handler(req):
        for key, (status, headers, body) in routes.items():
            if key in str(req.url):
                return httpx.Response(status, headers=headers, content=body)
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_redirect_to_internal_address_blocked(monkeypatch):
    monkeypatch.setattr(article, "check_url", lambda u: (_ for _ in ()).throw(UnsafeURL("внутренняя сеть"))
                        if "169.254" in u else u)
    c = _client({"news.example": (302, {"location": "http://169.254.169.254/latest"}, b"")})
    with pytest.raises(ValueError, match="внутренняя"):
        fetch_article("https://news.example/a", c)


def test_telegram_post_and_cp1251_page(monkeypatch):
    monkeypatch.setattr(article, "check_url", lambda u: u)
    tg = ('<div class="tgme_widget_message_text js-message_text">ЦБ сохранил ставку<br/>Подробности: 14%</div>'
          '<time datetime="2026-09-11T10:00:00+00:00">').encode()
    page = ARTICLE_PAGE.replace("<head>", '<head><meta charset="windows-1251">').encode("cp1251", "replace")
    c = _client({"t.me/banksta/100": (200, {"content-type": "text/html"}, tg),
                 "site.example": (200, {"content-type": "text/html"}, page)})
    a = fetch_article("https://t.me/banksta/100", c)
    assert a.kind == "telegram" and "14%" in a.text and a.published == "2026-09-11"
    b = fetch_article("https://site.example/news/1", c)
    assert "ключевую ставку" in b.text, "кодировка windows-1251 без заголовка распознаётся по meta"


def test_analyze_by_link_only(tmp_path, monkeypatch):
    import importlib
    import os

    from fastapi.testclient import TestClient
    os.environ.update(APP_CACHE_DIR=str(tmp_path), APP_TODAY="2026-10-21", RATE_LIMIT_PER_MIN="100")
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as m
    m = importlib.reload(m)
    monkeypatch.setattr(article, "fetch_article",
                        lambda url, client=None: article.Article(url, "ЦБ сохранил ставку", PARA * 3, "2026-09-11"))
    c = TestClient(m.app)
    r = c.post("/api/analyze", json={"company_id": "U_25835", "url": "https://news.example/a"})
    assert r.status_code == 200 and r.json()["mode"] == "rules", "ссылка прочитана; без ключа — черновой разбор правилами"
    assert r.json()["card"]["sources"][0]["url"] == "https://news.example/a"
    monkeypatch.setattr(article, "fetch_article", lambda url, client=None: (_ for _ in ()).throw(UnsafeURL("внутренняя сеть")))
    r = c.post("/api/analyze", json={"company_id": "U_25835", "url": "http://127.0.0.1/"})
    assert r.status_code == 422 and "внутренняя" in r.json()["detail"]


# ---------- не статья: раздел сайта, главная, страница без текста ----------
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_section_page_is_not_an_article(monkeypatch):
    """interfax.ru/business/ становился карточкой «возможность, высокое влияние» по ленте заголовков."""
    monkeypatch.setattr(article, "check_url", lambda u: u)
    page = (FIXTURES / "interfax_business.html").read_bytes()
    c = _client({"interfax.ru/business": (200, {"content-type": "text/html; charset=utf-8"}, page)})
    with pytest.raises(article.NotArticle) as e:
        fetch_article("https://www.interfax.ru/business/", c)
    assert str(e.value) == "Это не статья — пришлите ссылку на новость"


def test_short_page_without_article_mark_is_not_an_article(monkeypatch):
    monkeypatch.setattr(article, "check_url", lambda u: u)
    stub = f"<html><head><title>Вход</title></head><body><div><p>{PARA}</p><p>{PARA}</p></div></body></html>".encode()
    note = stub.replace(b"<head>", b'<head><meta property="og:type" content="article">')
    c = _client({"login.example": (200, {"content-type": "text/html"}, stub),
                 "flash.example": (200, {"content-type": "text/html"}, note)})
    with pytest.raises(article.NotArticle):
        fetch_article("https://login.example/", c)
    # короткая заметка агентства (около 200 знаков) помечена как статья — это материал
    assert "ключевую ставку" in fetch_article("https://flash.example/news/1", c).text


def test_article_with_long_related_list_is_still_an_article():
    related = "".join(f'<li><a href="/n/{k}">Читайте также: другая новость номер {k} про экономику и рынки</a></li>'
                      for k in range(30))
    page = ARTICLE_PAGE.replace('<div class="related"><ul>', f'<div class="related"><ul>{related}')
    _, text, _, kind = article._extract(page)
    article.check_article(text, kind)  # не бросает NotArticle
    assert kind["page_links"] >= 15 and kind["article"]


def test_section_page_by_link_gives_clear_error(tmp_path, monkeypatch):
    import importlib
    import os

    from fastapi.testclient import TestClient
    os.environ.update(APP_CACHE_DIR=str(tmp_path), APP_TODAY="2026-10-21", RATE_LIMIT_PER_MIN="100")
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as m
    m = importlib.reload(m)
    page = (FIXTURES / "interfax_business.html").read_bytes()
    monkeypatch.setattr(article, "fetch_article", lambda url, client=None: fetch_article(
        url, _client({"interfax.ru": (200, {"content-type": "text/html; charset=utf-8"}, page)})))
    monkeypatch.setattr(article, "check_url", lambda u: u)
    r = TestClient(m.app).post("/api/brief", json={"company_id": "U_25835", "url": "https://www.interfax.ru/business/"})
    assert r.status_code == 422 and r.json()["detail"] == "Это не статья — пришлите ссылку на новость"
