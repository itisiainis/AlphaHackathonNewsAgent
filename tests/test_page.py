"""Страница для пользователя: без служебных комментариев и пометок для разработчиков."""
import re

from frontend.page import render, strip_comments


def test_no_developer_comments_in_served_page():
    html = render("null")
    assert "/*" not in html and not re.search(r"(?m)^\s*//", html) and " // " not in html
    assert "прототип" not in html.lower() and "демо-режим" not in html


def test_urls_and_data_survive():
    assert strip_comments("const u = 'https://x.ru/a'; // хвост\n") == "const u = 'https://x.ru/a';\n"
    assert "// из новости" in render('{"t": "a // из новости"}')
