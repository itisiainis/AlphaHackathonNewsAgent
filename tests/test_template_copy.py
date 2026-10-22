"""Обещанные клиенту тексты страницы на месте (docs/product.md): плашка без ИИ, оговорка под разбором,
тихий день на главной, вставка своей новости. Поведение проверяется в браузере; здесь — что тексты не потерялись."""
from frontend.page import render

HTML = render("null")


def test_no_model_banner():
    assert "ИИ-аналитик недоступен — краткие разборы: факты, позиция банка и расчёт, без выводов ИИ" in HTML


def test_investment_disclaimer():
    assert "Не является индивидуальной инвестиционной рекомендацией." in HTML
    assert "Центр макроэкономического анализа Альфа-Банка" in HTML


def test_home_quiet_day_and_short_list():
    for s in ("Сегодня ничего, что требует решения.", "Последнее важное:", "Значимых событий за сутки нет",
              "Сегодня по-настоящему важно"):
        assert s in HTML


def test_paste_flow_copy():
    assert "Связи с вашей компанией не нашли — полный разбор ИИ не запускали" in HTML
    assert "Этот материал уже разобран — открыли готовый разбор" in HTML
    assert 'id="an-err" role="alert"' in HTML


def test_money_words_and_quote_toggle():
    for s in ("экономия", "доп. расходы", "потери", "доп. доход", "Читать полностью"):
        assert s in HTML
    assert "Escape" in HTML
