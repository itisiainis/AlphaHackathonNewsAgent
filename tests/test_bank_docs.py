"""Парсер аналитики банка: чистка ошибок OCR, даты, разметка фрагментов."""
from agent.ingest.bank_docs import chunk_pages, clean_ocr, detect_date


def test_ocr_fixes_found_in_case_materials():
    src = "Рост 5,3% r/r за 11126 и 0,4% rir в 2K26, доходность +17 6.п., noпрежнему 5 py6."
    out = clean_ocr(src)
    assert out == "Рост 5,3% г/г за 1П26 и 0,4% г/г в 2К26, доходность +17 б.п., попрежнему 5 руб."
    assert clean_ocr(out) == out, "чистка идемпотентна: повторный прогон ничего не меняет"


def test_latin_words_untouched():
    assert clean_ocr("данные HeadHunter и ETF") == "данные HeadHunter и ETF"


def test_dates_from_text_and_filename():
    assert detect_date("Центр\n7 сентября 2026\nМакростатистика", "x.pdf") == "2026-09-07"
    assert detect_date("без даты", "Бонд_Бриф_14092026.pdf") == "2026-09-14"
    assert detect_date("без даты", "Steels_15Apr2021.pdf") == "2021-04-15"


def test_chunks_are_tagged_with_sectors():
    ch = chunk_pages(["Продажи новых легковых автомобилей упали.\n\nЦены на сталь выросли."], size=30)
    assert ch[0]["sectors"] and "29" in ch[0]["sectors"] and "24" in ch[1]["sectors"]
