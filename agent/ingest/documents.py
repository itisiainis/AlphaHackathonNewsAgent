"""Извлечение текста из PDF-вложений (исследования, отчёты)."""

MAX_PAGES = 40      # читаем не больше этого числа страниц
MAX_CHARS = 20000   # и не больше этого объёма текста


def extract_pdf_text(path: str, max_pages: int = MAX_PAGES, max_chars: int = MAX_CHARS) -> str:
    """Текст PDF-файла. Пустая строка, если извлечь не удалось (скан без текста и т.п.)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        chunks, total = [], 0
        for page in reader.pages[:max_pages]:
            t = (page.extract_text() or "").strip()
            if t:
                chunks.append(t)
                total += len(t)
            if total >= max_chars:
                break
        return "\n".join(chunks)[:max_chars].strip()
    except Exception:
        return ""
