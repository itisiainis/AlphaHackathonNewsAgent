# Сервис новостной аналитики: API + интерфейс в одном контейнере
FROM python:3.13-slim

# OCR для сканов аналитики банка (agent/ingest/bank_docs.py): Tesseract с русским языком и poppler
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY agent ./agent
COPY backend ./backend
COPY frontend ./frontend
COPY data/company_profiles.json data/sources.json ./data/
COPY data/bank ./data/bank
COPY data/news/events.sample.json ./data/news/
COPY data/cards ./data/cards

RUN useradd --create-home app && mkdir -p /app/data/cache && chown -R app /app/data
USER app

ENV COLLECT_ON_START=1
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
