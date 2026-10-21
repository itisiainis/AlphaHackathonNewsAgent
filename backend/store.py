"""Хранилище карточек, обратной связи и журнала запусков агента (SQLite).

Карточки кешируются по паре (компания, карточка): повторный показ ленты не тратит бюджет модели.
У каждой карточки есть происхождение: seed — пример из frontend/mock_data.json (подготовлен командой
по материалам кейса), agent — собран агентом. Интерфейс показывает это пользователю.
Обратная связь и журнал запусков — данные для метрик качества (критерий 12).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards(company_id TEXT, card_id TEXT, origin TEXT, json TEXT, created REAL,
                                 PRIMARY KEY(company_id, card_id));
CREATE TABLE IF NOT EXISTS feedback(company_id TEXT, card_id TEXT, role TEXT, vote TEXT, ts REAL,
                                    PRIMARY KEY(company_id, card_id, role));
CREATE TABLE IF NOT EXISTS requests(ts REAL, company_id TEXT, card_id TEXT, role TEXT, kind TEXT, label TEXT, comment TEXT);
CREATE TABLE IF NOT EXISTS runs(ts REAL, company_id TEXT, event_id TEXT, ok INTEGER, errors TEXT,
                                cost REAL, ms INTEGER, tools INTEGER, revisions INTEGER);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    # ---------- карточки ----------
    def put_card(self, company_id: str, card: dict, origin: str) -> None:
        self._exec("INSERT OR REPLACE INTO cards VALUES(?,?,?,?,?)",
                   (company_id, card["id"], origin, json.dumps(card, ensure_ascii=False), time.time()))

    def cards(self, company_id: str) -> list[tuple[dict, str]]:
        rows = self._exec("SELECT json, origin FROM cards WHERE company_id=? ORDER BY created DESC", (company_id,))
        return [(json.loads(j), o) for j, o in rows.fetchall()]

    def card(self, company_id: str, card_id: str) -> dict | None:
        row = self._exec("SELECT json FROM cards WHERE company_id=? AND card_id=?", (company_id, card_id)).fetchone()
        return json.loads(row[0]) if row else None

    def companies_with_cards(self) -> list[str]:
        return [r[0] for r in self._exec("SELECT DISTINCT company_id FROM cards").fetchall()]

    def has_seed(self) -> bool:
        return bool(self._exec("SELECT 1 FROM cards WHERE origin='seed' LIMIT 1").fetchone())

    # ---------- обратная связь и журнал ----------
    def put_feedback(self, company_id, card_id, role, vote) -> None:
        self._exec("INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?)", (company_id, card_id, role, vote, time.time()))

    def log_run(self, company_id, event_id, ok, errors, cost, ms, tools, revisions) -> None:
        self._exec("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)",
                   (time.time(), company_id, event_id, int(ok), json.dumps(errors, ensure_ascii=False), cost, ms, tools, revisions))

    def put_request(self, company_id: str, card_id: str, role: str, kind: str, label: str, comment: str) -> None:
        """Запрос клиента менеджеру из карточки: в рабочей версии уходит в CRM банка, здесь — в журнал."""
        self._exec("INSERT INTO requests VALUES(?,?,?,?,?,?,?)", (time.time(), company_id, card_id, role, kind, label, comment))

    def metrics(self) -> dict:
        q = lambda sql: self._exec(sql).fetchone()  # noqa: E731
        fb_up, fb_down = q("SELECT SUM(vote='up'), SUM(vote='down') FROM feedback")
        runs, ok, cost, ms = q("SELECT COUNT(*), SUM(ok), SUM(cost), AVG(ms) FROM runs")
        by_origin = dict(self._exec("SELECT origin, COUNT(*) FROM cards GROUP BY origin").fetchall())
        votes = (fb_up or 0) + (fb_down or 0)
        req, req_cards = q("SELECT COUNT(*), COUNT(DISTINCT company_id || card_id) FROM requests")
        return {
            "cards": by_origin,
            "feedback": {"useful": fb_up or 0, "not_relevant": fb_down or 0,
                         "useful_share": round((fb_up or 0) / votes, 2) if votes else None},
            "requests": {"total": req or 0, "cards": req_cards or 0},  # конверсия инсайта в действие
            "agent_runs": {"total": runs or 0, "passed_validation": ok or 0,
                           "pass_rate": round((ok or 0) / runs, 2) if runs else None,
                           "cost_rub": round(cost or 0, 2), "avg_ms": int(ms or 0)},
        }
