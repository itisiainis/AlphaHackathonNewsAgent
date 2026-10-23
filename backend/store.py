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
-- панель управления банка: настройки, модерация карточек, журнал действий сотрудников
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS moderation(company_id TEXT, card_id TEXT, status TEXT, note TEXT, ts REAL,
                                      PRIMARY KEY(company_id, card_id));
CREATE TABLE IF NOT EXISTS audit(ts REAL, actor TEXT, action TEXT, detail TEXT);
-- обучение на опыте: заголовки по источникам и дням (распознавание шаблонов) и ответы из памяти разборов
CREATE TABLE IF NOT EXISTS story_titles(source TEXT, day TEXT, title TEXT, PRIMARY KEY(source, day, title));
CREATE TABLE IF NOT EXISTS reuse(ts REAL, company_id TEXT, card_id TEXT);
-- новости, которые клиент прислал сам (ссылка или текст): нужны для полного разбора позже и чтобы отличать их в ленте
CREATE TABLE IF NOT EXISTS user_events(company_id TEXT, event_id TEXT, json TEXT, ts REAL, PRIMARY KEY(company_id, event_id));
"""


class _Rows:
    """Результат запроса, прочитанный целиком под блокировкой (вместо живого курсора)."""

    def __init__(self, rows: list, description):
        self._rows, self.description = rows, description

    def fetchall(self) -> list:
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        try:  # статус запроса менеджеру добавлен позже — старые базы дополняем
            self.db.execute("ALTER TABLE requests ADD COLUMN status TEXT DEFAULT 'new'")
        except sqlite3.OperationalError:
            pass
        self.lock = threading.Lock()

    def _exec(self, sql, args=()) -> _Rows:
        """Запрос и чтение результата — под одной блокировкой. Соединение общее для потоков: если читать строки
        после снятия блокировки, параллельный запрос перезапишет курсор (замер: 21 ответ 500 из 90 при 12 потоках)."""
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = _Rows(cur.fetchall() if cur.description else [], cur.description)
            self.db.commit()
            return rows

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

    def card_meta(self, company_id: str, card_id: str) -> tuple[str, float] | None:
        """(происхождение, когда сохранена) — или None, если карточки нет."""
        row = self._exec("SELECT origin, created FROM cards WHERE company_id=? AND card_id=?", (company_id, card_id)).fetchone()
        return (row[0], row[1]) if row else None

    def drop_card(self, company_id: str, card_id: str) -> None:
        self._exec("DELETE FROM cards WHERE company_id=? AND card_id=?", (company_id, card_id))

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

    def recent_run_ms(self, n: int = 20) -> int | None:
        """Среднее время последних разборов агентом — по нему интерфейс оценивает, сколько осталось ждать."""
        row = self._exec("SELECT AVG(ms) FROM (SELECT ms FROM runs WHERE ms > 0 ORDER BY ts DESC LIMIT ?)", (n,)).fetchone()
        return int(row[0]) if row and row[0] else None

    def put_request(self, company_id: str, card_id: str, role: str, kind: str, label: str, comment: str) -> None:
        """Запрос клиента менеджеру из карточки: в рабочей версии уходит в CRM банка, здесь — в журнал."""
        self._exec("INSERT INTO requests(ts, company_id, card_id, role, kind, label, comment) VALUES(?,?,?,?,?,?,?)",
                   (time.time(), company_id, card_id, role, kind, label, comment))

    # ---------- панель управления ----------
    def rows(self, sql: str, args=()) -> list[dict]:
        cur = self._exec(sql, args)
        names = [c[0] for c in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def settings(self) -> dict:
        return {k: json.loads(v) for k, v in self._exec("SELECT key, value FROM settings").fetchall()}

    def set_setting(self, key: str, value) -> None:
        self._exec("INSERT OR REPLACE INTO settings VALUES(?,?,?)", (key, json.dumps(value, ensure_ascii=False), time.time()))

    def audit(self, actor: str, action: str, detail: str = "") -> None:
        self._exec("INSERT INTO audit VALUES(?,?,?,?)", (time.time(), actor, action, detail))

    def moderate(self, company_id: str, card_id: str, hidden: bool, note: str = "") -> None:
        if hidden:
            self._exec("INSERT OR REPLACE INTO moderation VALUES(?,?,?,?,?)", (company_id, card_id, "hidden", note, time.time()))
        else:
            self._exec("DELETE FROM moderation WHERE company_id=? AND card_id=?", (company_id, card_id))

    def hidden(self, company_id: str | None = None) -> set[tuple[str, str]]:
        rows = self._exec("SELECT company_id, card_id FROM moderation WHERE status='hidden'").fetchall()
        return {(c, k) for c, k in rows if company_id is None or c == company_id}

    def set_request_status(self, rowid: int, status: str) -> None:
        self._exec("UPDATE requests SET status=? WHERE rowid=?", (status, rowid))

    def all_cards(self, origin: str | None = None) -> list[tuple[str, dict, str, float]]:
        rows = self._exec("SELECT company_id, json, origin, created FROM cards WHERE (? IS NULL OR origin=?) ORDER BY created DESC",
                          (origin, origin)).fetchall()
        return [(c, json.loads(j), o, t) for c, j, o, t in rows]

    # ---------- обучение ----------
    def observe_titles(self, rows: list[tuple[str, str, str]], keep_days: int = 30) -> None:
        """(источник, день, заголовок); старше keep_days дней от самого свежего — удаляются."""
        with self.lock:
            self.db.executemany("INSERT OR IGNORE INTO story_titles VALUES(?,?,?)", rows)
            self.db.execute("DELETE FROM story_titles WHERE day < date((SELECT MAX(day) FROM story_titles), ?)",
                            (f"-{keep_days} days",))
            self.db.commit()

    def titles(self) -> list[tuple[str, str, str]]:
        return self._exec("SELECT source, day, title FROM story_titles").fetchall()

    def put_user_event(self, company_id: str, event: dict) -> None:
        self._exec("INSERT OR REPLACE INTO user_events VALUES(?,?,?,?)",
                   (company_id, event["id"], json.dumps(event, ensure_ascii=False), time.time()))

    def user_event(self, company_id: str, event_id: str) -> dict | None:
        row = self._exec("SELECT json FROM user_events WHERE company_id=? AND event_id=?", (company_id, event_id)).fetchone()
        return json.loads(row[0]) if row else None

    def user_card_ids(self, company_id: str) -> list[str]:
        """Карточки по новостям, которые клиент прислал сам, — свежие первыми."""
        rows = self._exec("SELECT event_id FROM user_events WHERE company_id=? ORDER BY ts DESC", (company_id,)).fetchall()
        return [f"{e}-{company_id}" for (e,) in rows]

    def log_reuse(self, company_id: str, card_id: str) -> None:
        self._exec("INSERT INTO reuse VALUES(?,?,?)", (time.time(), company_id, card_id))

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
