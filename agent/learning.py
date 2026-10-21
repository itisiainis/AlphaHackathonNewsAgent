"""Обучение сервиса на опыте: чем больше новостей, разборов и оценок клиентов, тем точнее отбор,
лучше выводы и быстрее ответ.

Модели по условиям кейса только разрешённые и доступны через API — дообучать их нельзя. Поэтому
сервис учится вокруг моделей, и каждый вид обучения измерим и виден в панели управления банка:

1. Память разборов. Новость, которая повторяет уже разобранный сюжет (другое издание, тот же
   заголовок), получает готовый разбор мгновенно и без расхода. Прошлые разборы той же компании
   агент видит как контекст: «в сентябре мы писали …, что изменилось». Чужие разборы в контекст не
   попадают — в них данные другого клиента.
2. Оценки клиентов. «Полезно / Не про нас» по карточкам двигает веса тем для роли (и для конкретной
   компании): финансовый директор, который отмечает новости о спросе «не про нас», видит их ниже.
3. Поток новостей. Сюжет, который выходит день за днём одинаковым (официальный курс на завтра,
   ежедневная сводка), сервис распознаёт сам — по истории, без ручных правил — и опускает.
4. Уроки проверки. Частые замечания проверки кодом и проверяющей моделью (цифра не из источника,
   жаргон, повтор заголовка) становятся «уроками» в задании агенту: меньше правок — быстрее и дешевле.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime

ROUTINE_DAYS = 3            # сюжет, встреченный в столько разных дней, — рутина
ROUTINE_FACTOR = 0.3
DUP_JACCARD = 0.6           # заголовки настолько похожи — это тот же сюжет
DUP_DAYS = 7                # разбор из памяти берём, если ему не больше недели
FEEDBACK_K = 3              # сглаживание: пока оценок мало, веса почти не двигаются
FEEDBACK_MAX = 3.0          # предел сдвига веса темы (веса ролей — от 0 до 3)
MIN_LESSON = 2              # урок попадает в задание агенту, когда замечание повторилось

MONTHS = r"январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр"
STOP = {"это", "как", "что", "для", "при", "его", "она", "они", "года", "году", "все", "уже", "или"}


def words(title: str) -> set[str]:
    """Основы слов заголовка без чисел, дат и служебных слов — для сравнения сюжетов."""
    t = re.sub(r"\d+|" + MONTHS, " ", title.lower().replace("ё", "е"))
    return {w[:6] for w in re.findall(r"[а-яa-z]{3,}", t) if w not in STOP}


def signature(title: str) -> str:
    """Подпись сюжета: одинакова у «ЦБ установил официальные курсы на 21 октября» и «… на 22 октября»."""
    return " ".join(sorted(words(title)))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _day(ts_or_iso) -> str:
    return ts_or_iso[:10] if isinstance(ts_or_iso, str) else datetime.fromtimestamp(ts_or_iso).date().isoformat()


LESSONS = [
    (r"не найдены в источнике|убрано: «", "Цифры — только из источников: если числа нет в тексте источника, не пиши его."),
    (r"код отрасли|ОКВЭД|оквэд", "Без кодов отраслей: называй отрасль по тем, кто в ней работает («автодилеры»)."),
    (r"ДКП|аббревиатур", "Расшифровывай сокращения: «денежно-кредитная политика ЦБ», а не «ДКП»."),
    (r"контрагент|входящий поток|поток»", "Пиши «покупатели», «поставщики», «поступления», «платежи», а не «контрагенты» и «поток»."),
    (r"повторяет заголовок|пересказ", "Не повторяй заголовок: вывод объясняет, что событие значит для компании."),
    (r"позиция банка не совпала|ссылается на источник вида", "Позиция банка — только дословной цитатой из документа банка."),
    (r"без источника", "У каждого факта и позиции банка — метка источника."),
    (r"не прошла проверку кодом", "Держи формат ответа: 2–3 обоснования вывода (key_points) и все разделы разбора."),
]


class Learner:
    """Учится на данных хранилища (backend/store.py). Без хранилища — ничему, и сервис работает как раньше."""

    def __init__(self, store):
        self.store = store
        self.enabled = True  # выключается в панели управления: сервис работает как без опыта, данные копятся

    # ---------- 3. поток новостей ----------
    def observe(self, events: list[dict]) -> int:
        pairs = [(signature(e["title"]), e["published"][:10]) for e in events if e.get("title") and e.get("published")]
        self.store.observe_story([p for p in pairs if p[0]])
        return len(pairs)

    def routine(self) -> dict[str, int]:
        return self.store.story_counts(ROUTINE_DAYS) if self.enabled else {}

    def is_routine(self, title: str, routine: dict | None = None) -> bool:
        return signature(title) in (self.routine() if routine is None else routine)

    # ---------- 2. оценки клиентов ----------
    def topic_weights(self) -> dict:
        """{"roles": {роль: {тема: сдвиг}}, "companies": {компания: {роль: {тема: сдвиг}}}} — по оценкам «полезно / не про нас»."""
        return self._weights() if self.enabled else {"roles": {}, "companies": {}}

    def _weights(self) -> dict:
        cards = {(c, card["id"]): card for c, card, _, _ in self.store.all_cards()}
        role_cnt: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        comp_cnt: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
        for f in self.store.rows("SELECT company_id, card_id, role, vote FROM feedback"):
            card = cards.get((f["company_id"], f["card_id"]))
            if not card:
                continue
            k = 0 if f["vote"] == "up" else 1
            for t in card.get("topics", []):
                role_cnt[f["role"]][t][k] += 1
                comp_cnt[f["company_id"]][f["role"]][t][k] += 1

        def delta(up_down):
            up, down = up_down
            return round(FEEDBACK_MAX * (up - down) / (up + down + FEEDBACK_K), 2)

        return {"roles": {r: {t: delta(v) for t, v in ts.items()} for r, ts in role_cnt.items()},
                "companies": {c: {r: {t: delta(v) for t, v in ts.items()} for r, ts in rs.items()} for c, rs in comp_cnt.items()}}

    # ---------- 1. память разборов ----------
    def remembered(self, company_id: str, title: str, today: date) -> dict | None:
        """Разбор того же сюжета для этой компании за последнюю неделю — отдаём сразу, без нового запуска модели."""
        if not self.enabled:
            return None
        w, hidden = words(title), self.store.hidden(company_id)
        for c, card, _origin, _created in self.store.all_cards("agent"):
            if c != company_id or (c, card["id"]) in hidden:  # скрытый сотрудником банка разбор не раздаём
                continue
            fresh = (today - date.fromisoformat(card.get("published", "1970-01-01"))).days <= DUP_DAYS
            if fresh and jaccard(w, words(card.get("title", ""))) >= DUP_JACCARD:
                return card
        return None

    def precedents(self, company_id: str, event: dict, k: int = 2) -> list[dict]:
        """Прошлые разборы этой же компании по той же теме — контекст «что изменилось». Только своя компания."""
        if not self.enabled:
            return []
        topics, w = set(event.get("topics", [])), words(event.get("title", ""))
        out, hidden = [], self.store.hidden(company_id)
        for c, card, origin, _ in self.store.all_cards():
            if c != company_id or origin == "rules" or (c, card["id"]) in hidden or card["id"].startswith(event.get("id", "\0")):
                continue
            score = len(topics & set(card.get("topics", []))) + 2 * jaccard(w, words(card.get("title", "")))
            if score >= 1:
                out.append((score, {"date": card.get("published"), "title": card.get("title"),
                                    "takeaway": card.get("takeaway", {}).get("cfo") or card.get("why_short", "")}))
        return [p for _, p in sorted(out, key=lambda x: -x[0])[:k]]

    # ---------- 4. уроки проверки ----------
    def lessons(self) -> list[dict]:
        texts = []
        for r in self.store.rows("SELECT errors FROM runs"):
            texts += json.loads(r["errors"] or "[]")
        for _c, card, _origin, _t in self.store.all_cards("agent"):
            texts += [t["detail"] for t in card.get("trace", []) if t.get("kind") == "check"]
        cnt = Counter()
        for t in texts:
            for rx, lesson in LESSONS:
                if re.search(rx, str(t), re.I):
                    cnt[lesson] += 1
        return [{"lesson": lesson, "seen": n, "active": n >= MIN_LESSON} for lesson, n in cnt.most_common()]

    def active_lessons(self) -> list[str]:
        if not self.enabled:
            return []
        return [x["lesson"] for x in self.lessons() if x["active"]][:5]

    # ---------- для панели управления ----------
    def summary(self) -> dict:
        trend = self.store.rows("""SELECT date(ts, 'unixepoch') AS day, COUNT(*) AS runs, AVG(ok) AS pass_rate,
                                          AVG(ms) AS avg_ms, AVG(revisions) AS revisions, SUM(cost) AS cost
                                   FROM runs GROUP BY day ORDER BY day DESC LIMIT 14""")
        reuse = self.store.rows("SELECT COUNT(*) AS n FROM reuse")[0]["n"]
        stories = self.store.rows("SELECT COUNT(DISTINCT sig) AS n, COUNT(DISTINCT day) AS days FROM story_days")[0]
        routine = sorted(self.store.story_counts(ROUTINE_DAYS).items(), key=lambda x: -x[1])[:10]
        return {"trend": trend[::-1], "reused": reuse, "stories": stories["n"], "days_observed": stories["days"],
                "routine": [{"story": s, "days": d} for s, d in routine], "weights": self._weights(),
                "lessons": self.lessons(), "memory": len(self.store.all_cards("agent")), "enabled": self.enabled}
