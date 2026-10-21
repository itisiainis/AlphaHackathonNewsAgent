"""Обучение сервиса на опыте: чем больше новостей, разборов и оценок клиентов, тем точнее отбор,
лучше выводы и быстрее ответ.

Модели по условиям кейса только разрешённые и доступны через API — дообучать их нельзя. Поэтому
сервис учится вокруг моделей. Каждое правило ниже выбрано по замеру на реальных данных (21 день новостей,
115 профилей, реальные замечания проверки), методика и цифры — docs/experiments/learning.md.

1. Память разборов. Тот же сюжет для той же компании за неделю — готовый разбор сразу, без модели.
   «Тот же сюжет» — agent/stories.py, проверено на независимой выборке размеченных пар (прежнее правило
   только по сходству слов ошибалось в 54 из 92 — отдавало вчерашний «курс ЦБ на 20 октября» на 21-е; новое — 0 из 28).
   Прошлые разборы той же компании агент видит как контекст; разборы других клиентов — никогда.
2. Оценки клиентов. «Полезно / Не про нас» сдвигают вес темы. Считается не сумма голосов, а насколько
   тема одобряется чаще, чем этот клиент одобряет в среднем («подъём», lift): тема, которая есть почти
   в каждой новости («отрасль»), не набирает случайный вес. И сдвиг есть, только когда отличие значимо
   (≥ 5 оценок темы, z ≥ 2): в симуляции на реальных лентах прежняя сумма голосов могла ухудшить ленту
   на 7,3 пункта NDCG, подъём без проверки значимости — на 0,9, с проверкой — не больше чем на 0,6.
   Выигрыш тоже небольшой (до 2 пунктов): это подстройка под клиента, который отличается от своей роли.
3. Рутина в потоке. Шаблон одного источника, выходящий в разные дни с другими датами и числами
   («Что произошло за день: среда, 14 октября»), опускается в отборе. Без ручных списков.
4. Уроки проверки. Повторяющиеся замечания проверки становятся правилами в задании агенту.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date

from .stories import jaccard, same_story, template_days, words

ROUTINE_FACTOR = 0.3        # во сколько раз опускается балл шаблонного сюжета
HISTORY_DAYS = 30           # сколько дней заголовков хранить для распознавания шаблонов
DUP_DAYS = 7                # разбор из памяти берём, если ему не больше недели
FEEDBACK_K = 3              # сглаживание: пока оценок мало, оценка темы тянется к средней
FEEDBACK_MAX = 3.0          # масштаб сдвига (веса тем у ролей — от 0 до 3); сдвиг = 2 × MAX × подъём
FEEDBACK_MIN_N = 5          # сдвиг — только по теме, оценённой хотя бы столько раз…
FEEDBACK_Z = 2.0            # …и если отличие от средней доли «полезно» значимо (z-критерий для доли)
MIN_LESSON = 2              # урок попадает в задание агенту, когда замечание повторилось

# Уроки проверки: (категория, шаблон замечания, правило для агента). Ищем только в словах проверки —
# цитаты из черновика («…», '…', [ … ]) вырезаются: иначе «ДКП» в процитированном тексте давало ложный урок.
LESSONS = [
    ("num", r"убрано: |не найдены в источнике|нет даты|нет в (?:этом )?источнике",
     "Цифры и даты — только из источников: если числа нет в тексте источника, не пиши его."),
    ("cite", r"ссылка S\d+→S\d+|ссылаю?тся на src|привязан[а-я]* к src|указан с src|в (?:этом )?источнике .{0,40}нет|содержится в S\d",
     "Метка источника — того, где факт написан дословно; если факт есть в нескольких источниках, выбери тот, где он сформулирован прямо."),
    ("fmt", r"is too (?:short|long)|is a required property|слишком (?:много|мало) элементов|нет обязательного поля|is not of type|не прошла проверку кодом:",
     "Держи формат ответа: 2–3 обоснования вывода (key_points) и все разделы разбора."),
    ("bank", r"позиция банка не совпала|bank-факт|bank\.quote|не подтверждается источником",
     "Позиция банка — только дословной цитатой из документа банка, с меткой этого документа."),
    ("prov", r"факт → позиция банка|прогноз банка: факт",
     "Прогноз или оценка банка — это «bank», а не «fact»."),
    ("jargon", r"код отрасли|ОКВЭД|жаргон|аббревиатур",
     "Без кодов отраслей и аббревиатур без расшифровки: «денежно-кредитная политика ЦБ», «автодилеры»."),
    ("retell", r"повторяет заголовок|пересказ",
     "Не повторяй заголовок: вывод объясняет, что событие значит для компании."),
]
_QUOTED = re.compile(r"«[^»]*»|'[^']*'|\[\{.*?\}\]", re.S)


def lesson_categories(finding: str) -> set[str]:
    text = _QUOTED.sub(" ", str(finding))
    return {cat for cat, rx, _ in LESSONS if re.search(rx, text, re.I)}


def _significant(up_down: list[int], p0: float) -> bool:
    """Отличается ли доля «полезно» по теме от средней p0 сильнее случайного (≈ 95%)."""
    n = sum(up_down)
    return n >= FEEDBACK_MIN_N and abs(up_down[0] / n - p0) >= FEEDBACK_Z * (p0 * (1 - p0) / n) ** 0.5


class Learner:
    """Учится на данных хранилища (backend/store.py). Без хранилища — ничему, и сервис работает как раньше."""

    def __init__(self, store):
        self.store = store
        self.enabled = True  # выключается в панели управления: сервис работает как без опыта, данные копятся
        self._template: dict[tuple[str, str], int] = {}

    # ---------- 3. поток новостей ----------
    def observe(self, events: list[dict]) -> int:
        """Запоминает заголовки (источник, день) и заранее считает шаблоны для текущих событий."""
        rows = [(e["source"], e["published"][:10], e["title"]) for e in events
                if e.get("title") and e.get("published") and e.get("source")]
        self.store.observe_titles(rows, keep_days=HISTORY_DAYS)
        by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for src, day, title in self.store.titles():
            by_source[src].append((day, title))
        self._template = {(s, t): template_days(t, by_source[s]) for s, _, t in rows}
        return len(rows)

    def routine_days(self, event: dict) -> int:
        """Сколько дней выходит этот шаблон (0 — не шаблон)."""
        if not self.enabled:
            return 0
        key = (event.get("source", ""), event.get("title", ""))
        if key not in self._template:
            history = [(d, t) for s, d, t in self.store.titles() if s == key[0]]
            self._template[key] = template_days(key[1], history)
        return self._template[key]

    def routine(self) -> list[dict]:
        """Узнанные шаблоны — для панели: пример заголовка, источник, сколько дней."""
        seen, out = set(), []
        for (src, title), n in sorted(self._template.items(), key=lambda x: -x[1]):
            sig = (src, " ".join(sorted(words(title))))
            if n and sig not in seen:
                seen.add(sig)
                out.append({"source": src, "example": title, "days": n})
        return out

    # ---------- 2. оценки клиентов ----------
    def topic_weights(self) -> dict:
        """{"roles": {роль: {тема: сдвиг}}, "companies": {компания: {роль: {тема: сдвиг}}}}.

        Сдвиг компании заменяет сдвиг роли (а не добавляется к нему): у компании, которая сама оценивала,
        её оценки уже включают общий опыт роли как априорное значение."""
        return self._weights() if self.enabled else {"roles": {}, "companies": {}}

    def _weights(self) -> dict:
        cards = {(c, card["id"]): card for c, card, _, _ in self.store.all_cards()}
        topic = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # (компания, роль) → тема → [за, против]
        total = defaultdict(lambda: [0, 0])                           # (компания, роль) → [за, против]
        for f in self.store.rows("SELECT company_id, card_id, role, vote FROM feedback"):
            card = cards.get((f["company_id"], f["card_id"]))
            if not card:
                continue
            k, key = (0 if f["vote"] == "up" else 1), (f["company_id"], f["role"])
            total[key][k] += 1
            for t in card.get("topics", []):
                topic[key][t][k] += 1
        roles, companies = {}, defaultdict(dict)
        K, M = FEEDBACK_K, FEEDBACK_MAX
        for role in {r for _, r in total}:
            keys = [k for k in total if k[1] == role]
            up = sum(total[k][0] for k in keys)
            p_role = (up + 1) / (sum(sum(total[k]) for k in keys) + 2)   # общая доля «полезно» у роли
            pooled = defaultdict(lambda: [0, 0])
            for k in keys:
                for t, v in topic[k].items():
                    pooled[t][0] += v[0]
                    pooled[t][1] += v[1]
            lift = {t: (v[0] + K * p_role) / (sum(v) + K) - p_role if _significant(v, p_role) else 0.0
                    for t, v in pooled.items()}
            roles[role] = {t: round(2 * M * x, 2) for t, x in lift.items()}
            for k in keys:
                p_own = (total[k][0] + 1) / (sum(total[k]) + 2)
                # своя оценка компании — только если её отличие значимо; иначе — общий опыт роли
                companies[k[0]][role] = {t: round(2 * M * ((v[0] + K * (p_own + lift[t])) / (sum(v) + K) - p_own
                                                           if _significant(v, p_own) else lift[t]), 2)
                                         for t, v in topic[k].items()}
        return {"roles": roles, "companies": dict(companies)}

    # ---------- 1. память разборов ----------
    def remembered(self, company_id: str, event: dict, today: date) -> dict | None:
        """Разбор того же сюжета для этой компании за последнюю неделю — отдаём сразу, без нового запуска модели."""
        if not self.enabled:
            return None
        hidden = self.store.hidden(company_id)
        for c, card, _origin, _created in self.store.all_cards("agent"):
            if c != company_id or (c, card["id"]) in hidden:  # скрытый сотрудником банка разбор не раздаём
                continue
            published = card.get("published", "1970-01-01")
            if (today - date.fromisoformat(published[:10])).days > DUP_DAYS:
                continue
            src = next((s["name"] for s in card.get("sources", []) if s.get("kind") == "news"), None)
            if same_story(event.get("title", ""), card.get("title", ""), source_a=event.get("source"), source_b=src,
                          day_a=event.get("published"), day_b=published):
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
            if score >= 1 and not same_story(event.get("title", ""), card.get("title", "")):
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
        cnt = Counter(cat for t in texts for cat in lesson_categories(t))
        text = {cat: lesson for cat, _, lesson in LESSONS}
        return [{"lesson": text[c], "category": c, "seen": n, "active": n >= MIN_LESSON} for c, n in cnt.most_common()]

    def active_lessons(self) -> list[str]:
        if not self.enabled:
            return []
        return [x["lesson"] for x in self.lessons() if x["active"]][:5]

    # ---------- для панели управления ----------
    def summary(self) -> dict:
        trend = self.store.rows("""SELECT date(ts, 'unixepoch') AS day, COUNT(*) AS runs, AVG(ok) AS pass_rate,
                                          AVG(ms) AS avg_ms, AVG(revisions) AS revisions, SUM(cost) AS cost
                                   FROM runs GROUP BY day ORDER BY day DESC LIMIT 14""")
        votes = self.store.rows("SELECT COUNT(*) AS n, SUM(vote='up') AS up FROM feedback")[0]
        seen = self.store.rows("SELECT COUNT(*) AS n, COUNT(DISTINCT day) AS days FROM story_titles")[0]
        return {"trend": trend[::-1], "reused": self.store.rows("SELECT COUNT(*) AS n FROM reuse")[0]["n"],
                "stories": seen["n"], "days_observed": seen["days"], "routine": self.routine()[:10],
                "weights": self._weights(), "votes": votes["n"],
                "useful_share": round(votes["up"] / votes["n"], 2) if votes["n"] else None,
                "lessons": self.lessons(), "memory": len(self.store.all_cards("agent")), "enabled": self.enabled}

