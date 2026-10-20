"""Инструменты агента и реестр источников.

Безопасность (ограничение кейса 05) заложена в устройство, а не в промпт:
- инструменты только читают данные; записи, сети по произвольным адресам и выполнения кода нет;
- сессия инструментов привязана к одной компании: профиль и контрагенты другой компании агент
  получить не может, даже если модель попросит (id компании не передаётся аргументом);
- вызов инструмента не из белого списка или с неизвестными аргументами отклоняется;
- каждый вызов пишется в журнал (имя, аргументы, размер результата, время) — журнал идёт в трассу.

Реестр источников: каждый найденный фрагмент новости или аналитики получает метку S1, S2, …
Модель ссылается на метки, а не пишет ссылки сама. При сборке карточки метки превращаются в
индексы sources[], а текст фрагментов используется валидатором для сверки чисел.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from . import profiles
from .kb import BM25Index, Hit


@dataclass
class Source:
    label: str
    name: str
    kind: str       # news | bank | stat
    date: str
    url: str
    text: str
    stale: bool = False
    doc: str = ""   # полное название документа банка
    page: int | None = None


@dataclass
class ToolSession:
    company_id: str
    event: dict
    bank: BM25Index
    news: BM25Index
    today: date
    sources: list[Source] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        ev = self.event
        self._register(Source("", ev.get("source", "Новость"), ev.get("kind", "news"), ev.get("published", ""),
                              ev.get("url", ""), (ev.get("title", "") + "\n" + ev.get("text", "")).strip()))

    # ---------- реестр ----------
    def _register(self, s: Source) -> Source:
        for old in self.sources:
            if old.url == s.url and old.text == s.text:
                return old
        s.label = f"S{len(self.sources) + 1}"
        self.sources.append(s)
        return s

    def _hit_source(self, h: Hit) -> Source:
        where = f", стр. {h.page}" if h.page else ""
        name = f"Альфа-Банк: «{h.title[:80]}»{where}" if h.kind == "bank" else h.title
        return self._register(Source("", name, h.kind, h.date or "", h.url, h.text, h.stale, h.title, h.page))

    # ---------- инструменты ----------
    def get_company_profile(self) -> dict:
        p = profiles.get(self.company_id)
        v = profiles.view(p)
        return {k: v[k] for k in ("id", "industry", "okved", "revenue_bn", "region", "employees", "products",
                                  "ved", "group_size", "counterparties", "data_quality")} | (
            {"group_industries": v["group_industries"]} if "group_industries" in v else {})

    def get_counterparties(self, side: str = "customers") -> dict:
        if side not in ("customers", "suppliers"):
            raise ValueError("side: customers | suppliers")
        s = profiles.get(self.company_id)[side]
        return {"n": s["n"], "total_rub": s["total_rub"], "known_industry_share": s["known_industry_share"],
                "intragroup_share": s.get("intragroup_share") or 0,
                "top": [{"amount_rub": t["amount_rub"], "industry": t["industry"], "intragroup": t["intragroup"]}
                        for t in s["top"][:10]]}

    def search_bank_analytics(self, query: str, max_age_days: int = 365) -> list[dict]:
        hits = self.bank.search(query, top_k=4, today=self.today)
        out = []
        for h in hits:
            s = self._hit_source(h)
            out.append({"label": s.label, "doc": h.title, "date": h.date, "page": h.page, "stale": h.stale,
                        "too_old_for_conclusions": bool(h.date) and
                        (self.today - date.fromisoformat(h.date)).days > max_age_days,
                        "text": h.text[:900]})
        return out

    def search_news(self, query: str, days: int = 30) -> list[dict]:
        out = []
        for h in self.news.search(query, top_k=4, max_age_days=days, today=self.today):
            s = self._hit_source(h)
            out.append({"label": s.label, "title": h.title, "date": h.date, "text": h.text[:600]})
        return out

    def get_event(self) -> dict:
        return {"label": "S1", "title": self.event.get("title"), "published": self.event.get("published"),
                "source": self.event.get("source"), "text": self.event.get("text", "")[:4000]}

    # ---------- диспетчер ----------
    TOOLS = {
        "get_event": {"args": {}, "about": "текст анализируемой новости (источник S1)"},
        "get_company_profile": {"args": {}, "about": "профиль выбранной компании: отрасль, масштаб, продукты банка, ВЭД, группа, контрагенты, качество данных"},
        "get_counterparties": {"args": {"side": "customers|suppliers"}, "about": "крупнейшие покупатели или поставщики и их отрасли"},
        "search_bank_analytics": {"args": {"query": "str", "max_age_days": "int, по умолчанию 365"},
                                  "about": "поиск по аналитике Альфа-Банка; у каждого фрагмента метка источника и дата"},
        "search_news": {"args": {"query": "str", "days": "int, по умолчанию 30"}, "about": "поиск связанных новостей за период"},
    }

    def call(self, name: str, args: dict | None) -> tuple[object, dict]:
        args = dict(args or {})
        t0 = time.monotonic()
        entry = {"tool": name, "args": args}
        try:
            if name not in self.TOOLS:
                raise PermissionError(f"инструмент «{name}» не разрешён")
            extra = set(args) - set(self.TOOLS[name]["args"])
            if extra:
                raise ValueError(f"неизвестные аргументы: {sorted(extra)}")
            result = getattr(self, name)(**args)
            entry["ok"] = True
        except Exception as e:  # ошибка инструмента — наблюдение для агента, а не падение цикла
            result = {"error": str(e)}
            entry["ok"] = False
        entry["ms"] = int((time.monotonic() - t0) * 1000)
        entry["result_size"] = len(str(result))
        self.log.append(entry)
        return result, entry
