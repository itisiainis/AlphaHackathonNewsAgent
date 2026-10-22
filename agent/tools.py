"""Инструменты агента и реестр источников.

Безопасность (ограничение кейса 05) заложена в устройство, а не в промпт:
- инструменты только читают данные; записи, сети по произвольным адресам и выполнения кода нет;
- сессия инструментов привязана к одной компании: профиль и контрагенты другой компании агент
  получить не может, даже если модель попросит (id компании не передаётся аргументом);
- вызов инструмента не из белого списка отклоняется, попытка передать id другой компании — тоже;
  лишние безвредные аргументы отбрасываются (и пишутся в журнал), чтобы модель не тратила шаги на ошибки;
- каждый вызов пишется в журнал (имя, аргументы, размер результата, время) — журнал идёт в трассу.

Реестр источников: каждый найденный фрагмент новости или аналитики получает метку S1, S2, …
Модель ссылается на метки, а не пишет ссылки сама. При сборке карточки метки превращаются в
индексы sources[], а текст фрагментов используется валидатором для сверки чисел.

Пересмотренные прогнозы банка (agent/house_view.py): фрагмент со старым значением модель видит с пометкой
«[ПЕРЕСМОТРЕНО 11.09 → 14,0% (ставка на конец 2026)]» перед текстом. Если в выдаче есть и старый, и действующий
прогноз, решение принимает код — поле divergence: «источники расходятся: ставка на конец 2026 — 13,5% (31.07)
и 14,0% (11.09); берём более свежий»; эта строка идёт и в шаг трассы. Текст источника в реестре остаётся
дословным: по нему сверяются цифры и цитаты.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from . import house_view, impact, profiles
from .kb import BM25Index, Hit, industry_assessment


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
    revised: str = ""  # «[ПЕРЕСМОТРЕНО 11.09 → 14,0% (ставка на конец 2026)]» — прогноз во фрагменте пересмотрен


@dataclass
class ToolSession:
    company_id: str
    event: dict
    bank: BM25Index
    news: BM25Index
    today: date
    bank_meta: list[dict] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    calcs: list[dict] = field(default_factory=list)  # расчёты эффекта кодом (impact.compute)

    def __post_init__(self):
        p = profiles.get(self.company_id)
        # коды ОКВЭД компании: основной полный и классы её группы с долей от 10%
        self.okved_codes = [p["okved"]] + [g["okved2"] for g in p["group"]["industries"] if g["share"] >= 0.1]
        self.sectors = set(profiles.features(p)["own_sectors"])
        self.industry = industry_assessment(self.bank_meta, self.okved_codes, self.today)
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
        revised = house_view.revised_prefix(list(h.revisions)) if h.revisions else ""
        return self._register(Source("", name, h.kind, h.date or "", h.url, h.text, h.stale, h.title, h.page, revised))

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

    def profile_source(self) -> Source:
        """Данные профиля как источник вида stat: на него ссылаются входные данные расчётов."""
        return self._register(Source("", "Профиль компании: данные банка (выручка, продукты, платёжный поток)", "stat",
                                     self.today.isoformat(), "", impact.profile_text(profiles.get(self.company_id))))

    def calculate_impact(self) -> list[dict]:
        """Расчёт эффекта по формулам кода. Метки источников: S1 — новость, профиль — отдельная метка."""
        calcs = impact.compute(self.event, profiles.get(self.company_id), "S1", "PROFILE")
        if any(i["src"] == "PROFILE" for c in calcs for i in c["inputs"]):
            lbl = self.profile_source().label
            for c in calcs:
                for i in c["inputs"]:
                    if i["src"] == "PROFILE":
                        i["src"] = lbl
        return calcs

    def search_bank_analytics(self, query: str, industry: str = "auto", max_age_days: int = 365) -> dict:
        """industry: "auto" — поднимать фрагменты про отрасли компании; код ОКВЭД2 — про эту отрасль; "none" — без фильтра."""
        boost = self.sectors if industry == "auto" else (set() if industry == "none" else {industry[:2]})
        hits = self.bank.search(query, top_k=4, today=self.today, boost_sectors=boost)
        out, both = [], []
        for h in hits:
            s = self._hit_source(h)
            out.append({"label": s.label, "doc": h.title, "doc_kind": h.doc_kind, "date": h.date, "page": h.page,
                        "stale": h.stale, "too_old_for_conclusions": bool(h.date) and
                        (self.today - date.fromisoformat(h.date)).days > max_age_days,
                        "text": (s.revised + " " if s.revised else "") + h.text[:900]})
            # старое значение и действующий прогноз (тот же документ и страница) оба в выдаче — источники расходятся
            both += [r | {"date": h.date} for r in h.revisions
                     if any(x.doc_id == r["current"]["doc_id"] and r["current"].get("page") in (None, x.page) for x in hits)]
        res = {"industry_assessment": self.industry, "fragments": out}
        if both:
            res["divergence"] = house_view.divergence(both)
        return res

    @staticmethod
    def note(result) -> str:
        """Решение кода для шага трассы: « · источники расходятся: …; берём более свежий» или пустая строка."""
        return f" · {result['divergence']}" if isinstance(result, dict) and result.get("divergence") else ""

    def search_news(self, query: str, days: int = 30) -> list[dict]:
        out = []
        for h in self.news.search(query, top_k=4, max_age_days=days, today=self.today):
            s = self._hit_source(h)
            out.append({"label": s.label, "title": h.title, "date": h.date, "text": h.text[:600]})
        return out

    def get_event(self) -> dict:
        return {"label": "S1", "title": self.event.get("title"), "published": self.event.get("published"),
                "source": self.event.get("source"), "text": self.event.get("text", "")[:6000],
                "text_kind": {"full": "полный текст статьи", "summary": "только анонс источника — полный текст недоступен"}
                .get(self.event.get("text_kind", ""), "текст, переданный пользователем")}

    # ---------- диспетчер ----------
    TOOLS = {
        "get_event": {"args": {}, "about": "текст анализируемой новости (источник S1)"},
        "get_company_profile": {"args": {}, "about": "профиль выбранной компании: отрасль, масштаб, продукты банка, ВЭД, группа, контрагенты, качество данных"},
        "get_counterparties": {"args": {"side": "customers|suppliers"}, "about": "крупнейшие покупатели или поставщики и их отрасли"},
        "search_bank_analytics": {"args": {"query": "str", "industry": "auto | код ОКВЭД2 | none", "max_age_days": "int, по умолчанию 365"},
                                  "about": "поиск по аналитике Альфа-Банка: фрагменты с меткой источника, датой и типом документа "
                                           "+ есть ли у банка отраслевая оценка по отрасли компании; прогноз, который банк уже "
                                           "пересмотрел, помечен [ПЕРЕСМОТРЕНО …], а при расхождении источников поле divergence "
                                           "говорит, какой прогноз действует"},
        "search_news": {"args": {"query": "str", "days": "int, по умолчанию 30"}, "about": "поиск связанных новостей за период"},
        "calculate_impact": {"args": {}, "about": "расчёт эффекта в рублях по формулам (ставка → стоимость долга, курс → валютные закупки, "
                                                  "спрос/цены отрасли → выручка и закупки); пустой список — в новости нет числового шока"},
    }

    def call(self, name: str, args: dict | None) -> tuple[object, dict]:
        args = dict(args or {})
        t0 = time.monotonic()
        entry = {"tool": name, "args": dict(args)}  # как попросила модель — для журнала
        try:
            if name not in self.TOOLS:
                raise PermissionError(f"инструмент «{name}» не разрешён")
            for key in ("company_id", "id", "company"):
                if key in args and key not in self.TOOLS[name]["args"]:
                    if str(args.pop(key)) != self.company_id:
                        raise PermissionError("данные другой компании недоступны: сессия привязана к одной компании")
            extra = set(args) - set(self.TOOLS[name]["args"])
            for k in extra:  # лишний безвредный аргумент (например, дата у get_event) не повод тратить шаг агента
                args.pop(k)
            if extra:
                entry["ignored_args"] = sorted(extra)
            result = getattr(self, name)(**args)
            entry["ok"] = True
        except Exception as e:  # ошибка инструмента — наблюдение для агента, а не падение цикла
            result = {"error": str(e)}
            entry["ok"] = False
        entry["ms"] = int((time.monotonic() - t0) * 1000)
        entry["result_size"] = len(str(result))
        self.log.append(entry)
        return result, entry
