"""Черновой разбор правилами, когда модель недоступна (нет ключа, исчерпан бюджет, сбой провайдера).

Зачем: сервис должен оставаться полезным и честным без LLM — например, когда проект запускают
без ключа (ключ команды публиковать нельзя) или общий дневной бюджет закончился.
Карточка собирается теми же инструментами, что и у агента, но без модели:
- что произошло — информативные предложения новости (факт, источник S1);
- почему важно — признаки компании, на которые сработал отбор (факт из профиля, источник «статистика»);
- позиция банка — фрагмент аналитики, только свежий и только если совпадает с темой события;
- сколько это в деньгах — расчёт кодом (agent/impact.py);
- что это значит для вас и что сделать — каналы влияния (agent/channels.py): главный вывод под роль, риски,
  возможности и действия. Это выводы по правилам (происхождение «rule»), а не выводы ИИ: каждый собран кодом из
  новости, профиля по данным банка, расчёта и взгляда банка. Карточка помечается как краткий разбор, и полный
  разбор агентом её заменит, когда модель станет доступна.
"""
from __future__ import annotations

import re
import time
from datetime import date

from . import channels, house_view, profiles
from .contract import validate_card
from .impact import _sentences
from .ingest.bank_docs import strip_artefacts
from .kb import BM25Index
from .matching import Match
from .okved import plain, topics_in, who
from .tools import Source, ToolSession

TOPICS = {"ставка", "долг", "курс", "ликвидность", "отрасль", "спрос", "контрагенты", "налоги"}
NOTICE = "Краткий разбор: факты новости, данные компании, позиция банка, расчёт и выводы по правилам — без выводов ИИ."
UNAVAILABLE = "ИИ-аналитик сейчас недоступен. " + NOTICE
# Честная плашка, когда дословной позиции банка нет — одна и та же в кратком разборе и у агента (agent/loop.py)
NO_BANK_VIEW = "Свежей позиции банка по этому событию нет"
CHANNELS_STEP = "Каналы влияния (код)"  # шаг трассы; по нему сервер узнаёт разборы, собранные до каналов влияния


# частые слова, совпадение по которым ничего не говорит о теме
GENERIC = {"росси", "банка", "банко", "сохра", "рынка", "рынке", "компа", "сообщ", "заяви", "новос",
           "эконо", "прогн", "ожида", "данны", "октяб", "сентя", "недел", "месяц", "будет", "также", "после",
           "может", "могут", "хочет", "более", "менее", "около", "всего", "этого", "через", "между"}
# Позиция банка по теме события: что добавить к запросу в аналитику и к словам, по которым выбирается предложение.
# По ставке взгляд банка — в обзоре решения ЦБ, а заголовок новости («…пресс-конференция…») его не находит
TOPIC_HINT = {"ставка": "ключевая ставка прогноз ожидаем"}


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _by_kind(match: Match) -> dict[str, list[dict]]:
    """Причины отбора по видам признака (kind из agent/matching.py); понижающие пометки — не причина."""
    by: dict[str, list[dict]] = {}
    for r in match.reasons:
        if r["weight"] > 1 and r.get("kind"):
            by.setdefault(r["kind"], []).append(r)
    return by


# отрасль события — своя, отрасль товара или отрасль группы (role из agent/matching.py)
OWN_HEAD = {"own": "Событие в вашей отрасли ({}) — касается вас напрямую",
            "goods": "Событие в отрасли вашего товара ({}) — может изменить ваши продажи и цены",
            "group": "Событие в отрасли вашей группы ({}) — касается компаний группы"}
OWN_WHY = {"own": "Это ваша отрасль — {}", "goods": "Вы продаёте товар этой отрасли — {}",
           "group": "В этой отрасли работают компании вашей группы — {}"}


def _pct(r: dict) -> str:
    return f"{round(r['share'] * 100)}%"


def headline(match: Match, order: tuple[str, ...]) -> str:
    """Главный вывод без модели простой причинной фразой: что в бизнесе компании затрагивает событие и почему."""
    by = _by_kind(match)
    for k in order:
        if k not in by:
            continue
        r = by[k][0]
        if k == "debt":
            return f"Это меняет стоимость вашего долга: {' и '.join(x['feature'] for x in by[k])} зависят от ставки"
        if k == "deposits":
            return "Это меняет доход по вашим депозитам: он следует за ставкой"
        if k == "fx":
            return f"Это меняет стоимость ваших расчётов с зарубежными партнёрами — {r['feature']}"
        if k == "customers":
            return f"{_cap(who(r['sector']))} — ваши покупатели ({_pct(r)} поступлений): что меняется у них, скажется на ваших продажах"
        if k == "suppliers":
            return f"{_cap(who(r['sector']))} — ваши поставщики ({_pct(r)} платежей): что меняется у них, скажется на ваших закупках"
        if k == "own":
            return OWN_HEAD[r.get("role", "own")].format(plain(r["sector"]))
        if k == "demand":
            return "Это меняет спрос ваших покупателей"
    return f"Касается вас: {match.reasons[0]['feature']}" if match.reasons else "Прямой связи с профилем компании не найдено"


def _why(by: dict[str, list[dict]]) -> list[str]:
    """«Почему это важно вам» без модели: признаки компании из профиля, по одной фразе на вид признака."""
    out = []
    for k, rs in by.items():
        if k == "debt":
            out.append(f"У вас в банке {' и '.join(x['feature'] for x in rs)}: платежи по ним зависят от ставки")
        elif k == "customers":
            out.extend(f"{_pct(x)} ваших поступлений через банк — от покупателей: {who(x['sector'])}. "
                       "Что меняется у них, скажется на ваших продажах" for x in rs)
        elif k == "suppliers":
            out.extend(f"{_pct(x)} ваших платежей через банк — поставщикам: {who(x['sector'])}. "
                       "Что меняется у них, скажется на ваших закупках" for x in rs)
        elif k == "own":
            out.extend(OWN_WHY[x.get("role", "own")].format(plain(x["sector"])) for x in rs)
        else:
            out.extend(f"{_cap(x['feature'])}: {x['why']}" for x in rs)
    return out


def _take(c: dict) -> str:
    """Главный вывод из расчёта кодом: сначала деньги, потом почему."""
    title, value = c["title"], c["value"]
    if title.startswith("Выручка под влиянием") and " — " in title:
        return f"{_cap(title.split(' — ', 1)[1])} — ваши покупатели: от них {value.replace(' в год', '')} выручки в год"
    if title.startswith("Выручка от покупателей") and " — " in title:
        return f"{_cap(title.split(' — ', 1)[1])} — ваши покупатели: выручка от них изменится на {value}"
    if title.startswith("Выручка:"):
        return f"Ваша выручка может измениться на {value}"
    return f"{title}: {value}"


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яё]{5,}", text.lower())}  # 5 букв: «ставку» и «ставки» — одна основа


def _repeats(s: str, title: str) -> bool:
    """Предложение пересказывает заголовок: почти все его основы слов уже есть в заголовке."""
    st = _stems(s)
    return bool(st) and len(st & _stems(title)) / len(st) >= 0.7


def _facts(event: dict) -> list[str]:
    """2–3 информативных предложения новости: с числами или достаточно длинные, без повторов заголовка."""
    title = event.get("title", "").strip()
    out = []
    for s in _sentences(event.get("text", "")):
        if len(s) < 40 or s.strip(" .") == title.strip(" .") or _repeats(s, title) or len(s) > 400:
            continue
        if re.search(r"\d", s) or len(out) < 1:
            out.append(s)
        if len(out) == 3:
            break
    return out or [title]


# ---------- позиция банка: только дословно и только там, где банк говорит от себя ----------
# Слова берутся целиком: «ЦБ ожидаемо сохранил ставку» — пересказ решения ЦБ, а не взгляд банка
POSITION_RE = re.compile(r"(?<![а-яё])(?:ожидаем|считаем|полагаем|жд[её]м|прогнозируем|ориентируемся|допускаем)(?![а-яё])"
                         r"|(?<![а-яё])(?:наш(?:\s+базовый)?\s+прогноз|на\s+наш\s+взгляд|по\s+нашим\s+оценкам)(?![а-яё])", re.I)
RETRO_RE = re.compile(r"^ранее мы (?:писали|отмечали|говорили)", re.I)  # пересказ прошлой публикации, а не взгляд
NESTED_RE = re.compile(r"«[^«»]*»")  # название другой публикации: «Ждем паузу на 14,0%…» — не позиция этого текста
ONE_LETTER = set("авиксоуяг")        # однобуквенные слова и «г.» (год) — не мусор распознавания
NOISE_SHARE, QUOTE_MIN, QUOTE_MAX = 0.15, 40, 400
# Заголовок, сросшийся с текстом в одной строке: строчное слово без знака препинания, за ним слово-заголовок
# с заглавной («…сохраняется Диверсификация резервов…»). Имена собственные (Минфин, России) этих окончаний не имеют
FUSED_RE = re.compile(r"[а-яё]{3,}\s+[А-ЯЁ][а-яё]*(?:ация|яция|ение|ание|ость|ика)(?![а-яё])")
HEADER_RE = re.compile(r"^центр макроэкономического анализа", re.I)
# Одиночная буква в начале — остаток буквицы или значка списка после OCR («Я Мы понижаем…», «О Ожидаем…»).
# Однобуквенный предлог («В России…») не трогаем, если за ним не слово, с которого банк начинает свою позицию
OCR_LEAD = re.compile(r"^(?:[А-ЯЁA-Z](?<![ВСКОУИА])|[ВСКОУИА](?= (?:Мы|Ожидаем|Ждем|Ждём|Считаем|Полагаем)(?![а-яё]))) (?=[А-ЯЁ])")
SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z«\"(\d])")
# Решение ЦБ в событии и в тексте банка: направление события и уже сделанный шаг (прошедшее время)
CB_RE = re.compile(r"(?<![А-Яа-яЁё])ЦБ(?![а-яё])|Банк\w* России|регулятор")
RATE_RE = re.compile(r"ставк|пауз", re.I)
EVENT_DIR = {"hold": re.compile(r"сохран|остав|без изменени|пауз|не (?:стал|будет) менять", re.I),
             "cut": re.compile(r"сниз|пониз|снижени", re.I),
             "hike": re.compile(r"повыс|повыш|подня", re.I)}
# что ЦБ уже сделал (прошедшее время) и чего банк ждёт от ставки
DONE = {"hold": re.compile(r"(?:сохранил|оставил)[аио]?(?![а-яё])|взял[аио]? паузу|без изменени|решение (?:оставить|сохранить)", re.I),
        "cut": re.compile(r"(?:снизил|понизил)[аио]?(?![а-яё])|решение (?:снизить|понизить)", re.I),
        "hike": re.compile(r"(?:повысил|поднял)[аио]?(?![а-яё])|решение (?:повысить|поднять)", re.I)}
EXPECT = {"hold": re.compile(r"сохранени\w* (?:паузы|ставки|ключевой|ее|её|на текущем)|пауз|без изменени|неизменн", re.I),
          "cut": re.compile(r"снижени\w* (?:ключевой )?ставк|(?:снижени|смягчени)\w* (?:по|на) \d", re.I),
          "hike": re.compile(r"повышени\w* (?:ключевой )?ставк|ужесточени", re.I)}
MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря")
DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s+(" + "|".join(MONTHS_GEN) + r")(?:\s+(20\d\d))?")


def _norm_words(text: str) -> str:
    return " ".join(re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е")))


def cb_direction(event: dict) -> str:
    """Решение ЦБ по ставке в событии: hold | cut | hike, или пустая строка, если событие не о решении ЦБ."""
    for text in (event.get("title", ""), event.get("text", "")[:500]):
        if not (CB_RE.search(text) and re.search(r"ставк", text, re.I)):
            continue
        hits = [(m.start(), d) for d, rx in EVENT_DIR.items() for m in [rx.search(text)] if m]
        if hits:
            return min(hits)[1]
    return ""


def _noise_share(s: str) -> float:
    """Доля мусора распознавания: одиночные знаки (кроме однобуквенных слов) и токены без кириллицы и цифр."""
    toks = [t for t in s.split() if t.strip("—–-")]
    bad = 0
    for t in toks:
        w = t.strip(".,;:!?()«»\"'")
        bad += not w or (len(w) == 1 and w.lower() not in ONE_LETTER) or not re.search(r"[а-яё\d]", w, re.I)
    return bad / len(toks) if toks else 1.0


def _expired(s: str, doc_date: str | None, today: date | None) -> bool:
    """Ожидание к дате, которая уже прошла: «ожидаем, что 24 июля регулятор сохранит ставку» — 21 октября это не позиция."""
    if not (doc_date and today):
        return False
    d0 = date.fromisoformat(doc_date)
    for m in DAY_RE.finditer(s):
        try:
            d = date(int(m.group(3) or d0.year), MONTHS_GEN.index(m.group(2)) + 1, int(m.group(1)))
        except ValueError:
            continue
        if d0 < d < today:
            return True
    return False


def position_issue(s: str, title: str = "", direction: str = "", today: date | None = None,
                   doc_date: str | None = None) -> str:
    """Почему предложение — не позиция банка для цитаты (пустая строка — годится). Без ремонта текста: не годится —
    не показываем."""
    if not QUOTE_MIN <= len(s) <= QUOTE_MAX:
        return "длина"
    if not re.match(r"[А-ЯЁA-Z0-9«]", s) or not re.search(r"[.!?…][»\")]?$", s):
        return "обрывок предложения"
    if not POSITION_RE.search(NESTED_RE.sub("", s)) or RETRO_RE.search(s):
        return "нет позиции банка"
    if _noise_share(s) > NOISE_SHARE:
        return "мусор распознавания"
    head = _norm_words(title.split(":")[0])
    if HEADER_RE.search(s) or (len(head) >= 8 and _norm_words(s).startswith(head)) or FUSED_RE.search(s):
        return "сросшийся заголовок"
    if direction and RATE_RE.search(s):
        # «ЦБ ожидаемо сохранил ставку» или «ожидаем сохранения паузы» под новостью о снижении — противоречит событию
        acts = {d for d, rx in DONE.items() if rx.search(s)} if CB_RE.search(s) else set()
        acts |= {d for d, rx in EXPECT.items() if rx.search(s)}
        if acts and direction not in acts:
            return "решение ЦБ противоречит событию"
    if _expired(s, doc_date, today):
        return "ожидание к прошедшей дате"
    if today and house_view.revisions_in(s, today, before=doc_date):
        return "прогноз пересмотрен"
    return ""


def _paragraphs(text: str) -> list[list[str]]:
    """Предложения фрагмента по абзацам. Перевод строки перед заглавной — граница абзаца или заголовка
    («…сохраняется\\nДиверсификация…»), перед строчной — продолжение строки; предложение через абзац не переходит."""
    return [[s.strip() for s in SENT_RE.split(re.sub(r"\s+", " ", para)) if s.strip()]
            for para in re.split(r"\n+(?=\s*[А-ЯЁA-Z«•—–-])", text)]


def _clean(s: str) -> str:
    """Без остатка буквицы и ссылок на рисунки — остальной текст предложения не меняется."""
    return strip_artefacts(OCR_LEAD.sub("", s))


def _bank_quote(text: str, event_stems: set[str], title: str = "", event: dict | None = None,
                today: date | None = None, doc_date: str | None = None) -> str:
    """Позиция банка из фрагмента дословно: предложение, где банк говорит от себя («ожидаем», «наш прогноз»), сильнее
    всего совпадающее с темой события, и следующее за ним в том же абзаце, если оно тоже такое. Нет такого — пустая
    строка: честнее плашка «позиции банка нет», чем расписание заседаний, сросшийся заголовок или мусор OCR."""
    direction = cb_direction(event) if event else ""
    stems = event_stems - GENERIC
    best, score = None, 0
    raw = _paragraphs(text)
    paras = [[_clean(s) for s in sents] for sents in raw]
    ok = {(pi, si) for pi, sents in enumerate(paras) for si, s in enumerate(sents)
          if not position_issue(s, title, direction, today, doc_date)}
    for pi, si in sorted(ok):
        k = len(_stems(paras[pi][si]) & stems)
        if k > score:
            best, score = (pi, si), k
    if not best:
        return ""
    pi, si = best
    quote = paras[pi][si]
    # второе предложение — только встык с первым: две цитаты через пропуск в одних кавычках были бы не дословны
    nxt = paras[pi][si + 1] if (pi, si + 1) in ok and not OCR_LEAD.match(raw[pi][si + 1]) else ""
    return f"{quote} {nxt}" if nxt and len(quote) + len(nxt) < QUOTE_MAX else quote


def bank_position(ses: ToolSession, event: dict, facts: list[str], today: date) -> tuple[Source | None, str, str, str]:
    """Позиция банка для краткого разбора: (источник, дословная цитата, заголовок и деталь шага трассы).

    Фрагмент берётся, только если он про ту же тему, что и событие, и совпадает с заголовком по смыслу; по ставке
    первым идёт обзор решения ЦБ (документ вида rates). Из фрагмента — только предложение, где банк говорит от себя
    (_bank_quote). Ни в одном фрагменте такого нет — (None, "", …): в карточке честная плашка NO_BANK_VIEW."""
    ev_topics = set(event.get("topics") or topics_in(event.get("title", "") + " " + event.get("text", ""))) - {"отрасль"}
    hint = " ".join(TOPIC_HINT[t] for t in sorted(ev_topics) if t in TOPIC_HINT)
    ev_stems = _stems(event.get("title", "") + " " + " ".join(facts) + " " + hint)
    res, _ = ses.call("search_bank_analytics", {"query": (event.get("title", "")[:200] + " " + hint).strip()})
    title_stems = (_stems(event.get("title", "")) | _stems(hint)) - GENERIC
    by_lbl = {s.label: s for s in ses.sources}
    found = [f for f in res.get("fragments", []) if ev_topics and not f["stale"] and not f["too_old_for_conclusions"]
             and len(title_stems & _stems(f["doc"] + " " + by_lbl[f["label"]].text)) >= 2
             and set(topics_in(f["doc"] + " " + by_lbl[f["label"]].text)) & ev_topics]
    if "ставка" in ev_topics:
        found.sort(key=lambda f: f.get("doc_kind") != "rates")
    title = f"search_bank_analytics(\"{event.get('title', '')[:60]}{' + ' + hint if hint else ''}\")"
    for f in found:
        src = by_lbl[f["label"]]
        quote = _bank_quote(src.text, ev_stems, title=src.doc, event=event, today=today, doc_date=src.date)
        if quote:
            return src, quote, title, f"позиция банка: «{(src.doc or src.name)[:60]}» от {src.date}"
    return None, "", title, f"фрагментов по теме события: {len(found)}, дословной позиции банка в них нет — «{NO_BANK_VIEW}»"


# Запасной главный вывод под роль, если каналов влияния нет: (расчёты по приоритету, виды признаков отбора по приоритету)
FALLBACK = {"cfo": (("Стоимость долга", "Валютные", "Выручка"), ("debt", "fx", "customers", "suppliers", "own")),
            "ceo": (("Выручка", "Закупки", "Стоимость долга"), ("customers", "own", "suppliers", "demand", "debt", "fx")),
            "treasurer": (("Стоимость долга", "Валютные", "Доход по депозитам"), ("debt", "fx", "deposits", "customers")),
            "accountant": (("Утильсбор", "Пошлина", "Налог", "Валютные", "Стоимость долга"),
                           ("fx", "debt", "customers", "suppliers", "own"))}


def rules_card(event: dict, company_id: str, match: Match, bank: BM25Index, news: BM25Index,
               today: date, bank_meta: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Карточка контракта без LLM и трасса её сборки."""
    ses = ToolSession(company_id, event, bank, news, today, bank_meta or [])
    trace: list[dict] = []

    def step(kind, title, detail, t0):
        trace.append({"kind": kind, "title": title, "detail": detail, "ms": int((time.monotonic() - t0) * 1000), "cost": 0})

    t0 = time.monotonic()
    facts = _facts(event)
    step("plan", "План (правила)", "модель недоступна: факты новости → признаки компании → позиция банка → расчёт", t0)

    t0 = time.monotonic()
    prof = ses.profile_source()
    p = profiles.get(company_id)
    why = [{"t": t, "p": "fact", "src": prof.label} for t in _why(_by_kind(match))[:3]]
    step("tool", "get_company_profile()", f"признаков компании: {len(why)}", t0)

    t0 = time.monotonic()
    src, quote, step_title, detail = bank_position(ses, event, facts, today)
    bank_stmt = {"t": quote, "p": "bank", "src": src.label} if src else None
    bank_block = {"quote": quote, "doc": src.doc or src.name, "date": src.date, "stale": src.stale} if src else None
    step("tool", step_title, detail, t0)

    t0 = time.monotonic()
    calcs = ses.calculate_impact()
    step("tool", "calculate_impact()", "; ".join(f"{c['title']}: {c['value']}" for c in calcs)[:200] or
         "числового шока в новости нет — расчёта нет", t0)

    t0 = time.monotonic()
    chs = channels.build(event, match, p, calcs, today, channels.meetings_for(bank, news))
    step("plan", CHANNELS_STEP, channels.as_text(chs)[:600] or "каналов влияния не найдено — вывод по признакам отбора", t0)

    # метки S… → индексы sources[] в порядке появления в карточке
    cited: list[str] = ["S1"]

    def idx(label: str) -> int:
        if label not in cited:
            cited.append(label)
        return cited.index(label)

    def fix(s: dict) -> dict:
        return {**s, "src": idx(s["src"])}

    what = [fix({"t": f, "p": "fact", "src": "S1"}) for f in facts]
    if bank_stmt:
        bank_stmt = fix(bank_stmt)
    why = [fix(s) for s in why]
    calc = [c | {"inputs": [fix(i) for i in c["inputs"]]} for c in calcs]
    by_label = {s.label: s for s in ses.sources}
    sources = [{"name": by_label[lbl].name, "kind": by_label[lbl].kind, "date": by_label[lbl].date or today.isoformat(),
                "url": by_label[lbl].url} for lbl in cited]

    def pick(prefer: tuple[str, ...], order: tuple[str, ...]) -> str:
        c = next((c for c in calcs if c["title"].startswith(prefer)), calcs[0] if calcs else None)
        return _take(c) if c else headline(match, order)

    takes = channels.takeaways(chs)
    st = channels.statements(chs)
    card = {
        "id": f"{event['id']}-{company_id}",
        "type": channels.card_type(chs, calcs),
        "impact": "high" if match.score >= 6 else "medium" if match.score >= 3.5 else "low",
        "topics": [t for t in event.get("topics", []) if t in TOPICS] or ["отрасль"],
        "is_new": True, "published": event.get("published") or today.isoformat(),
        "title": event.get("title", "")[:200],
        # главный вывод под роль — из каналов влияния; каналов нет — из расчёта или признаков отбора
        "takeaway": {role: takes[role] or pick(*FALLBACK[role]) for role in FALLBACK},
        # короткий слой: почему вам, позиция банка, факт новости — заголовок события уже виден в карточке
        "key_points": (why[:1] + ([bank_stmt] if bank_stmt else []) + what[:1])[:3],
        "why_short": match.reasons[0]["why"] if match.reasons else "Событие из отобранных для компании новостей",
        "matched": match.matched, "sources": sources,
        "blocks": {"what": what, "why": why, "risks": st["risks"], "opportunities": st["opportunities"], "actions": st["actions"],
                   **({"calc": calc} if calc else {}),
                   "bank": bank_block or {"quote": "", "doc": NO_BANK_VIEW, "date": today.isoformat(), "stale": False},
                   "basis": {"features": match.matched + [f"Продукты банка: {', '.join(p['bank_products']) or 'нет'}"],
                             "coverage": " ".join(x for x in [NOTICE, ses.industry["note"], "; ".join(profiles.data_quality(p))] if x)}},
        "trace": [], "chat": [],
    }
    if len(card["key_points"]) < 2:
        card["key_points"] = (what + why)[:2] if len(what + why) >= 2 else what * 2
    rep = validate_card(card, today=today, source_texts={i: by_label[lbl].text for i, lbl in enumerate(cited)})
    trace.append({"kind": "check", "title": "Самопроверка (код)", "detail": rep.summary(), "ms": 0, "cost": 0})
    trace.append({"kind": "done", "title": "Черновой разбор готов" if rep.ok else "Разбор не прошёл проверку",
                  "detail": NOTICE, "ms": 0, "cost": 0})
    card["trace"] = trace
    return (card if rep.ok else None), trace
