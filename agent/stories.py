"""Тот же ли это сюжет: сравнение заголовков новостей.

Нужно в трёх местах: склейка дублей при сборе новостей, память разборов (готовый разбор того же сюжета)
и распознавание рутины (ежедневные шаблоны вроде «ЦБ установил курсы на 21 октября»).

Как работает. Сходство — доля общих основ слов (коэффициент Жаккара) без чисел и дат. Одного сходства мало:
замер на 289 размеченных парах заголовков за 21 день (docs/experiments/learning.md) показал, что среди
пар со сходством ≥ 0,6 больше половины — разные события (54 из 92 на независимой выборке). Самые похожие заголовки — это шаблоны
с другой датой или числом («индекс Мосбиржи +0,34%» и «−0,07%») и противоположности («поздравил» /
«не поздравил»). Поэтому поверх сходства — запреты, каждый из которых взят из найденных ошибок:
- числа, даты, месяцы, дни недели расходятся (ни один набор не входит в другой);
- один и тот же источник в разные дни — это новая публикация, а не повтор;
- разные имена собственные с обеих сторон (Внуково / Шереметьево, США / КНР);
- отрицание только с одной стороны («не поздравил»);
- противоположное направление («подорожала» / «подешевела», «рост» / «снижение»).
С запретами на независимой выборке — 0 ложных совпадений из 28 (точность ≥ 89% с 95% уверенностью, «правило
трёх»); без них — 54 ложных из 92. Синонимы названий (ЦБ = Банк России) добавлены после этой проверки — см. отчёт.
"""
from __future__ import annotations

import re

MONTHS = r"январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр"
WEEKDAYS = r"понедельник|вторник|сред[аыу]|четверг|пятниц|суббот|воскресень"
STOP = {"это", "как", "что", "для", "при", "его", "она", "они", "года", "году", "все", "уже", "или"}
NUM = re.compile(r"\d+(?:[.,]\d+)?")
ANTONYMS = [("подорож", "подешев"), ("повыс", "пониз"), ("повыс", "сниз"), ("рост", "сниж"), ("вырос", "сниз"),
            ("вырос", "упал"), ("подъем", "сниж"), ("укреп", "ослаб"), ("открыл", "закрыл"), ("ввел", "отмен"),
            ("сохран", "сниз"), ("сохран", "повыс")]


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def words(title: str) -> set[str]:
    """Основы слов заголовка без чисел, дат (месяцы, дни недели) и служебных слов."""
    t = re.sub(r"\d+|" + MONTHS + "|" + WEEKDAYS, " ", _norm(title))
    return {w[:6] for w in re.findall(r"[а-яa-z]{3,}", t) if w not in STOP}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def specifiers(title: str) -> set[str]:
    """Что отличает один выпуск шаблона от другого: числа, месяцы, дни недели."""
    t = _norm(title)
    nums = {n.replace(",", ".").rstrip("0").rstrip(".") if re.search(r"[.,]", n) else n for n in NUM.findall(t)}
    return nums | set(re.findall(MONTHS, t)) | set(re.findall(WEEKDAYS, t))


# Одно и то же под разными именами — иначе «ЦБ» и «Банк России» считались бы разными участниками
ALIASES = {"центр": "цб", "евросоюз"[:5]: "ес", "росси": "рф", "штаты": "сша"}
# Издания и агентства в подписи («— ТАСС», «опрос РБК») — это источник, а не участник события
MEDIA = {"тасс", "рбк", "rtrs", "интер", "ведом", "комме", "reute", "bloom", "wsj", "ft", "axios", "polit", "cnn",
         "риа", "прайм", "cnbc", "the", "ms", "now", "times", "post", "washi", "econo", "lance"}


def entities(title: str) -> set[str]:
    """Имена собственные и аббревиатуры. В заголовке капсом регистр ничего не говорит — тогда пусто."""
    letters = [c for c in title if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        return set()
    text = re.sub(r"[Бб]анк[а-я]* [Рр]осси[а-я]*", "ЦБ", title)   # «Банк России» = «ЦБ»
    # слово с заглавной в начале второго и следующих предложений («…Мосбиржи. Завтра…», «По его словам…») — не имя;
    # первое слово заголовка оставляем: там часто и есть участник («Goldman повысил…» / «Barclays повысил…»)
    text = re.sub(r"([.!?:]\s+)[A-ZА-ЯЁ][a-zа-яё]+", r"\1", text)
    names = {_norm(w)[:5] for w in re.findall(r"\b[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё]+", text)}
    return {ALIASES.get(n, n) for n in names} - MEDIA


def conflict(a: str, b: str, source_a: str | None = None, source_b: str | None = None,
             day_a: str | None = None, day_b: str | None = None) -> str | None:
    """Причина, по которой похожие заголовки — разные события, или None."""
    sa, sb = specifiers(a), specifiers(b)
    if sa and sb and not (sa <= sb or sb <= sa):
        return "числа или даты расходятся"
    if source_a and source_a == source_b and day_a and day_b and day_a[:10] != day_b[:10]:
        return "тот же источник, другой день"
    ea, eb = entities(a), entities(b)
    if ea - eb and eb - ea:
        return "разные названия"
    la, lb = _norm(a), _norm(b)
    if bool(re.search(r"\bне\b", la)) != bool(re.search(r"\bне\b", lb)):
        return "отрицание с одной стороны"
    for x, y in ANTONYMS:
        if (x in la and y in lb and y not in la) or (y in la and x in lb and x not in la):
            return "противоположное направление"
    return None


SAME_STORY = 0.6  # порог сходства для памяти разборов: зафиксирован до проверки на независимой выборке


def same_story(a: str, b: str, threshold: float = SAME_STORY, **meta) -> bool:
    return jaccard(words(a), words(b)) >= threshold and conflict(a, b, **meta) is None


TEMPLATE_SIM = 0.6       # сходство выпусков одного шаблона
TEMPLATE_DAYS = 3        # шаблон — если выходил в столько разных дней
TEMPLATE_SHORT = 5       # короткий заголовок без чисел («Важное к началу основной сессии») — тоже шаблон


def template_days(title: str, history: list[tuple[str, str]]) -> int:
    """В скольких разных днях источник публиковал этот шаблон. history — [(день, заголовок)] того же источника.

    Шаблон — одинаковые слова при разных деталях (дата, число, день недели). Длинный заголовок с теми же
    деталями в разные дни — это продолжающееся освещение события или повтор поста, а не рутина: замер
    показал, что иначе в «рутину» попадают опросы перед заседанием ЦБ по ставке."""
    w = words(title)
    similar = [(d, t) for d, t in history if jaccard(w, words(t)) >= TEMPLATE_SIM]
    days = {d[:10] for d, _ in similar}
    if len(days) < TEMPLATE_DAYS:
        return 0
    varies = any(specifiers(t) != specifiers(title) for _, t in similar)
    return len(days) if varies or len(w) <= TEMPLATE_SHORT else 0
