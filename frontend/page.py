"""Сборка страницы для пользователя: шаблон + данные, без служебных комментариев разработчиков.

Комментарии остаются в template.html для команды; в отдаваемый браузеру HTML они не попадают.
Вырезаются только безопасные формы: блочные /* … */ в <style> и <script>, строки, целиком состоящие
из // комментария, и хвостовые « // комментарий» после кода (URL вида https://… не затрагиваются:
перед // в них нет пробела)."""
import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "template.html"
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_LINE = re.compile(r"^[ \t]*//.*\n", re.M)
_TAIL = re.compile(r"(?<=[;,{}()\]`'\"\w])[ \t]+//[ \t][^\n]*")


def strip_comments(html: str) -> str:
    return _TAIL.sub("", _LINE.sub("", _BLOCK.sub("", html)))


def render(data_json: str) -> str:
    """Данные подставляются после очистки: текст новостей с «//» не затрагивается."""
    return strip_comments(TEMPLATE.read_text(encoding="utf-8")).replace("__DATA__", data_json)
