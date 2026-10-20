"""Собирает prototype.html из template.html + mock_data.json: python build.py"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
data = json.loads((D / "mock_data.json").read_text(encoding="utf-8"))
tpl = (D / "template.html").read_text(encoding="utf-8")
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
(D / "prototype.html").write_text(tpl.replace("__DATA__", payload), encoding="utf-8")
print("prototype.html собран")
