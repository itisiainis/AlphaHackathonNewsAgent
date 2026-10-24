"""Собирает prototype.html из template.html + mock_data.json: python build.py

В примеры карточек добавляется расчёт эффекта кодом (agent/impact.py) — так же, как это делает бэкенд."""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent))
from agent import impact, profiles, roles  # noqa: E402
from frontend.page import render  # noqa: E402

data = json.loads((D / "mock_data.json").read_text(encoding="utf-8"))
today = data.get("generated_at", "")[:10] or "2026-10-21"
# компании — из профилей (как отдаёт бэкенд), подписи демо-компаний — из моков
data["companies"] = [profiles.view(profiles.get(c["id"])) | {"display": c["display"]} for c in data["companies"]]
# роли и зоны ответственности — из общего конфига data/roles.json, как отдаёт бэкенд
data["roles"], data["zones"] = roles.public(), roles.zones()
data["insights"] = {cid: [impact.attach(c, profiles.get(cid), today) for c in cards] for cid, cards in data["insights"].items()}
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
(D / "prototype.html").write_text(render(payload), encoding="utf-8")
print("prototype.html собран")
