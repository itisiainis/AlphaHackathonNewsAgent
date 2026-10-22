"""Дымовая проверка демо на живом сервере (B03, шаг 6). Только стандартная библиотека Python.

    python scripts/smoke_demo.py --base http://127.0.0.1:8000 [--no-model] [--strict]

Проверки:
- health — /healthz и /api/health отвечают, сервис жив;
- bootstrap — /api/bootstrap не тяжелее --max-bootstrap-kb, в нём все компании (--companies) и три демо-компании;
- demo_cards — у каждой демо-компании в ленте есть карточки;
- demo_traces — готовые разборы ИИ демо-компаний (data/cards) не пишут «Все проверки пройдены» после проваленной
  проверки. Пока эти разборы не пересчитаны (W4-T1), это предупреждение; с --strict — ошибка;
- fresh_run_trace — свежий разбор агентом на модели сервера: карточка собрана, шаги выполнены моделью, а не взяты
  из кеша, итог — последний шаг, и в трассе нет «Все проверки пройдены» после проваленной проверки.

Свежий разбор — один запуск агента на той модели, что подключена к серверу: с заглушкой — 0 ₽, с настоящим ключом —
около рубля, поэтому против боевого сервера запускайте с --no-model. Разбор сохраняется как новость, присланная
клиентом, — у недемонстрационной компании (--company), чтобы не засорять ленты демо-компаний.
Итог — JSON на stdout; код выхода 1, если не прошла хотя бы одна проверка (имя и причина — в stderr).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

DEMO = ("U_25835", "U_177581", "U_200890")
CLAIM = "Все проверки пройдены"
# провал проверки: ошибки кода, проверка не выполнена, правка или карточка не прошли проверку
FAILED = re.compile(r"ошибок: [1-9]|не выполнена|не прошла|не проверен", re.I)


def trace_problem(trace: list[dict]) -> str:
    """Где трасса противоречит себе: «Все проверки пройдены» после проваленной проверки. Пустая строка — всё честно.

    Проверка кодом относится к черновику перед ней: новый черновик снимает её прежний провал, и «Все проверки
    пройдены» у проверки нового черновика — правда. Замечание проверяющей модели снимает только её же повторная
    проверка без замечаний: пока оно не снято, итог разбора не может заявлять, что пройдены все проверки."""
    code_fail = model_fail = ""
    for n, st in enumerate(trace, 1):
        kind, title, detail = st.get("kind", ""), st.get("title", ""), str(st.get("detail") or "")
        if kind == "draft":
            code_fail = ""
        if CLAIM in f"{title} {detail}" and (code_fail or (kind == "done" and model_fail)):
            return f"шаг {n} «{title}»: «{CLAIM}» после проваленной проверки «{code_fail or model_fail}»"
        if kind != "check" and "проверк" not in title.lower():
            continue
        failed = bool(FAILED.search(f"{title} {detail}"))
        if "(модель)" in title:
            # у проверяющей модели провал — любые замечания
            failed = failed or detail.strip() not in ("", "замечаний нет")
            model_fail = f"{title}: {detail[:100]}" if failed else ""
        elif failed:
            code_fail = f"{title}: {detail[:100]}"
    return ""


class Server:
    def __init__(self, base: str, timeout: float):
        self.base, self.timeout = base.rstrip("/"), timeout  # схема http(s) проверена в main()

    def call(self, path: str, body: dict | None = None, timeout: float | None = None) -> tuple[int, object, int]:
        """(код ответа, JSON или текст, размер ответа в байтах). Ошибки HTTP — тоже ответ, а не исключение."""
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",  # noqa: S310 — схема проверена
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:  # noqa: S310 — схема проверена
                code, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, raw = e.code, e.read()
        try:
            return code, json.loads(raw), len(raw)
        except ValueError:
            return code, raw.decode("utf-8", "replace")[:300], len(raw)


class Check(Exception):
    """Проверка не прошла; status — fail (ошибка), warn (предупреждение) или skip (пропущена)."""

    def __init__(self, detail: str, status: str = "fail"):
        super().__init__(detail)
        self.status = status


def must(cond, why: str) -> None:
    if not cond:
        raise Check(why)


def check_health(s: Server, _a, _ctx) -> str:
    code, _, _ = s.call("/healthz")
    must(code == 200, f"/healthz → HTTP {code}")
    code, h, _ = s.call("/api/health")
    must(code == 200 and isinstance(h, dict) and h.get("ok") is True, f"/api/health → HTTP {code}: {h}")
    return (f"модель {'доступна' if h.get('llm_online') else 'недоступна'}, новостей {h.get('news_events')}, "
            f"фрагментов аналитики банка {h.get('bank_fragments')}, дата сервиса {h.get('today')}")


def check_bootstrap(s: Server, a, ctx) -> str:
    t0 = time.monotonic()
    code, boot, size = s.call("/api/bootstrap")
    sec = time.monotonic() - t0
    must(code == 200 and isinstance(boot, dict), f"/api/bootstrap → HTTP {code}")
    ctx["boot"] = boot
    kb = size / 1024
    n = len(boot.get("all_companies", []))
    must(kb <= a.max_bootstrap_kb, f"ответ {kb:.0f} КБ — больше {a.max_bootstrap_kb} КБ")
    must(n == a.companies, f"компаний в списке {n}, ожидалось {a.companies}")
    loaded = [c["id"] for c in boot.get("companies", [])]
    missing = [c for c in DEMO if c not in loaded]
    must(not missing, f"нет демо-компаний: {', '.join(missing)}")
    return f"{kb:.0f} КБ за {sec:.2f} с; компаний {n}, загружены сразу {len(loaded)}"


def check_demo_cards(_s, _a, ctx) -> str:
    boot = ctx.get("boot")
    must(boot, "нет ответа /api/bootstrap")
    out, empty = [], []
    for cid in DEMO:
        cards = [c for c in boot["insights"].get(cid, []) if c.get("id") and c.get("title")]
        origins = boot.get("origins", {}).get(cid, {})
        by = {}
        for c in cards:
            by[origins.get(c["id"], "?")] = by.get(origins.get(c["id"], "?"), 0) + 1
        out.append(f"{cid}: {len(cards)} ({', '.join(f'{k} {v}' for k, v in sorted(by.items()))})")
        if not cards:
            empty.append(cid)
    must(not empty, "нет карточек у " + ", ".join(empty))
    return "; ".join(out)


def check_demo_traces(_s, a, ctx) -> str:
    boot = ctx.get("boot")
    must(boot, "нет ответа /api/bootstrap")
    bad, n = [], 0
    for cid in DEMO:
        origins = boot.get("origins", {}).get(cid, {})
        for c in boot["insights"].get(cid, []):
            if origins.get(c["id"]) != "agent":
                continue
            n += 1
            why = trace_problem(c.get("trace") or [])
            if why:
                bad.append(f"{c['id']}: {why}")
    if bad:
        raise Check(f"{len(bad)} из {n} разборов ИИ противоречат себе — " + " | ".join(bad),
                    "fail" if a.strict else "warn")
    return f"разборов ИИ у демо-компаний: {n}, противоречий в трассе нет"


def check_fresh_run(s: Server, a, ctx) -> str:
    if a.no_model:
        raise Check("пропущено: --no-model", "skip")
    boot = ctx.get("boot")
    must(boot or a.company, "нет ответа /api/bootstrap — не из чего выбрать компанию для свежего разбора")
    cid = a.company or next((c["id"] for c in boot.get("all_companies", []) if c["id"] not in DEMO), None)
    must(cid, "не из чего выбрать компанию для свежего разбора — укажите --company")
    # число в заголовке своё у каждого запуска: другое событие, промах кеша модели и не «тот же сюжет» для памяти разборов
    tag = time.time_ns() // 1000 % 10**9
    title = f"Банк России повысил ключевую ставку до 18% годовых — проверка {tag}"
    text = (f"{title}. Совет директоров Банка России повысил ключевую ставку до 18% годовых. "
            "Регулятор допустил дальнейшее повышение, если инфляция не начнёт замедляться.")
    t0 = time.monotonic()
    code, r, _ = s.call("/api/analyze", {"company_id": cid, "title": title, "text": text}, timeout=a.run_timeout)
    sec = time.monotonic() - t0
    must(code != 429, "лимит запросов к модели — повторите через минуту")
    must(code == 200 and isinstance(r, dict), f"/api/analyze → HTTP {code}: {r}")
    must(r.get("mode") != "rules", f"модель недоступна ({r.get('reason')}) — свежего разбора агентом не было")
    must(not r.get("reused"), "ответ взят из памяти разборов — свежего разбора агентом не было")
    trace = (r.get("card") or {}).get("trace") or r.get("trace") or []
    must(trace, "в ответе нет трассы")
    why = trace_problem(trace)
    must(not why, why)
    must(trace[-1].get("kind") == "done", f"последний шаг трассы — «{trace[-1].get('title')}», а не итог")
    must(r.get("ok"), "агент не собрал карточку: " + "; ".join(map(str, r.get("errors", [])))[:300])
    model = [t for t in trace if t.get("model")]
    must(not model or any(not t.get("cached") for t in model), "все шаги модели взяты из кеша — запуск не свежий")
    return (f"{cid}: {len(trace)} шагов за {sec:.1f} с, шагов модели {len(model)} "
            f"(из кеша {sum(1 for t in model if t.get('cached'))}); итог: {trace[-1].get('detail', '')[:160]}")


CHECKS = [("health", check_health), ("bootstrap", check_bootstrap), ("demo_cards", check_demo_cards),
          ("demo_traces", check_demo_traces), ("fresh_run_trace", check_fresh_run)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Дымовая проверка демо на живом сервере (только стандартная библиотека).")
    ap.add_argument("--base", required=True, help="адрес сервера, например http://127.0.0.1:8000")
    ap.add_argument("--companies", type=int, default=115, help="сколько компаний должно быть в списке (по умолчанию 115)")
    ap.add_argument("--max-bootstrap-kb", type=int, default=1024, help="предел размера /api/bootstrap, КБ")
    ap.add_argument("--company", default="", help="компания для свежего разбора (по умолчанию — крупнейшая не из демо)")
    ap.add_argument("--no-model", action="store_true", help="не запускать свежий разбор агентом (боевой ключ)")
    ap.add_argument("--strict", action="store_true", help="предупреждения тоже считать ошибками")
    ap.add_argument("--timeout", type=float, default=30, help="ожидание ответа на чтение, с")
    ap.add_argument("--run-timeout", type=float, default=240, help="ожидание свежего разбора агентом, с")
    a = ap.parse_args(argv)
    if not re.match(r"https?://", a.base):
        ap.error(f"--base: нужен адрес http:// или https://, а не {a.base!r}")
    s, ctx, results = Server(a.base, a.timeout), {}, []
    for name, fn in CHECKS:
        t0 = time.monotonic()
        try:
            res = {"name": name, "ok": True, "status": "pass", "detail": fn(s, a, ctx)}
        except Check as e:
            res = {"name": name, "ok": e.status != "fail", "status": e.status, "detail": str(e)}
        except (OSError, KeyError, TypeError, ValueError) as e:  # сервер недоступен или ответил не тем форматом
            res = {"name": name, "ok": False, "status": "fail", "detail": f"{type(e).__name__}: {e}"}
        res["ms"] = round((time.monotonic() - t0) * 1000)
        results.append(res)
    failed = [r["name"] for r in results if not r["ok"]]
    for r in results:
        if r["status"] in ("fail", "warn"):
            print(f"{r['status'].upper()} {r['name']}: {r['detail']}", file=sys.stderr)
    print(json.dumps({"ok": not failed, "base": a.base, "failed": failed,
                      "warnings": [r["name"] for r in results if r["status"] == "warn"], "checks": results},
                     ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
