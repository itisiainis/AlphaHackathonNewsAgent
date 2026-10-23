"""API под параллельной нагрузкой — повтор замера аудита на HEAD (B03, шаг 6).

До fc4b14c строки SQLite читались после снятия блокировки с общего соединения: 12 параллельных клиентов
получили 21 ответ 500 из 90. Здесь та же картина в большем объёме: 12 потоков × 90 запросов к трём
демо-компаниям, их событиям и /api/bootstrap на чистой базе — краткие разборы при этом пишутся в базу
одновременно с чтением. Ждём ни одного ответа 5xx и ни одного ответа с чужими или перепутанными строками.
На старом хранилище (курсор читается после снятия блокировки) тест падает: около 20 ответов 500 из 1080.
Идёт ~40 с: каждый запрос заново отбирает новости под компанию (agent/matching.py), /api/bootstrap — ~0,1 с."""
import importlib
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

DEMO = ("U_25835", "U_177581", "U_200890")
URLS = [f"/api/companies/{c}" for c in DEMO] + [f"/api/companies/{c}/events" for c in DEMO] + ["/api/bootstrap"]
THREADS, PER_THREAD = 12, 90


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    os.environ["APP_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))  # чистая база: брифы пишутся под нагрузкой
    os.environ["APP_TODAY"] = "2026-10-21"
    os.environ.pop("LLM_API_KEY", None)
    import backend.app as m
    return importlib.reload(m)


def _broken(url: str, body) -> str:
    """Ответ 200, но строки перепутаны (курсор перезаписан соседним запросом) — пустая строка, если всё сходится."""
    if url == "/api/bootstrap":
        if len(body["all_companies"]) != 115:
            return "в bootstrap не 115 компаний"
        bad = [c for c in body["insights"] if [i["id"] for i in body["insights"][c]] != list(body["origins"][c])]
        return f"карточки и происхождение расходятся у {bad}" if bad else ""
    if url.endswith("/events"):
        return "" if isinstance(body, list) and all(e["id"] and e["title"] for e in body) else "события без id или заголовка"
    cid = url.rsplit("/", 1)[1]
    if body["company"]["id"] != cid:
        return f"вместо {cid} пришла компания {body['company']['id']}"
    ids = [i["id"] for i in body["insights"]]
    # карточки разборов именуются <событие>-<компания>; примеры команды (seed) — без суффикса
    alien = [i for i in ids if "-U_" in i and not i.endswith("-" + cid)]
    if alien or ids != list(body["origins"]):
        return f"чужие или перепутанные карточки у {cid}: {alien[:3]}"
    return ""


def test_parallel_reads_never_fail(app_module):
    def work(k: int) -> list[tuple[str, int, str]]:
        client, out = TestClient(app_module.app, raise_server_exceptions=False), []
        for n in range(PER_THREAD):
            url = URLS[(k + n) % len(URLS)]  # потоки идут со сдвигом: в каждый момент смешаны все адреса
            r = client.get(url)
            out.append((url, r.status_code, _broken(url, r.json()) if r.status_code == 200 else ""))
        return out

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = [x for part in pool.map(work, range(THREADS)) for x in part]

    assert len(results) == THREADS * PER_THREAD == 1080
    codes = Counter(code for _, code, _ in results)
    failed = Counter(url for url, code, _ in results if code >= 500)
    assert not failed, f"ответы 5xx: {dict(failed)}; все коды: {dict(codes)}"
    broken = [(url, why) for url, _, why in results if why]
    assert not broken, f"перепутанные ответы ({len(broken)}): {broken[:5]}"
    assert codes == {200: 1080}
