"""Хранилище под параллельной нагрузкой: аудит нашёл ответы 500 при 12 потоках (строки читались после снятия
блокировки с общего соединения SQLite). Тест воспроизводит ту нагрузку."""
from concurrent.futures import ThreadPoolExecutor

from backend.store import Store


def test_parallel_reads_and_writes_do_not_corrupt_rows(tmp_path):
    s = Store(tmp_path / "app.db")
    for c in range(30):
        for k in range(5):
            s.put_card(f"C{c}", {"id": f"e{k}", "title": f"Новость {k} " * 20, "topics": ["ставка"]}, "rules")

    def work(i):
        c = f"C{i % 30}"
        for _ in range(20):
            assert len(s.cards(c)) >= 5
            assert all(card["id"] for card in s.cards(c) for card in [card[0]])
            s.all_cards("rules")
            s.put_card(c, {"id": f"x{i}", "title": "обновление", "topics": []}, "rules")
            s.put_feedback(c, "e1", "cfo", "up")
        return True

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert all(pool.map(work, range(60)))
