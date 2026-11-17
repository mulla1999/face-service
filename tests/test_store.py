import threading

import numpy as np
import pytest

from app.engine import normalize
from app.store import EMB_DIM, IdentityStore


def vec(i: int) -> np.ndarray:
    v = np.zeros(EMB_DIM, np.float32)
    v[i] = 1.0
    return v


def test_enroll_search_and_reload(tmp_path):
    path = str(tmp_path / "ids.db")
    store = IdentityStore(path)
    assert store.search(vec(0)) == []
    a = store.enroll("Asha", vec(0))
    store.enroll("Ravi", vec(1))
    name, iid, sim = store.search(vec(0), top_k=5)[0]
    assert (name, iid) == ("Asha", a) and sim == pytest.approx(1.0)
    assert len(store.search(vec(0), top_k=5)) == 2  # never more than what's enrolled
    store.close()

    reopened = IdentityStore(path)  # embeddings survive a restart
    assert reopened.list_people() == {"Asha": [a], "Ravi": [reopened.list_people()["Ravi"][0]]}
    assert reopened.search(vec(1))[0][0] == "Ravi"
    reopened.close()


def test_remove_rebuilds_index(tmp_path):
    store = IdentityStore(str(tmp_path / "ids.db"))
    a = store.enroll("Asha", vec(0))
    b = store.enroll("Asha", normalize(vec(0) + vec(2)))
    assert store.list_people() == {"Asha": [a, b]}
    assert store.remove(a) and not store.remove(a)
    assert store.count() == 1 and store.search(vec(0))[0][1] == b
    store.close()


def test_wrong_dimension_is_rejected(tmp_path):
    store = IdentityStore(str(tmp_path / "ids.db"))
    with pytest.raises(ValueError, match="512"):
        store.enroll("X", np.ones(128, np.float32))
    store.close()


def test_search_is_safe_while_samples_are_removed(tmp_path):
    """search() used to read row ids outside the lock and could crash during a concurrent remove()."""
    store = IdentityStore(str(tmp_path / "ids.db"))
    ids = [store.enroll(f"P{i}", vec(i)) for i in range(50)]
    errors: list[BaseException] = []

    def searcher():
        try:
            for _ in range(300):
                store.search(vec(49), top_k=10)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the assertion below
            errors.append(exc)

    thread = threading.Thread(target=searcher)
    thread.start()
    for iid in ids[:-1]:
        store.remove(iid)
    thread.join()
    assert errors == [] and store.count() == 1
    store.close()
