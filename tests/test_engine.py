import numpy as np
import pytest

from app.engine import THRESH_MATCH, THRESH_UNCERTAIN, cosine_similarity, normalize, similarity_to_percent, verdict


def test_thresholds_land_on_the_documented_percentages():
    assert similarity_to_percent(THRESH_UNCERTAIN) == 45.0
    assert similarity_to_percent(THRESH_MATCH) == 70.0
    assert similarity_to_percent(-1.0) == 0.0
    assert similarity_to_percent(1.0) == 100.0


def test_percent_never_decreases_as_similarity_grows():
    sims = np.linspace(-1, 1, 401)
    pcts = [similarity_to_percent(s) for s in sims]
    assert pcts == sorted(pcts)


@pytest.mark.parametrize(("sim", "expected"), [
    (THRESH_MATCH, "same_person"),
    (0.9, "same_person"),
    (THRESH_UNCERTAIN, "uncertain"),
    (THRESH_MATCH - 0.01, "uncertain"),
    (THRESH_UNCERTAIN - 0.01, "different_person"),
    (-0.2, "different_person"),
])
def test_verdict_boundaries(sim, expected):
    assert verdict(sim) == expected


def test_cosine_of_normalized_vectors():
    a, b = normalize(np.array([3.0, 4.0])), normalize(np.array([4.0, -3.0]))
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)
    assert normalize(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]  # no division by zero
