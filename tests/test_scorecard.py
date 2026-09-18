import numpy as np
import pytest

from src.models.scorecard import cutoff_probability, decide, points_table, scaling, to_score


def test_base_odds_map_to_base_score(cfg):
    assert to_score(np.array([0.5]), cfg)[0] == pytest.approx(600)


def test_doubling_odds_adds_pdo(cfg):
    low, high = to_score(np.array([1 / 2, 2 / 3]), cfg)  # odds 1:1 and 2:1
    assert high - low == pytest.approx(50)


def test_scores_are_clipped_to_range(cfg):
    scores = to_score(np.array([0.0, 1.0]), cfg)
    assert scores.tolist() == [300, 850]


def test_cutoff_probability_inverts_to_score(cfg):
    assert cutoff_probability(cfg) == pytest.approx(0.5)
    shifted = {**cfg, "scoring": {**cfg["scoring"], "cutoff_score": 650}}
    p = cutoff_probability(shifted)
    assert to_score(np.array([p]), shifted)[0] == pytest.approx(650)


def test_decide_approves_at_cutoff(cfg):
    assert decide(np.array([0.49, 0.5, 0.9]), cfg).tolist() == [0, 1, 1]


def test_points_table_reproduces_scores(cfg, data, models):
    pipe = models["LogisticRegression"]
    table = points_table(pipe, cfg)
    base, weights = table.iloc[0], table.iloc[1:].set_index("feature")["points"]
    encoded = pipe.named_steps["prep"].transform(data["X_test"])
    encoded = encoded.toarray() if hasattr(encoded, "toarray") else encoded
    names = [n.split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    rebuilt = base["points"] + encoded @ weights.loc[names].to_numpy()
    factor, offset = scaling(cfg)
    expected = offset + factor * pipe.decision_function(data["X_test"])
    assert np.allclose(rebuilt, expected)
