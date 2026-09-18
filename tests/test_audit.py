import numpy as np
import pandas as pd
import pytest

from src.data.profile import base_rates
from src.fairness.audit import age_threshold_sensitivity, disparities, group_metrics
from src.models.scorecard import decide

# Group a: 4 rows, 3 approved. Group b: 4 rows, 1 approved.
Y_TRUE = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 1, 0, 0, 0])
GROUPS = pd.DataFrame({"g": list("aaaabbbb")})


def test_base_rates():
    rates = base_rates(Y_TRUE, GROUPS)
    assert rates["positive_rate"].tolist() == [0.5, 0.5]
    assert rates["count"].tolist() == [4, 4]


def test_group_metrics_by_hand():
    table = group_metrics(Y_TRUE, Y_PRED, np.arange(8) * 100.0, GROUPS).set_index("group")
    assert table.loc["a", "selection_rate"] == pytest.approx(0.75)
    assert table.loc["b", "selection_rate"] == pytest.approx(0.25)
    assert table.loc["a", "tpr"] == pytest.approx(1.0)
    assert table.loc["b", "tpr"] == pytest.approx(0.5)
    assert table.loc["a", "fpr"] == pytest.approx(0.5)
    assert table.loc["b", "fpr"] == pytest.approx(0.0)
    assert table.loc["b", "mean_score"] == pytest.approx(550)
    assert table["count"].tolist() == [4, 4]


def test_disparities_by_hand():
    row = disparities(Y_TRUE, Y_PRED, GROUPS).iloc[0]
    assert row["dp_difference"] == pytest.approx(0.5)
    assert row["dp_ratio"] == pytest.approx(1 / 3)
    assert row["eo_difference"] == pytest.approx(0.5)
    assert not row["passes_four_fifths"]


def test_identical_treatment_passes_four_fifths():
    row = disparities(Y_TRUE, np.array([1, 0, 1, 0, 1, 0, 1, 0]), GROUPS).iloc[0]
    assert row["dp_ratio"] == pytest.approx(1.0)
    assert row["passes_four_fifths"]


def test_age_sensitivity_covers_every_threshold(cfg, data, models):
    proba = models["LogisticRegression"].predict_proba(data["X_test"])[:, 1]
    table = age_threshold_sensitivity(cfg, data["X_test"], data["y_test"], decide(proba, cfg))
    assert table["threshold"].tolist() == [25, 30, 40]
    assert table["n_young"].is_monotonic_increasing


def test_age_sensitivity_skipped_without_config(cfg, data):
    bare = {**cfg, "sensitive": {"attributes": ["sex"], "primary": "sex"}}
    assert age_threshold_sensitivity(bare, data["X_test"], data["y_test"], None) is None
