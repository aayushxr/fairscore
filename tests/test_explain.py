import numpy as np
import pytest

from src.explain.reasons import contributions, global_importance, group_gap, reason_codes
from src.models.scorecard import decide, scaling


@pytest.fixture(scope="module")
def contrib(cfg, data, models):
    return contributions(models["LogisticRegression"], data["X_test"], data["X_train"], cfg)


def test_contributions_sum_to_score(cfg, data, models, contrib):
    table, average_score = contrib
    factor, offset = scaling(cfg)
    score = offset + factor * models["LogisticRegression"].decision_function(data["X_test"])
    assert list(table.columns) == ["income", "age", "housing"]
    assert np.allclose(average_score + table.sum(axis=1), score)


def test_reason_codes_only_for_declined(cfg, data, models, contrib):
    proba = models["LogisticRegression"].predict_proba(data["X_test"])[:, 1]
    y_pred = decide(proba, cfg)
    reasons = reason_codes(contrib[0], y_pred, top_k=2)
    assert len(reasons) == int((y_pred == 0).sum())
    assert reasons["reason_1"].str.contains(r"\(-\d+\)").all()


def test_group_gap_sums_to_mean_score_gap(data, contrib):
    table = contrib[0]
    gap = group_gap(table, data["groups_test"])
    totals = table.sum(axis=1).groupby(data["groups_test"]["sex"]).mean()
    row = gap.iloc[0]
    expected = totals[row["group"]] - totals[row["reference"]]
    assert gap["points_gap"].sum() == pytest.approx(expected)
    # Income was built to differ by sex, so it should dominate the gap.
    assert gap.loc[gap["points_gap"].abs().idxmax(), "feature"] == "income"


def test_permutation_importance_finds_income(data, models):
    importance = global_importance(models, data["X_test"], data["y_test"], random_state=0)
    for _, sub in importance.groupby("model"):
        assert sub.loc[sub["auc_drop"].idxmax(), "feature"] == "income"
