"""Bias mitigation for the scorecard model, targeting equalized odds on the
primary sensitive attribute.

Two fairlearn methods, compared against the unmitigated decisions:

- ThresholdOptimizer: keeps the fitted model, picks group-specific thresholds.
  It needs the sensitive attribute at decision time, which many lenders cannot
  legally use. Reported as a bound on what post-processing can reach.
- ExponentiatedGradient: retrains under a fairness constraint. It uses the
  attribute during training only; decisions need just the features.

Both predict with randomization, so results carry the config's random_state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.fairness.audit import disparities


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def mitigate(
    cfg: dict,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups_test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return test-set decisions per mitigation method for the primary attribute."""
    rs = cfg["split"]["random_state"]
    attr = cfg["sensitive"]["primary"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt_train, Xt_test = _dense(prep.transform(X_train)), _dense(prep.transform(X_test))

    threshold = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    threshold.fit(Xt_train, y_train, sensitive_features=groups_train[attr])

    reduction = ExponentiatedGradient(
        LogisticRegression(max_iter=2000, random_state=rs),
        constraints=EqualizedOdds(),
        eps=0.02,
    )
    reduction.fit(Xt_train, y_train, sensitive_features=groups_train[attr])

    return {
        "ThresholdOptimizer": np.asarray(
            threshold.predict(Xt_test, sensitive_features=groups_test[attr], random_state=rs)
        ),
        "ExponentiatedGradient": np.asarray(reduction.predict(Xt_test, random_state=rs)),
    }


def compare(
    cfg: dict, y_true: pd.Series, decisions: dict[str, np.ndarray], groups: pd.DataFrame
) -> pd.DataFrame:
    """Accuracy and primary-attribute disparities per method (first = unmitigated)."""
    attr = cfg["sensitive"]["primary"]
    rows = []
    for method, y_pred in decisions.items():
        row = disparities(y_true, y_pred, groups[[attr]]).iloc[0].to_dict()
        row.pop("attribute")
        rows.append(
            {
                "method": method,
                "accuracy": accuracy_score(y_true, y_pred),
                "approval_rate": float(np.mean(y_pred)),
                **row,
            }
        )
    return pd.DataFrame(rows)


def plot_tradeoff(comparison: pd.DataFrame, attr: str, path: Path) -> None:
    """Accuracy against equalized-odds gap, one point per method."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(comparison["eo_difference"], comparison["accuracy"], s=60)
    for _, row in comparison.iterrows():
        ax.annotate(
            row["method"],
            (row["eo_difference"], row["accuracy"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel(f"equalized odds difference on {attr} (lower is fairer)")
    ax.set_ylabel("accuracy")
    ax.set_title("Mitigation trade-off")
    ax.margins(0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
