"""Explanations for the scorecard.

The scorecard is linear in log-odds, so every explanation here is exact, not
an approximation. An applicant's contribution for a feature is the points
that feature adds relative to the average training applicant; one-hot columns
are summed back into their source feature. Contributions add up to the
applicant's unclipped score minus the average applicant's score.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.models.scorecard import scaling


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _source_features(pipe: Pipeline, cfg: dict) -> list[str]:
    """Source feature for every encoded column, in encoded order."""
    onehot = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    sources = list(cfg["features"]["numeric"])
    for feature, cats in zip(cfg["features"]["categorical"], onehot.categories_):
        sources += [feature] * len(cats)
    return sources


def contributions(
    pipe: Pipeline, X: pd.DataFrame, X_ref: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, float]:
    """Per-applicant, per-feature score points relative to the average row of X_ref.

    Returns (contributions, average_score) where average_score is the unclipped
    score of that average applicant.
    """
    factor, offset = scaling(cfg)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    coef = model.coef_[0]
    mean = _dense(prep.transform(X_ref)).mean(axis=0)
    points = factor * (_dense(prep.transform(X)) - mean) * coef
    contrib = pd.DataFrame(points, index=X.index, columns=_source_features(pipe, cfg))
    contrib = contrib.T.groupby(level=0, sort=False).sum().T
    average_score = offset + factor * (float(model.intercept_[0]) + float(mean @ coef))
    return contrib, average_score


def reason_codes(contrib: pd.DataFrame, y_pred: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    """For each declined applicant, the features that cost the most points."""
    declined = contrib[np.asarray(y_pred) == 0]
    rows = []
    for idx, row in declined.iterrows():
        worst = row.nsmallest(top_k)
        entry = {"applicant": idx, "points_vs_average": float(row.sum())}
        for rank, (feature, pts) in enumerate(worst.items(), start=1):
            entry[f"reason_{rank}"] = f"{feature} ({pts:+.0f})" if pts < 0 else ""
        rows.append(entry)
    return pd.DataFrame(rows)


def group_gap(contrib: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Which features drive the mean score gap between groups.

    For each attribute, every group is compared with the largest one. The
    per-feature values sum to that group's mean-score gap (before clipping).
    """
    rows = []
    for attr in groups.columns:
        means = contrib.groupby(groups[attr], observed=True).mean()
        reference = groups[attr].value_counts().idxmax()
        for group in means.index:
            if group == reference:
                continue
            gap = means.loc[group] - means.loc[reference]
            for feature, pts in gap.items():
                rows.append(
                    {
                        "attribute": attr,
                        "group": group,
                        "reference": reference,
                        "feature": feature,
                        "points_gap": float(pts),
                    }
                )
    return pd.DataFrame(rows)


def global_importance(
    models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series, random_state: int
) -> pd.DataFrame:
    """Permutation importance (drop in held-out AUC) per source feature and model."""
    tables = []
    for name, pipe in models.items():
        result = permutation_importance(
            pipe, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=random_state
        )
        tables.append(
            pd.DataFrame(
                {
                    "model": name,
                    "feature": X_test.columns,
                    "auc_drop": result.importances_mean,
                    "std": result.importances_std,
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def plot_global_importance(importance: pd.DataFrame, path: Path) -> None:
    """One panel per model, features sorted by AUC drop."""
    models = importance["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, name in zip(axes.ravel(), models):
        sub = importance[importance["model"] == name].sort_values("auc_drop")
        ax.barh(sub["feature"], sub["auc_drop"], xerr=sub["std"])
        ax.set_title(name)
        ax.set_xlabel("drop in AUC when permuted")
    fig.suptitle("Permutation importance on the test set")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_gap(gap: pd.DataFrame, attr: str, path: Path) -> None:
    """Per-feature score gap for each group of `attr` against its reference group."""
    sub = gap[gap["attribute"] == attr]
    pairs = sub["group"].unique()
    fig, axes = plt.subplots(len(pairs), 1, figsize=(8, 4 * len(pairs)), squeeze=False)
    for ax, group in zip(axes.ravel(), pairs):
        rows = sub[sub["group"] == group].sort_values("points_gap")
        colors = ["tab:red" if v < 0 else "tab:blue" for v in rows["points_gap"]]
        ax.barh(rows["feature"], rows["points_gap"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        total = rows["points_gap"].sum()
        reference = rows["reference"].iloc[0]
        ax.set_title(f"{group} vs {reference}: mean score gap {total:+.1f} points")
        ax.set_xlabel("points")
    fig.suptitle(f"Features behind the score gap by {attr}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
