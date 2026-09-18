"""Group fairness audit of approve/decline decisions.

Metrics per group: selection rate (share approved), TPR, FPR, accuracy, mean
score. Disparities per attribute: demographic parity difference and ratio
(the ratio is the disparate-impact number behind the four-fifths rule) and
equalized odds difference (the larger of the TPR and FPR gaps).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score

FOUR_FIFTHS = 0.8


def group_metrics(
    y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray, groups: pd.DataFrame
) -> pd.DataFrame:
    """One tidy table: attribute, group, count, selection_rate, tpr, fpr, accuracy, mean_score."""
    metrics = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "accuracy": accuracy_score,
    }
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    scores = pd.Series(np.asarray(scores), index=y_true.index)
    tables = []
    for attr in groups.columns:
        frame = MetricFrame(
            metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=groups[attr]
        ).by_group
        frame.insert(0, "count", groups[attr].value_counts().reindex(frame.index))
        frame["mean_score"] = scores.groupby(groups[attr], observed=True).mean()
        frame = frame.rename_axis("group").reset_index()
        frame.insert(0, "attribute", attr)
        tables.append(frame)
    return pd.concat(tables, ignore_index=True)


def disparities(y_true: pd.Series, y_pred: np.ndarray, groups: pd.DataFrame) -> pd.DataFrame:
    """Per-attribute disparity summary, with the four-fifths rule verdict."""
    rows = []
    for attr in groups.columns:
        kwargs = {"y_true": y_true, "y_pred": y_pred, "sensitive_features": groups[attr]}
        ratio = demographic_parity_ratio(**kwargs)
        rows.append(
            {
                "attribute": attr,
                "dp_difference": demographic_parity_difference(**kwargs),
                "dp_ratio": ratio,
                "eo_difference": equalized_odds_difference(**kwargs),
                "passes_four_fifths": bool(ratio >= FOUR_FIFTHS),
            }
        )
    return pd.DataFrame(rows)


def age_threshold_sensitivity(
    cfg: dict, X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame | None:
    """Re-run the age-group disparities at other binarization thresholds.

    The young/old cut in the config is a judgement call; this shows whether
    the audit conclusion depends on it. None if the config has no such block.
    """
    b = cfg["sensitive"].get("binarize_age")
    if not b or "sensitivity_thresholds" not in b:
        return None
    age = X[b["column"]].astype(float)
    rows = []
    for t in b["sensitivity_thresholds"]:
        young = age.le(t).map({True: f"<={t}", False: f">{t}"}).rename(b["group_name"])
        row = disparities(y_true, y_pred, young.to_frame()).iloc[0].to_dict()
        row.pop("attribute")
        rows.append({"threshold": t, "n_young": int(age.le(t).sum()), **row})
    return pd.DataFrame(rows)


def plot_group_metrics(metrics: pd.DataFrame, path: Path, title: str) -> None:
    """Selection rate, TPR, and FPR per group, one row of panels per attribute."""
    attrs = metrics["attribute"].unique()
    cols = ["selection_rate", "tpr", "fpr"]
    fig, axes = plt.subplots(
        len(attrs), len(cols), figsize=(11, 2.6 * len(attrs)), squeeze=False, sharex=True
    )
    for row, attr in zip(axes, attrs):
        sub = metrics[metrics["attribute"] == attr]
        labels = [f"{g} (n={n:,})" for g, n in zip(sub["group"], sub["count"])]
        for ax, col in zip(row, cols):
            ax.barh(labels, sub[col])
            for i, v in enumerate(sub[col]):
                ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
            ax.set_xlim(0, 1.1)
            ax.set_title(f"{col} by {attr}", fontsize=10)
            ax.invert_yaxis()
            if ax is not row[0]:
                ax.set_yticklabels([])
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_distributions(
    scores: np.ndarray, groups: pd.DataFrame, cutoff: float, path: Path
) -> None:
    """Score histograms per group with the approval cutoff marked."""
    attrs = list(groups.columns)
    fig, axes = plt.subplots(len(attrs), 1, figsize=(8, 3 * len(attrs)), squeeze=False)
    bins = np.linspace(np.min(scores), np.max(scores), 40)
    for ax, attr in zip(axes.ravel(), attrs):
        for group, idx in groups.groupby(attr, observed=True).groups.items():
            vals = pd.Series(scores, index=groups.index).loc[idx]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, label=str(group))
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Score distribution by {attr} (dashed: cutoff {cutoff:.0f})")
        ax.set_xlabel("score")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
