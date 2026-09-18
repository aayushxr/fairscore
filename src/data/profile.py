"""Audit-oriented profiling: per-group base rates and proxy detection.

Proxy detection asks: can the sensitive attribute be predicted from the model
features alone? Held-out AUC well above 0.5 means the attribute survives in
the features as a proxy, so dropping the column doesn't remove it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.models.baseline import feature_names, make_preprocessor

TOP_N = 10


def base_rates(y: pd.Series, groups: pd.DataFrame) -> pd.DataFrame:
    """Positive-outcome rate and group size for every group of every
    sensitive attribute. One tidy table: attribute, group, count, positive_rate."""
    rows = []
    for attr in groups.columns:
        stats = y.groupby(groups[attr], observed=True).agg(count="size", positive_rate="mean")
        for group, row in stats.sort_index().iterrows():
            rows.append(
                {
                    "attribute": attr,
                    "group": group,
                    "count": int(row["count"]),
                    "positive_rate": float(row["positive_rate"]),
                }
            )
    return pd.DataFrame(rows)


def proxy_detection(
    cfg: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    groups_train: pd.DataFrame,
    groups_test: pd.DataFrame,
) -> list[dict]:
    """Fit models predicting each sensitive attribute from the features.

    Returns one dict per (attribute, model) with the held-out ROC-AUC and the
    top 10 features (|coefficient| for logistic regression, impurity
    importance for the random forest).
    """
    rs = cfg["split"]["random_state"]
    results = []
    for attr in groups_train.columns:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, random_state=rs),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
        }
        for model_name, model in models.items():
            pipe = Pipeline([("prep", make_preprocessor(cfg)), ("model", model)])
            pipe.fit(X_train, groups_train[attr])
            proba = pipe.predict_proba(X_test)
            classes = pipe.named_steps["model"].classes_
            if len(classes) == 2:
                auc = roc_auc_score(groups_test[attr], proba[:, 1])
            else:  # race is multi-class: macro one-vs-rest AUC
                auc = roc_auc_score(
                    groups_test[attr], proba, multi_class="ovr", labels=classes
                )
            if model_name == "LogisticRegression":
                importance = np.abs(model.coef_).mean(axis=0)
            else:
                importance = model.feature_importances_
            top = pd.Series(importance, index=feature_names(pipe.named_steps["prep"]))
            results.append(
                {
                    "attribute": attr,
                    "model": model_name,
                    "auc": float(auc),
                    "top_features": top.nlargest(TOP_N),
                }
            )
    return results


def plot_base_rates(rates: pd.DataFrame, path: Path) -> None:
    """Bar chart of positive rate per group, one panel per sensitive attribute."""
    attrs = rates["attribute"].unique()
    fig, axes = plt.subplots(len(attrs), 1, figsize=(7, 2.5 * len(attrs)), squeeze=False)
    for ax, attr in zip(axes.ravel(), attrs):
        sub = rates[rates["attribute"] == attr]
        labels = [f"{g} (n={n:,})" for g, n in zip(sub["group"], sub["count"])]
        ax.barh(labels, sub["positive_rate"])
        for i, rate in enumerate(sub["positive_rate"]):
            ax.text(rate + 0.01, i, f"{rate:.1%}", va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("positive-outcome rate")
        ax.set_title(f"Base rate by {attr}")
        ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_proxy_detection(results: list[dict], path: Path) -> None:
    """Top proxy features per (attribute, model), AUC in each panel title."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        top = res["top_features"]
        ax.barh(top.index, top.values)
        ax.invert_yaxis()
        ax.set_title(f"predict {res['attribute']} with {res['model']} — AUC {res['auc']:.3f}")
        xlabel = "|coefficient|" if res["model"] == "LogisticRegression" else "importance"
        ax.set_xlabel(xlabel)
    fig.suptitle("Proxy detection: sensitive attributes predicted from features")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
