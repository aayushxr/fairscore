"""Shared preprocessing, model training, and held-out evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(cfg: dict) -> ColumnTransformer:
    """Median impute + scale numerics; most-frequent impute + one-hot categoricals."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, cfg["features"]["numeric"]),
            ("cat", categorical, cfg["features"]["categorical"]),
        ]
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Post-encoding feature names without the num__/cat__ prefixes."""
    return [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]


def train_models(cfg: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Fit the scorecard model (logistic regression) and a gradient-boosting
    challenger on the same preprocessing. Sensitive attributes never enter X."""
    rs = cfg["split"]["random_state"]
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=rs),
        "GradientBoosting": GradientBoostingClassifier(random_state=rs),
    }
    return {
        name: Pipeline([("prep", make_preprocessor(cfg)), ("model", model)]).fit(X_train, y_train)
        for name, model in models.items()
    }


def evaluate(models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Held-out discrimination and calibration per model.

    KS is the max gap between the score CDFs of positives and negatives, the
    usual credit-scoring separation measure.
    """
    rows = []
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        rows.append(
            {
                "model": name,
                "auc": roc_auc_score(y_test, proba),
                "ks": float(np.max(tpr - fpr)),
                "brier": brier_score_loss(y_test, proba),
                "accuracy": accuracy_score(y_test, proba >= 0.5),
            }
        )
    return pd.DataFrame(rows)
