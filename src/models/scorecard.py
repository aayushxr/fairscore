"""Turn model probabilities into credit scores (points-to-double-odds scaling).

score = offset + factor * ln(odds), with odds = p / (1 - p) of the positive
outcome, factor = pdo / ln(2), and offset chosen so that `base_odds` maps to
`base_score`. Every `pdo` points doubles the odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.baseline import feature_names

EPS = 1e-6


def scaling(cfg: dict) -> tuple[float, float]:
    """Return (factor, offset) from the config's scoring block."""
    s = cfg["scoring"]
    factor = s["pdo"] / np.log(2)
    offset = s["base_score"] - factor * np.log(s["base_odds"])
    return float(factor), float(offset)


def to_score(proba: np.ndarray, cfg: dict) -> np.ndarray:
    """Map positive-outcome probabilities to scores, clipped to the score range."""
    factor, offset = scaling(cfg)
    p = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    score = offset + factor * np.log(p / (1 - p))
    return np.clip(score, cfg["scoring"]["min_score"], cfg["scoring"]["max_score"])


def cutoff_score(cfg: dict) -> float:
    """Approval cutoff. Defaults to base_score when the config gives none."""
    return float(cfg["scoring"].get("cutoff_score", cfg["scoring"]["base_score"]))


def cutoff_probability(cfg: dict) -> float:
    """The probability that maps to the cutoff score (inverse of to_score)."""
    factor, offset = scaling(cfg)
    return float(1 / (1 + np.exp(-(cutoff_score(cfg) - offset) / factor)))


def decide(proba: np.ndarray, cfg: dict) -> np.ndarray:
    """1 = approve (score at or above the cutoff), 0 = decline."""
    return (np.asarray(proba) >= cutoff_probability(cfg)).astype(int)


def points_table(pipe: Pipeline, cfg: dict) -> pd.DataFrame:
    """Points per encoded feature for the logistic-regression scorecard.

    Numerics are standardized, so their points are per standard deviation.
    The first row holds the base points (offset + intercept).
    """
    factor, offset = scaling(cfg)
    model = pipe.named_steps["model"]
    table = pd.DataFrame(
        {
            "feature": feature_names(pipe.named_steps["prep"]),
            "coefficient": model.coef_[0],
            "points": factor * model.coef_[0],
        }
    ).sort_values("points", key=np.abs, ascending=False)
    base = pd.DataFrame(
        [
            {
                "feature": "(base points)",
                "coefficient": float(model.intercept_[0]),
                "points": offset + factor * float(model.intercept_[0]),
            }
        ]
    )
    return pd.concat([base, table], ignore_index=True)
