"""Synthetic dataset shared by the tests, so nothing touches OpenML."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import split_data
from src.models.baseline import train_models


@pytest.fixture(scope="session")
def cfg() -> dict:
    return {
        "dataset": {"name": "synthetic"},
        "sensitive": {
            "attributes": ["sex"],
            "primary": "sex",
            "binarize_age": {
                "column": "age",
                "threshold": 30,
                "group_name": "age_group",
                "sensitivity_thresholds": [25, 30, 40],
            },
        },
        "features": {"numeric": ["income", "age"], "categorical": ["housing"]},
        "split": {"test_size": 0.25, "random_state": 0},
        "scoring": {
            "pdo": 50,
            "base_score": 600,
            "base_odds": 1.0,
            "cutoff_score": 600,
            "min_score": 300,
            "max_score": 850,
        },
    }


@pytest.fixture(scope="session")
def data(cfg: dict) -> dict:
    """600 applicants where income drives the outcome and correlates with sex."""
    rng = np.random.default_rng(0)
    n = 600
    sex = rng.choice(["female", "male"], size=n)
    income = rng.normal(50, 10, n) + np.where(sex == "male", 6, 0)
    X = pd.DataFrame(
        {
            "income": income,
            "age": rng.integers(18, 70, n).astype(float),
            "housing": rng.choice(["own", "rent", "free"], size=n),
        }
    )
    logit = (income - 53) / 5 + np.where(X["housing"] == "own", 0.8, 0.0)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="y")
    groups = pd.DataFrame({"sex": sex})
    names = ["X_train", "X_test", "y_train", "y_test", "groups_train", "groups_test"]
    return dict(zip(names, split_data(cfg, X, y, groups)))


@pytest.fixture(scope="session")
def models(cfg: dict, data: dict) -> dict:
    return train_models(cfg, data["X_train"], data["y_train"])
