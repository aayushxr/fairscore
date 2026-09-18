"""Load datasets from OpenML per a YAML config.

Sensitive attributes are held out of the feature matrix ("fairness through
unawareness") and returned separately as `groups`, for auditing only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

Frames = tuple[pd.DataFrame, pd.Series, pd.DataFrame]
SplitFrames = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]


def load_config(path: str | Path) -> dict:
    """Read a YAML config and sanity-check it."""
    cfg = yaml.safe_load(Path(path).read_text())
    features = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    sensitive = cfg["sensitive"]["attributes"]
    assert not set(features) & set(sensitive), "sensitive attributes must not be features"
    assert cfg["target"]["column"] not in features, "target must not be a feature"
    assert cfg["sensitive"]["primary"] in sensitive
    assert 0 < cfg["split"]["test_size"] < 1
    return cfg


def _fetch_raw(cfg: dict) -> pd.DataFrame:
    """Fetch the raw frame from OpenML, caching to data/raw/ as parquet.

    Later runs are offline. A CSV copy is kept alongside for eyeballing.
    """
    cache = RAW_DIR / f"{cfg['dataset']['name']}.parquet"
    if cache.exists():
        frame = pd.read_parquet(cache)
    else:
        frame = fetch_openml(
            cfg["dataset"]["openml_name"],
            version=cfg["dataset"]["openml_version"],
            as_frame=True,
            parser="auto",
        ).frame
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache)
    csv_copy = cache.with_suffix(".csv")
    if not csv_copy.exists():
        frame.to_csv(csv_copy, index=False)
    return frame


def _to_nan(frame: pd.DataFrame, na_values: list[str]) -> pd.DataFrame:
    """Convert sentinel strings like "?" to NaN."""
    frame = frame.copy()
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            present = [v for v in na_values if v in frame[col].cat.categories]
            frame[col] = frame[col].cat.remove_categories(present)
        elif not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].where(~frame[col].isin(na_values))
    return frame


def load_dataset(cfg: dict) -> Frames:
    """Return (X, y, groups) for one dataset config.

    X = model features only, y = 1 for the positive outcome, groups = the
    sensitive attributes (same row order as X), used only for auditing.
    """
    frame = _fetch_raw(cfg)

    # Fail loudly if the download doesn't match what the config expects.
    expected = cfg["dataset"]["expected_rows"]
    assert len(frame) == expected, f"expected {expected} rows, got {len(frame)}"
    features = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    missing = set(features + [cfg["target"]["column"]]) - set(frame.columns)
    assert not missing, f"columns missing from raw data: {sorted(missing)}"

    if cfg.get("na_values"):
        frame = _to_nan(frame, cfg["na_values"])

    # Build the audit groups. Sex may need deriving (German Credit encodes it
    # inside personal_status, e.g. "male single" / "female div/dep/mar").
    groups = pd.DataFrame(index=frame.index)
    sens = cfg["sensitive"]
    if "derive_sex_from" in sens:
        sex = frame[sens["derive_sex_from"]].astype(str).str.split().str[0]
        assert set(sex.unique()) <= {"male", "female"}, f"bad sex values: {set(sex.unique())}"
        groups["sex"] = sex
    if "binarize_age" in sens:
        b = sens["binarize_age"]
        t = b["threshold"]
        groups[b["group_name"]] = frame[b["column"]].astype(float).le(t).map(
            {True: f"<={t}", False: f">{t}"}
        )
    for attr in sens["attributes"]:
        if attr not in groups.columns:
            groups[attr] = frame[attr].astype(str)
    groups = groups[sens["attributes"]]

    target = frame[cfg["target"]["column"]].astype(str)
    positive = str(cfg["target"]["positive_label"])
    assert positive in set(target), f"positive label {positive!r} not in target values"
    y = (target == positive).astype(int).rename("y")

    X = frame[features].copy()
    assert not set(X.columns) & set(sens["attributes"]), "sensitive attribute leaked into X"
    return X, y, groups


def split_data(cfg: dict, X: pd.DataFrame, y: pd.Series, groups: pd.DataFrame) -> SplitFrames:
    """80/20 split, stratified jointly on (target, sensitive attributes) so
    small groups keep their share of the test set."""
    key = y.astype(str)
    for attr in groups.columns:
        key = key + "|" + groups[attr]
    idx_train, idx_test = train_test_split(
        X.index,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=key,
    )
    return (
        X.loc[idx_train], X.loc[idx_test],
        y.loc[idx_train], y.loc[idx_test],
        groups.loc[idx_train], groups.loc[idx_test],
    )
