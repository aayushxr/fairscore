import copy

import pytest
import yaml

from src.data.loader import RAW_DIR, REPO_ROOT, load_config, load_dataset


def _write(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("name", ["german_credit", "adult_census"])
def test_shipped_configs_are_valid(name):
    cfg = load_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert cfg["dataset"]["name"] == name


def test_config_rejects_sensitive_feature(tmp_path, cfg):
    bad = copy.deepcopy(cfg)
    bad["target"] = {"column": "y"}
    bad["features"]["categorical"].append("sex")
    with pytest.raises(AssertionError, match="sensitive"):
        load_config(_write(tmp_path, bad))


def test_split_keeps_group_shares(data):
    train = data["groups_train"]["sex"].value_counts(normalize=True)
    test = data["groups_test"]["sex"].value_counts(normalize=True)
    assert (train - test).abs().max() < 0.02
    assert abs(data["y_train"].mean() - data["y_test"].mean()) < 0.02


@pytest.mark.skipif(
    not (RAW_DIR / "german_credit.parquet").exists(), reason="German Credit not cached locally"
)
def test_german_credit_loads_without_leaking_sex():
    cfg = load_config(REPO_ROOT / "configs" / "german_credit.yaml")
    X, y, groups = load_dataset(cfg)
    assert len(X) == len(y) == len(groups) == 1000
    assert "personal_status" not in X.columns
    assert set(groups["sex"]) == {"male", "female"}
    assert set(groups["age_group"]) == {"<=25", ">25"}
    assert y.mean() == pytest.approx(0.7)
