import numpy as np

from src.fairness.mitigate import compare, mitigate
from src.models.baseline import evaluate
from src.models.scorecard import decide


def test_models_beat_chance(data, models):
    perf = evaluate(models, data["X_test"], data["y_test"])
    assert set(perf["model"]) == {"LogisticRegression", "GradientBoosting"}
    assert (perf["auc"] > 0.7).all()
    assert perf["brier"].between(0, 0.25).all()


def test_mitigation_returns_binary_decisions_for_every_method(cfg, data, models):
    pipe = models["LogisticRegression"]
    mitigated = mitigate(
        cfg, pipe, data["X_train"], data["y_train"], data["groups_train"],
        data["X_test"], data["groups_test"],
    )
    baseline = decide(pipe.predict_proba(data["X_test"])[:, 1], cfg)
    table = compare(cfg, data["y_test"], {"Unmitigated": baseline, **mitigated}, data["groups_test"])
    assert table["method"].tolist() == ["Unmitigated", "ThresholdOptimizer", "ExponentiatedGradient"]
    for y_pred in mitigated.values():
        assert len(y_pred) == len(data["y_test"])
        assert set(np.unique(y_pred)) <= {0, 1}
    assert table["accuracy"].min() > 0.6
