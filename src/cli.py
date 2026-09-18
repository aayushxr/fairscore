"""Pipeline entry point.

Usage: python -m src.cli --config configs/german_credit.yaml

Stages: data loading + split, profiling (base rates, proxy detection), models
+ scorecard, fairness audit, mitigation, explanations. Figures land in
reports/figures/<dataset>/, tables in reports/tables/<dataset>/, and the
written report in reports/<dataset>.md.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.loader import REPO_ROOT, load_config, load_dataset, split_data
from src.data.profile import base_rates, plot_base_rates, plot_proxy_detection, proxy_detection
from src.explain.reasons import (
    contributions,
    global_importance,
    group_gap,
    plot_global_importance,
    plot_group_gap,
    reason_codes,
)
from src.fairness.audit import (
    age_threshold_sensitivity,
    disparities,
    group_metrics,
    plot_group_metrics,
    plot_score_distributions,
)
from src.fairness.mitigate import compare, mitigate, plot_tradeoff
from src.models.baseline import evaluate, train_models
from src.models.scorecard import cutoff_probability, cutoff_score, decide, points_table, to_score
from src.report import save_tables, write_report

SCORECARD = "LogisticRegression"
SMALL_GROUP = 100


def _caveats(cfg: dict, metrics: pd.DataFrame, proxies: list[dict]) -> list[str]:
    """Things a reader should know before trusting the numbers."""
    caveats = []
    small = metrics[(metrics["model"] == SCORECARD) & (metrics["count"] < SMALL_GROUP)]
    for _, row in small.iterrows():
        caveats.append(
            f"`{row['attribute']}={row['group']}` has {row['count']} test rows. "
            "Its rates move several points on a handful of applicants."
        )
    b = cfg["sensitive"].get("binarize_age")
    if b and b["column"] in cfg["features"]["numeric"]:
        proxies = [r for r in proxies if r["attribute"] != b["group_name"]]
        caveats.append(
            f"Raw `{b['column']}` is a model feature while `{b['group_name']}` is audited, "
            "so the model can condition on age directly. Its proxy AUC near 1.0 is expected."
        )
    leaky = [r for r in proxies if r["auc"] >= 0.7]
    if leaky:
        worst = max(leaky, key=lambda r: r["auc"])
        caveats.append(
            f"Holding sensitive columns out does not hide them. `{worst['attribute']}` is "
            f"recoverable from the features at AUC {worst['auc']:.2f}."
        )
    caveats.append(
        "Mitigated predictions are randomized. Rerunning with another random_state "
        "shifts the mitigation rows slightly."
    )
    return caveats


def main(argv: list[str] | None = None) -> None:
    """Run the pipeline for one dataset config."""
    parser = argparse.ArgumentParser(description="FairScore pipeline")
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    name = cfg["dataset"]["name"]
    primary = cfg["sensitive"]["primary"]
    rs = cfg["split"]["random_state"]
    reports_dir = REPO_ROOT / "reports"
    figures_dir = reports_dir / "figures" / name

    X, y, groups = load_dataset(cfg)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(cfg, X, y, groups)

    print(f"=== {name} ===")
    print(f"X: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"y: {int(y.sum())} positive / {int(len(y) - y.sum())} negative ({y.mean():.1%} positive)")
    print(f"groups: {list(groups.columns)} | train={len(X_train)} test={len(X_test)}")
    print()

    rates = base_rates(y, groups)
    plot_base_rates(rates, figures_dir / "base_rates.png")
    print("base rates:")
    print(rates.to_string(index=False))
    print()

    proxies = proxy_detection(cfg, X_train, X_test, groups_train, groups_test)
    plot_proxy_detection(proxies, figures_dir / "proxy_detection.png")
    print("proxy detection (held-out ROC-AUC):")
    for res in proxies:
        top3 = ", ".join(res["top_features"].index[:3])
        print(f"  {res['attribute']} <- {res['model']}: AUC {res['auc']:.3f} (top: {top3})")
    print()

    models = train_models(cfg, X_train, y_train)
    performance = evaluate(models, X_test, y_test)
    print("model performance (test):")
    print(performance.to_string(index=False, float_format="%.3f"))
    print()

    probas = {m: pipe.predict_proba(X_test)[:, 1] for m, pipe in models.items()}
    scores = {m: to_score(p, cfg) for m, p in probas.items()}
    decisions = {m: decide(p, cfg) for m, p in probas.items()}
    points = points_table(models[SCORECARD], cfg)

    metrics = pd.concat(
        [
            group_metrics(y_test, decisions[m], scores[m], groups_test).assign(model=m)
            for m in models
        ],
        ignore_index=True,
    )
    gaps = pd.concat(
        [disparities(y_test, decisions[m], groups_test).assign(model=m) for m in models],
        ignore_index=True,
    )
    gaps = gaps[["model"] + [c for c in gaps.columns if c != "model"]]
    plot_group_metrics(
        metrics[metrics["model"] == SCORECARD],
        figures_dir / "group_metrics.png",
        f"Scorecard decisions at cutoff {cutoff_score(cfg):.0f}",
    )
    plot_score_distributions(
        scores[SCORECARD], groups_test, cutoff_score(cfg), figures_dir / "score_distributions.png"
    )
    print("fairness audit (test):")
    print(gaps.to_string(index=False, float_format="%.3f"))
    print()

    age_sens = age_threshold_sensitivity(cfg, X_test, y_test, decisions[SCORECARD])
    if age_sens is not None:
        print("age threshold sensitivity (scorecard):")
        print(age_sens.to_string(index=False, float_format="%.3f"))
        print()

    mitigated = mitigate(
        cfg, models[SCORECARD], X_train, y_train, groups_train, X_test, groups_test
    )
    comparison = compare(cfg, y_test, {"Unmitigated": decisions[SCORECARD], **mitigated}, groups_test)
    plot_tradeoff(comparison, primary, figures_dir / "mitigation_tradeoff.png")
    print(f"mitigation, equalized odds on {primary} (test):")
    print(comparison.to_string(index=False, float_format="%.3f"))
    print()

    importance = global_importance(models, X_test, y_test, rs)
    plot_global_importance(importance, figures_dir / "global_importance.png")
    contrib, average_score = contributions(models[SCORECARD], X_test, X_train, cfg)
    gap = group_gap(contrib, groups_test)
    plot_group_gap(gap, primary, figures_dir / "group_gap.png")
    reasons = reason_codes(contrib, decisions[SCORECARD])
    top_gap = gap[gap["attribute"] == primary].sort_values("points_gap").head(3)
    print(f"average applicant scores {average_score:.0f}. biggest drags on the {primary} gap:")
    for _, row in top_gap.iterrows():
        print(f"  {row['group']} vs {row['reference']}: {row['feature']} {row['points_gap']:+.1f}")
    print()

    tables = {
        "base_rates": rates,
        "performance": performance,
        "scorecard_points": points,
        "group_metrics": metrics,
        "disparities": gaps,
        "age_sensitivity": age_sens,
        "mitigation": comparison,
        "global_importance": importance,
        "group_gap": gap,
        "reason_codes": reasons,
    }
    summary = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "positive_rate": float(y.mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "proxies": proxies,
        "cutoff_score": cutoff_score(cfg),
        "cutoff_probability": cutoff_probability(cfg),
        "caveats": _caveats(cfg, metrics, proxies),
    }
    save_tables(tables, reports_dir / "tables" / name)
    write_report(cfg, summary, tables, reports_dir / f"{name}.md")
    print(f"report: {reports_dir / f'{name}.md'}")
    print(f"figures: {figures_dir}")


if __name__ == "__main__":
    main()
