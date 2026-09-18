"""Write the per-dataset audit report (markdown) and its CSV tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""

    def fmt(value) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "|" + "|".join(" --- " for _ in df.columns) + "|",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_tables(tables: dict[str, pd.DataFrame | None], tables_dir: Path) -> None:
    """One CSV per table, skipping stages that produced nothing."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table is not None:
            table.to_csv(tables_dir / f"{name}.csv", index=False)


def write_report(cfg: dict, summary: dict, tables: dict[str, pd.DataFrame | None], path: Path) -> None:
    """Assemble the markdown report. Figure links are relative to reports/."""
    name = cfg["dataset"]["name"]
    primary = cfg["sensitive"]["primary"]
    fig = f"figures/{name}"
    proxies = pd.DataFrame(
        [
            {
                "attribute": r["attribute"],
                "model": r["model"],
                "auc": r["auc"],
                "top_features": ", ".join(r["top_features"].index[:3]),
            }
            for r in summary["proxies"]
        ]
    )

    parts = [
        f"# FairScore audit: {name}",
        f"{summary['n_rows']:,} rows, {summary['n_features']} features, "
        f"{summary['positive_rate']:.1%} positive. Train {summary['n_train']:,}, "
        f"test {summary['n_test']:,}. Sensitive attributes "
        f"({', '.join(cfg['sensitive']['attributes'])}) are held out of the features. "
        "All numbers below come from the test set unless stated.",
        "## 1. Base rates",
        "Share of positive outcomes in the raw labels, before any model.",
        md_table(tables["base_rates"]),
        f"![base rates]({fig}/base_rates.png)",
        "## 2. Proxy detection",
        "Held-out AUC for predicting each sensitive attribute from the model features. "
        "0.5 means the features carry no trace of it. Anything well above that means "
        "dropping the column did not remove the information.",
        md_table(proxies),
        f"![proxy detection]({fig}/proxy_detection.png)",
        "## 3. Models",
        md_table(tables["performance"]),
        f"Scores use {cfg['scoring']['pdo']} points to double the odds, with "
        f"{cfg['scoring']['base_score']} at odds {cfg['scoring']['base_odds']}:1. "
        f"Applicants at or above {summary['cutoff_score']:.0f} "
        f"(probability {summary['cutoff_probability']:.2f}) are approved.",
        "Largest scorecard weights (numeric points are per standard deviation):",
        md_table(tables["scorecard_points"].head(11), digits=2),
        "## 4. Fairness audit",
        "`dp_ratio` is the lowest group approval rate divided by the highest. Below 0.8 "
        "fails the four-fifths rule. `eo_difference` is the larger of the TPR and FPR gaps.",
        md_table(tables["disparities"]),
        "Per-group detail for the scorecard:",
        md_table(tables["group_metrics"][tables["group_metrics"]["model"] == "LogisticRegression"].drop(columns="model")),
        f"![group metrics]({fig}/group_metrics.png)",
        f"![score distributions]({fig}/score_distributions.png)",
    ]
    if tables.get("age_sensitivity") is not None:
        b = cfg["sensitive"]["binarize_age"]
        parts += [
            "### Age threshold sensitivity",
            f"The audit splits age at {b['threshold']}. The same scorecard decisions, "
            "regrouped at other cut points:",
            md_table(tables["age_sensitivity"]),
        ]
    parts += [
        "## 5. Mitigation",
        f"Both methods target equalized odds on `{primary}`. ThresholdOptimizer needs "
        f"`{primary}` at decision time. ExponentiatedGradient needs it only in training.",
        md_table(tables["mitigation"]),
        f"![mitigation trade-off]({fig}/mitigation_tradeoff.png)",
        "## 6. Explanations",
        f"![permutation importance]({fig}/global_importance.png)",
        f"Features behind the mean score gap on `{primary}`. The bars sum to the gap.",
        f"![group gap]({fig}/group_gap.png)",
        "Sample reason codes for declined applicants (points lost against the average "
        "training applicant):",
        md_table(tables["reason_codes"].head(10), digits=1),
        "## Caveats",
        "\n".join(f"- {c}" for c in summary["caveats"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(parts) + "\n")
