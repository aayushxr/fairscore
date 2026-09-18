# FairScore

A credit-scoring pipeline with a fairness audit attached. It trains a
scorecard on public data, scores applicants on a 300-850 scale, and then
checks how approvals, errors, and scores differ across sex, age, and race.

Two datasets ship with configs: German Credit (1,000 rows) and Adult Census
(48,842 rows, income above 50K standing in for "good" outcome).

## Setup

Python 3.11.

```
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```
.venv/bin/python -m src.cli --config configs/german_credit.yaml
.venv/bin/python -m src.cli --config configs/adult_census.yaml
.venv/bin/python -m pytest
```

The first run downloads the dataset from OpenML and caches it in `data/raw/`.
Later runs are offline. German Credit takes about 5 seconds, Adult about 25.

Each run writes `reports/<dataset>.md`, figures under
`reports/figures/<dataset>/`, and CSV tables under `reports/tables/<dataset>/`.

## What the pipeline does

1. **Load and split.** Sensitive attributes are removed from the features and
   kept aside as audit groups. German Credit hides sex inside
   `personal_status`, so the loader parses it out and drops that column. The
   80/20 split is stratified on outcome and sensitive groups together.
2. **Profile.** Base rates per group, then proxy detection: models try to
   predict each sensitive attribute from the remaining features. On Adult,
   sex comes back at AUC 0.93, mostly through `relationship`.
3. **Model and score.** Logistic regression is the scorecard. Gradient
   boosting runs beside it as a challenger. Probabilities become scores with
   points-to-double-odds scaling (50 points doubles the odds, 600 at 1:1).
   Applicants at or above `cutoff_score` are approved.
4. **Audit.** Selection rate, TPR, FPR, accuracy, and mean score per group.
   Per attribute: demographic parity difference and ratio (the four-fifths
   rule number) and equalized odds difference. German Credit also re-runs the
   age audit at several cut points, since splitting at 25 is a judgement call.
5. **Mitigate.** Two fairlearn methods target equalized odds on the primary
   attribute. ThresholdOptimizer sets per-group thresholds and needs the
   attribute at decision time. ExponentiatedGradient retrains under the
   constraint and needs it only in training.
6. **Explain.** Permutation importance for both models. For the scorecard,
   exact per-applicant point contributions, reason codes for declined
   applicants, and a breakdown of which features produce the score gap
   between groups.

## Layout

```
configs/          one YAML per dataset: columns, sensitive attributes, scoring
src/data/         loader.py (fetch, split), profile.py (base rates, proxies)
src/models/       baseline.py (preprocessing, training), scorecard.py (scaling)
src/fairness/     audit.py (group metrics), mitigate.py (fairlearn methods)
src/explain/      reasons.py (contributions, reason codes, group gaps)
src/report.py     markdown report and CSV tables
src/cli.py        runs every stage for one config
tests/            run on synthetic data, no network needed
```

## Adding a dataset

Copy a config, point `openml_name` at the dataset, and list the numeric,
categorical, and sensitive columns. `load_config` refuses a config that lists
a sensitive attribute as a feature.

## Limits

- Test-set groups get small. Adult has 81 rows for `race=Other`, German Credit
  has 38 applicants aged 25 or under. Read those rates as rough.
- German Credit keeps raw `age` as a feature while auditing `age_group`. The
  model can use age directly. That is a config choice, not an oversight.
- Scores are clipped to 300-850, which piles confident approvals up at 850.
- Adult is an income dataset, not a lending one. It is here because its group
  disparities are large and well studied.
