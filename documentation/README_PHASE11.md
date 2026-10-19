# Phase 11 — Statistical Inference and Robustness

Phase 11 **does not fit a new model and does not tune Phase 8–10 choices**. It treats Phase-10 deployment outcomes as frozen and quantifies uncertainty around the realized OOS effects.

## Primary inferential object

The primary result is policy-specific selective repair. For every `fold × asset × baseline × model` path, Phase 11 pairs the adaptive ledger with the corresponding fixed-policy ledger on exactly the same trading dates. The same moving-block bootstrap indices are applied to both paths, so market shocks common to the pair cancel as much as possible.

Default bootstrap:

- circular moving-block bootstrap;
- block length = 20 trading days;
- 1,000 bootstrap resamples;
- 95% percentile confidence intervals;
- same-asset OOS and asset-OOS reported separately.

For each group Phase 11 reports uncertainty for:

- `log_utility_improvement`;
- `CAGR_delta`;
- `MaxDD_delta` (positive means drawdown became less negative / improved);
- `turnover_delta`;
- normalized transaction-cost delta;
- Sharpe delta.

It also reports `bootstrap_prob_gt_zero` and `bootstrap_prob_lt_zero`. These are bootstrap sign probabilities, **not classical p-values**.

## Decision-level inference

The frozen Phase-10 decision diagnostics are block-resampled by decision date to obtain confidence intervals for:

- deployment coverage;
- harm rate among interventions;
- mean realized H-day advantage among interventions;
- positive-advantage rate.

No Oracle or future label is used to change any deployment action. These are evaluation-only quantities after Phase 10 has already frozen the actions.

## Stability and continuous pseudo-live robustness

Phase 11 separately writes:

- annual stability by test year;
- asset stability across years;
- continuous 2016–2025 paired block-bootstrap inference using the non-reset pseudo-live ledgers;
- publication-ready selective-repair and meta stress-test tables.

The global meta-selector remains a secondary stress test. Phase 11 does not promote it based on Phase-10 test performance.

## Inputs

Phase 11 reads only frozen Phase-10 outputs:

- `deployment_ledger.parquet`
- `decision_diagnostics.parquet`
- `selective_repair_oos.parquet`
- `meta_selection_oos.parquet`
- `continuous_ledger.parquet`
- `continuous_performance.parquet`

It does not read Phase 5/6 hindsight Oracle or label files, and it does not retrain Phase 8 models.

## Outputs

- `paired_bootstrap_inference.parquet`
- `decision_bootstrap_inference.parquet`
- `continuous_bootstrap_inference.parquet`
- `annual_stability.parquet`
- `asset_stability.parquet`
- `meta_annual_stability.parquet`
- `publication_selective_repair.parquet`
- `publication_meta.parquet`
- `phase11_audit.json`

## Run

```bat
python -m pip install -e ".[dev]"
pytest -q
python scripts\run_phase11.py
```

If the Phase-10 processed data and checkpoint already exist in this clean project, only `run_phase11.py` is needed. If this is a freshly extracted directory with no processed data/checkpoints, run Phase 0 through Phase 10 first.
