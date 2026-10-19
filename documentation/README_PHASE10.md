# Phase 10 - Selective Repair and Meta-Policy Selector

Phase 10 is the first phase that **executes adaptive deployment actions**.

It implements the two frozen deployment algorithms from the research specification:

- **Algorithm A - Policy-Specific Selective Repair**: for each incumbent baseline policy, predict all policy continuation utilities from the current past-only state; choose the candidate with the largest calibrated lower bound over the baseline; use that candidate's next target only when `LCB > deploy_margin`, otherwise keep the baseline action.
- **Algorithm B - Regret-Aware Meta-Policy Selector**: maintain a current policy identity, rank all frozen policies by predicted continuation utility minus a fixed policy-switch penalty, and switch only when the calibrated best-vs-current lower bound exceeds the deployment margin.

A `naive_meta` no-LCB argmax branch is saved only as a diagnostic comparator. Ridge and GBT remain separate frozen branches; Phase 10 does not pick a winner from Phase-9 test results.

## Oracle/label isolation

The pseudo-live deployment path reads only:

1. Phase-1 canonical market data;
2. Phase-7 past-only features, fold membership, and train-only preprocessors;
3. Phase-8 frozen **utility model artifacts** and their model index;
4. Phase-9 **calibration-only** pairwise residual quantiles;
5. the frozen policy definitions.

It deliberately does **not** read Phase-5 Oracle outputs, Phase-6 labels, or Phase-8 test prediction tables. H=60 realized candidate advantages are computed only *after* deployment actions are written and hashed, and only for decision dates whose full horizon stays inside the same test block.

## Cost semantics

Phase-8 utility targets were created with the common execution/cost engine, so predicted continuation utilities already learn cost-inclusive outcomes as a function of current exposure. Phase 10 therefore does not subtract a second estimated transaction cost. Realized exposure turnover is charged by the shared execution engine. The meta-selector additionally uses one fixed 5 bp policy-identity stability penalty to discourage chattering; Phase 11 can vary this as a robustness check.

## Outputs

- `deployment_actions.parquet`
- `deployment_ledger.parquet`
- `deployment_fills.parquet`
- `strategy_performance.parquet`
- `decision_diagnostics.parquet`
- `selective_repair_oos.parquet`
- `selective_repair_summary.parquet`
- `meta_selection_oos.parquet`
- `meta_selection_summary.parquet`
- `continuous_actions.parquet`
- `continuous_ledger.parquet`
- `continuous_fills.parquet`
- `continuous_performance.parquet`
- `phase10_audit.json`

Fold-local paths reset to the same unit wealth / initial exposure. Continuous pseudo-live paths carry account state across annual OOS model updates.

## v0.10.1 calendar-boundary fix

Phase 10 diagnostics now resolve calendar fold endpoints such as `2016-12-31`
to the last realized trading date on or before that boundary. This prevents a
`KeyError` when a test-year endpoint falls on a weekend or market holiday. The
research contract, models, calibration, deployment logic, execution engine, and
metrics are unchanged. Empty fill frames are also excluded before concatenation
to avoid an unrelated pandas FutureWarning.
