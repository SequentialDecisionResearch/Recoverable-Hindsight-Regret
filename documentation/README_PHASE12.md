# Phase 12 - Synthetic Ground Truth and Publication Bundle

Phase 12 is the terminal experiment-packaging stage. It **does not change any real-data model, feature, gate, threshold, policy, or execution rule from Phase 0-11**.

## Required frozen inputs

Core Phase-12 publication generation requires the frozen Phase-10/11 files below:

- `data/processed/phase11/publication_selective_repair.parquet`
- `data/processed/phase11/publication_meta.parquet`
- `data/processed/phase11/annual_stability.parquet`
- `data/processed/phase11/asset_stability.parquet`
- `data/processed/phase11/continuous_bootstrap_inference.parquet`
- `data/processed/phase10/continuous_ledger.parquet`
- `data/processed/phase10/selective_repair_oos.parquet`
- a `checkpoints/phase11_*.json` checkpoint.

Two earlier frozen files are optional. If present, Phase 12 also creates the corresponding extended paper figures:

- `data/processed/phase9/coverage_harm_summary.parquet` -> coverage-harm frontier figures;
- `data/processed/phase8/prediction_deciles.parquet` -> predicted-vs-realized regret decile figures.

A full Phase 0-12 run naturally creates them. If only Phase-10/11 outputs are copied into this clean directory, the core Phase-12 bundle still runs and records the skipped optional figures in its manifest.

## Synthetic ground-truth experiment

Two predeclared worlds are simulated:

1. **Recoverable world** - persistent trend/mean-reversion/noise/jump regimes and a noisy regime proxy available at time `t`. As feature signal strength increases, the deployable Student should recover more of the hindsight-positive opportunity.
2. **Clairvoyance-only world** - future return/jump innovations are independent of all time-`t` features. Hindsight still sees positive opportunities, but the calibrated Student should mostly abstain and recover approximately zero deployable gain.

The synthetic Student is a Ridge model used **only** for this controlled validation. It never touches or replaces the frozen real-data Phase-8 models.

## Main outputs

Processed outputs:

- `data/processed/phase12/synthetic_runs.parquet`
- `data/processed/phase12/synthetic_summary.parquet`
- `data/processed/phase12/claim_evidence.parquet`
- `data/processed/phase12/publication_selective_repair_compact.parquet`
- `data/processed/phase12/publication_meta_compact.parquet`
- `data/processed/phase12/phase12_audit.json`

Publication bundle:

- `outputs/publication_bundle_phase12/README.md`
- `outputs/publication_bundle_phase12/tables/*.csv`
- `outputs/publication_bundle_phase12/tables/*.tex`
- `outputs/publication_bundle_phase12/figures/*.png`
- `outputs/publication_bundle_phase12/figures/*.pdf`
- `outputs/publication_bundle_phase12/table_index.csv`
- `outputs/publication_bundle_phase12/figure_index.csv`
- `outputs/publication_bundle_phase12/publication_manifest.json`
- `outputs/publication_bundle_phase12.zip`

The manifest hashes all publication artifacts and frozen inputs used by Phase 12.

## Interpretation contract

- A positive decision-level continuation-advantage CI supports **local recoverability**, not universal profitable repair.
- Path-level and continuous wealth CIs remain separate evidence layers.
- Bootstrap sign probabilities are not classical p-values.
- Negative controls and failed policy/model combinations remain in the tables and figures.
- The synthetic experiment is a leakage/identifiability sanity check, not evidence of market profitability.
- With the current config `publication_data_quality_required: false`, the bundle is publication-formatted but the market data remain explicitly flagged as exploratory rather than certified point-in-time institutional data.


## v0.12.1 runtime dependency fix

Phase 12 writes LaTeX publication tables through `pandas.DataFrame.to_latex()`, whose Styler backend requires Jinja2 in current pandas releases. `Jinja2>=3.1,<4` is now an explicit runtime dependency, so `python -m pip install -e ".[dev]"` installs it automatically. This changes no research logic, model, threshold, synthetic design, or inference result.
