# Code and reproducibility scope

This repository contains the Phase 0--12 Python research sources, representative tests, archived YAML configurations, a manuscript PDF, and aggregate CSV result tables for *Recoverable Hindsight Regret*.

**Configuration location.** The runnable entry points use `config/phase*.yaml` (singular `config`) by default, and several phases refer to `config/policies.yaml`. This directory contains copies of the manuscript-supplied YAML configuration files. The older `configs/` directory, if present, is an archival duplicate; **use `config/` for running scripts**. Keep any future edits synchronized.

**Scope.** The repository is **not an independently verified, end-to-end numerical reproduction package**. Market-data snapshots, per-run manifests, intermediate processed Parquet files, and trained/frozen checkpoints are not included. Reported tables are archived publication outputs, not evidence of a fresh run. Vendor records may change when downloaded again.

**Install environment** (Python 3.11 or 3.12):

```bash
python -m pip install -e ".[dev]"
```

**Optional tests**, after installing dependencies:

```bash
python -m pytest -q tests
```

The full Phase 0--12 pipeline requires additional approved inputs and checkpoints; do not describe it as one-command reproducible or claim any tests have passed unless they are actually run and verified. Raw yfinance/Yahoo market data are not included and should not be redistributed without checking vendor terms.

See `REPRODUCIBILITY_NOTES.md` and `documentation/README_PHASE12.md` for additional requirements.
