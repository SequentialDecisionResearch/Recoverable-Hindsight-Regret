# Code snapshot — Recoverable Hindsight Regret

This folder is a **source-only snapshot**, extracted from the author's cumulative `small13` Phase 0–12 code archive. It contains the package modules, phase entry points, representative unit tests, dependency specification, and selected phase documentation. Data and generated artifacts are **not** included.

**Scope and limitation.** This is not yet a complete, one-command reproducibility release. In particular, the uploaded `scripts(20261019-171632).zip` did not contain the `config/phase*.yaml` runtime configurations referenced by its entry points, or the frozen dataset/checkpoint files needed to reproduce the paper tables. Do not state that the empirical results can be reproduced from this snapshot alone. Full reproducibility requires the frozen configuration files, recorded market-data snapshot or legally compliant download instructions, input manifests, checkpoints and validated steps. Archive metadata should not be described as formal pre-registration unless independently documented.

**Install development dependencies** (Python 3.11 or 3.12):

```bash
python -m pip install -e ".[dev]"
```

**Example tests**, once the environment is installed:

```bash
python -m pytest -q tests
```

The `scripts/run_phase*.py` entry points use `config/phase*.yaml` by default. These configurations are not part of this source-only delivery. Do not attempt to reproduce full experiments until you have restored and audited the original configs and input data.

The underlying publication study uses yfinance/Yahoo historical market records; please check terms of use and third-party redistribution rights before publishing archived price data. No market data are included here.
