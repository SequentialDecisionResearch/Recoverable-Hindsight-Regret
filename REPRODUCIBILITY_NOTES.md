# Archived manuscript configurations and result tables

The archive includes 14 YAML configurations (Phase 0--12 and policies) and seven aggregate CSV result tables from the author-supplied October 2026 manuscript bundle. They have not been regenerated in this repository.

- `config/`: runnable configuration path expected by `scripts/run_phase*.py` and by phase configurations referring to `config/policies.yaml`.
- `configs/`: optional archival duplicate of the supplied YAML files; if retained, do not treat it as the default path or edit it independently.
- `results/`: seven archived aggregate CSV summaries used in the manuscript.
- `src/`, `scripts/`, and `tests/`: implementation and selected checks.

The repository deliberately excludes raw vendor prices, full frozen intermediate data and checkpoints. Independent complete regeneration of the published results has **not** been established. Vendor historical records may be revised; the implementation uses a price-taking execution model, limited equity universe, and an approximate dependence bootstrap.

No proof of formal prospective pre-registration is provided by this archive. Terms such as "frozen" and "registered" refer to the reported study specification and should not be interpreted as independently documented preregistration.

For synthetic-validation and publication pipeline input requirements, see `documentation/README_PHASE12.md`.
