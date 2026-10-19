from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase12_builder import build_phase12, run_phase12_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 12: synthetic validation and publication bundle")
    p.add_argument("--config", default="config/phase12.yaml")
    p.add_argument("--skip-tests", action="store_true")
    return p.parse_args()


def run_pytest(root: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_path = (root / args.config).resolve()
    cfg = load_config(cfg_path)
    logger = configure_logging()
    logger.info("Phase 12 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase12_toy_validation()
    logger.info("Synthetic toy validation passed: %s", toy)
    outputs = build_phase12(cfg, root)
    logger.info("Synthetic runs written: %s", outputs.synthetic_runs.relative_to(root))
    logger.info("Synthetic summary written: %s", outputs.synthetic_summary.relative_to(root))
    logger.info("Claim/evidence map written: %s", outputs.claim_evidence.relative_to(root))
    logger.info("Publication manifest written: %s", outputs.publication_manifest.relative_to(root))
    logger.info("Publication bundle ZIP written: %s", outputs.publication_bundle_zip.relative_to(root))

    if cfg.phase12 and cfg.phase12.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files,
        outputs.synthetic_runs,
        outputs.synthetic_summary,
        outputs.claim_evidence,
        outputs.compact_selective_repair,
        outputs.compact_meta,
        outputs.publication_manifest,
        outputs.publication_bundle_zip,
        outputs.audit_json,
        *outputs.figure_files,
        *outputs.table_files,
    ]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase12.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(ctx, status="PASSED", extra={
        "phase12_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}
    })

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=12,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "synthetic_summary": str(outputs.synthetic_summary.relative_to(root)),
            "claim_evidence": str(outputs.claim_evidence.relative_to(root)),
            "publication_manifest": str(outputs.publication_manifest.relative_to(root)),
            "publication_bundle_zip": str(outputs.publication_bundle_zip.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": None,
        },
    )

    print("\nPHASE 12 PASSED")
    print(f"experiment_id:             {ctx.experiment_id}")
    print(f"synthetic_summary:         {outputs.synthetic_summary.relative_to(root)}")
    print(f"claim_evidence:            {outputs.claim_evidence.relative_to(root)}")
    print(f"publication_manifest:      {outputs.publication_manifest.relative_to(root)}")
    print(f"publication_bundle_dir:    {outputs.publication_bundle_dir.relative_to(root)}")
    print(f"publication_bundle_zip:    {outputs.publication_bundle_zip.relative_to(root)}")
    print(f"figures:                   {len(outputs.figure_files)}")
    print(f"table_artifacts:           {len(outputs.table_files)}")
    print(f"audit:                     {outputs.audit_json.relative_to(root)}")
    print(f"manifest:                  {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:                {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
