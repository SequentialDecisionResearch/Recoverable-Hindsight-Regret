from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase9_builder import build_phase9, run_phase9_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 9: calibrated LCB and coverage-harm diagnostics")
    p.add_argument("--config", default="config/phase9.yaml")
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
    logger.info("Phase 9 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase9_toy_validation(seed=int(cfg.seed) % 100000)
    logger.info("Toy validation passed: %s", toy)
    outputs = build_phase9(cfg, root)
    logger.info("Pairwise calibrators written: %s", outputs.pairwise_calibrators.relative_to(root))
    logger.info("Evidence predictions written: %s", outputs.evidence_predictions.relative_to(root))
    logger.info("Coverage-harm frontier written: %s", outputs.coverage_harm_frontier.relative_to(root))
    logger.info("Coverage-harm summary written: %s", outputs.coverage_harm_summary.relative_to(root))
    logger.info("Calibration coverage written: %s", outputs.calibration_coverage.relative_to(root))

    if cfg.phase9 and cfg.phase9.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files, outputs.pairwise_calibrators, outputs.evidence_predictions,
        outputs.coverage_harm_frontier, outputs.coverage_harm_summary,
        outputs.calibration_coverage, outputs.audit_json, *outputs.csv_files,
    ]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase9.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(ctx, status="PASSED", extra={
        "phase9_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}
    })

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=9,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "pairwise_calibrators": str(outputs.pairwise_calibrators.relative_to(root)),
            "evidence_predictions": str(outputs.evidence_predictions.relative_to(root)),
            "coverage_harm_frontier": str(outputs.coverage_harm_frontier.relative_to(root)),
            "coverage_harm_summary": str(outputs.coverage_harm_summary.relative_to(root)),
            "calibration_coverage": str(outputs.calibration_coverage.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 10,
        },
    )

    print("\nPHASE 9 PASSED")
    print(f"experiment_id:          {ctx.experiment_id}")
    print(f"pairwise_calibrators:   {outputs.pairwise_calibrators.relative_to(root)}")
    print(f"evidence_predictions:   {outputs.evidence_predictions.relative_to(root)}")
    print(f"coverage_harm_frontier: {outputs.coverage_harm_frontier.relative_to(root)}")
    print(f"coverage_harm_summary:  {outputs.coverage_harm_summary.relative_to(root)}")
    print(f"calibration_coverage:   {outputs.calibration_coverage.relative_to(root)}")
    print(f"audit:                  {outputs.audit_json.relative_to(root)}")
    print(f"manifest:               {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:             {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
