from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase11_builder import build_phase11, run_phase11_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 11: statistical inference and robustness")
    p.add_argument("--config", default="config/phase11.yaml")
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
    logger.info("Phase 11 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase11_toy_validation()
    logger.info("Toy validation passed: %s", toy)
    outputs = build_phase11(cfg, root)
    logger.info("Paired bootstrap inference written: %s", outputs.paired_bootstrap_inference.relative_to(root))
    logger.info("Decision bootstrap inference written: %s", outputs.decision_bootstrap_inference.relative_to(root))
    logger.info("Continuous inference written: %s", outputs.continuous_bootstrap_inference.relative_to(root))
    logger.info("Annual stability written: %s", outputs.annual_stability.relative_to(root))
    logger.info("Publication selective-repair table written: %s", outputs.publication_selective_repair.relative_to(root))
    logger.info("Publication meta table written: %s", outputs.publication_meta.relative_to(root))

    if cfg.phase11 and cfg.phase11.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files,
        outputs.paired_bootstrap_inference,
        outputs.decision_bootstrap_inference,
        outputs.continuous_bootstrap_inference,
        outputs.annual_stability,
        outputs.asset_stability,
        outputs.meta_annual_stability,
        outputs.publication_selective_repair,
        outputs.publication_meta,
        outputs.audit_json,
        *outputs.csv_files,
    ]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase11.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(ctx, status="PASSED", extra={
        "phase11_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}
    })

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=11,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "paired_bootstrap_inference": str(outputs.paired_bootstrap_inference.relative_to(root)),
            "decision_bootstrap_inference": str(outputs.decision_bootstrap_inference.relative_to(root)),
            "continuous_bootstrap_inference": str(outputs.continuous_bootstrap_inference.relative_to(root)),
            "publication_selective_repair": str(outputs.publication_selective_repair.relative_to(root)),
            "publication_meta": str(outputs.publication_meta.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 12,
        },
    )

    print("\nPHASE 11 PASSED")
    print(f"experiment_id:                  {ctx.experiment_id}")
    print(f"paired_bootstrap_inference:     {outputs.paired_bootstrap_inference.relative_to(root)}")
    print(f"decision_bootstrap_inference:   {outputs.decision_bootstrap_inference.relative_to(root)}")
    print(f"continuous_bootstrap_inference: {outputs.continuous_bootstrap_inference.relative_to(root)}")
    print(f"annual_stability:               {outputs.annual_stability.relative_to(root)}")
    print(f"asset_stability:                {outputs.asset_stability.relative_to(root)}")
    print(f"publication_selective_repair:   {outputs.publication_selective_repair.relative_to(root)}")
    print(f"publication_meta:               {outputs.publication_meta.relative_to(root)}")
    print(f"audit:                          {outputs.audit_json.relative_to(root)}")
    print(f"manifest:                       {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:                     {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
