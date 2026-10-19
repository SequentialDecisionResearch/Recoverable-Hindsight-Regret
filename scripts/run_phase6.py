from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase6_builder import build_phase6, run_phase6_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 6: common-state offline continuation labels")
    p.add_argument("--config", default="config/phase6.yaml")
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
    logger.info("Phase 6 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase6_toy_validation()
    logger.info("Toy validation passed: %s", toy)

    outputs = build_phase6(cfg, root)
    logger.info("Continuation labels written: %s", outputs.continuation_labels.relative_to(root))
    logger.info("Pairwise advantages written: %s", outputs.pairwise_advantage.relative_to(root))
    logger.info("State summary written: %s", outputs.state_summary.relative_to(root))

    if cfg.phase6 and cfg.phase6.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files,
        outputs.continuation_labels,
        outputs.pairwise_advantage,
        outputs.state_summary,
        outputs.audit_json,
    ]
    if outputs.csv_sample is not None:
        data_files.append(outputs.csv_sample)

    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase6.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(
        ctx,
        status="PASSED",
        extra={"phase6_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}},
    )

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=6,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "continuation_labels": str(outputs.continuation_labels.relative_to(root)),
            "pairwise_advantage": str(outputs.pairwise_advantage.relative_to(root)),
            "state_summary": str(outputs.state_summary.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 7,
        },
    )

    print("\nPHASE 6 PASSED")
    print(f"experiment_id:       {ctx.experiment_id}")
    print(f"continuation_labels:  {outputs.continuation_labels.relative_to(root)}")
    print(f"pairwise_advantage:   {outputs.pairwise_advantage.relative_to(root)}")
    print(f"state_summary:        {outputs.state_summary.relative_to(root)}")
    print(f"audit:                {outputs.audit_json.relative_to(root)}")
    print(f"manifest:             {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:           {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
