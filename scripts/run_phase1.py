from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase1_builder import build_phase1, run_phase1_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 1: canonical market data + event clock")
    p.add_argument("--config", default="config/phase1.yaml")
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
    logger.info("Phase 1 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase1_toy_validation()
    logger.info("Toy validation passed: %s", toy)

    outputs = build_phase1(cfg, root)
    logger.info("Canonical market written: %s", outputs.canonical_market.relative_to(root))
    logger.info("Cash curve written: %s", outputs.cash_curve.relative_to(root))
    logger.info("Event clock written: %s", outputs.event_clock.relative_to(root))

    if cfg.phase1 and cfg.phase1.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = list(outputs.input_raw_files) + [
        outputs.canonical_market,
        outputs.cash_curve,
        outputs.event_clock,
        outputs.audit_json,
        *outputs.csv_files,
    ]

    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase1.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(
        ctx,
        status="PASSED",
        extra={"phase1_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}},
    )

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=1,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "canonical_market": str(outputs.canonical_market.relative_to(root)),
            "cash_curve": str(outputs.cash_curve.relative_to(root)),
            "event_clock": str(outputs.event_clock.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 2,
        },
    )

    print("\nPHASE 1 PASSED")
    print(f"experiment_id:   {ctx.experiment_id}")
    print(f"canonical_market: {outputs.canonical_market.relative_to(root)}")
    print(f"cash_curve:       {outputs.cash_curve.relative_to(root)}")
    print(f"event_clock:      {outputs.event_clock.relative_to(root)}")
    print(f"audit:            {outputs.audit_json.relative_to(root)}")
    print(f"manifest:         {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:       {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
