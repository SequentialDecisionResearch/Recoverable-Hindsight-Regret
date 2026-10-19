from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase4_builder import build_phase4, run_phase4_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 4: replay and audit outputs")
    p.add_argument("--config", default="config/phase4.yaml")
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
    logger.info("Phase 4 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase4_toy_validation()
    logger.info("Toy validation passed: %s", toy)

    outputs = build_phase4(cfg, root)
    logger.info("Replay ledger written: %s", outputs.replay_ledger.relative_to(root))
    logger.info("Replay fills written: %s", outputs.replay_fills.relative_to(root))
    logger.info("Replay actions written: %s", outputs.replay_actions.relative_to(root))
    logger.info("Performance summary written: %s", outputs.performance_summary.relative_to(root))
    logger.info("Cost sensitivity written: %s", outputs.cost_sensitivity.relative_to(root))

    if cfg.phase4 and cfg.phase4.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files,
        outputs.replay_ledger,
        outputs.replay_fills,
        outputs.replay_actions,
        outputs.performance_summary,
        outputs.cost_sensitivity,
        outputs.audit_json,
        *outputs.csv_files,
    ]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase4.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(
        ctx,
        status="PASSED",
        extra={"phase4_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}},
    )

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=4,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "replay_ledger": str(outputs.replay_ledger.relative_to(root)),
            "replay_fills": str(outputs.replay_fills.relative_to(root)),
            "replay_actions": str(outputs.replay_actions.relative_to(root)),
            "performance_summary": str(outputs.performance_summary.relative_to(root)),
            "cost_sensitivity": str(outputs.cost_sensitivity.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 5,
        },
    )

    print("\nPHASE 4 PASSED")
    print(f"experiment_id:      {ctx.experiment_id}")
    print(f"replay_ledger:       {outputs.replay_ledger.relative_to(root)}")
    print(f"replay_fills:        {outputs.replay_fills.relative_to(root)}")
    print(f"replay_actions:      {outputs.replay_actions.relative_to(root)}")
    print(f"performance_summary: {outputs.performance_summary.relative_to(root)}")
    print(f"cost_sensitivity:    {outputs.cost_sensitivity.relative_to(root)}")
    print(f"audit:               {outputs.audit_json.relative_to(root)}")
    print(f"manifest:            {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:          {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
