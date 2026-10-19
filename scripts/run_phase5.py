from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase5_builder import build_phase5, run_phase5_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 5: hindsight oracle infrastructure")
    p.add_argument("--config", default="config/phase5.yaml")
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
    logger.info("Phase 5 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase5_toy_validation()
    logger.info("Toy validation passed: %s", toy)

    outputs = build_phase5(cfg, root)
    logger.info("Bounded-switch oracle written: %s", outputs.bounded_switch_oracle.relative_to(root))
    logger.info("Bounded-switch actions written: %s", outputs.bounded_switch_actions.relative_to(root))
    logger.info("Policy-class oracle written: %s", outputs.policy_class_oracle.relative_to(root))
    logger.info("Local continuation smoke written: %s", outputs.local_continuation_smoke.relative_to(root))

    if cfg.phase5 and cfg.phase5.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files,
        outputs.bounded_switch_oracle,
        outputs.bounded_switch_actions,
        outputs.policy_class_oracle,
        outputs.local_continuation_smoke,
        outputs.audit_json,
        *outputs.csv_files,
    ]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase5.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(
        ctx,
        status="PASSED",
        extra={"phase5_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}},
    )

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=5,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "bounded_switch_oracle": str(outputs.bounded_switch_oracle.relative_to(root)),
            "bounded_switch_actions": str(outputs.bounded_switch_actions.relative_to(root)),
            "policy_class_oracle": str(outputs.policy_class_oracle.relative_to(root)),
            "local_continuation_smoke": str(outputs.local_continuation_smoke.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 6,
        },
    )

    print("\nPHASE 5 PASSED")
    print(f"experiment_id:             {ctx.experiment_id}")
    print(f"bounded_switch_oracle:     {outputs.bounded_switch_oracle.relative_to(root)}")
    print(f"bounded_switch_actions:    {outputs.bounded_switch_actions.relative_to(root)}")
    print(f"policy_class_oracle:       {outputs.policy_class_oracle.relative_to(root)}")
    print(f"local_continuation_smoke:  {outputs.local_continuation_smoke.relative_to(root)}")
    print(f"audit:                     {outputs.audit_json.relative_to(root)}")
    print(f"manifest:                  {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:                {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
