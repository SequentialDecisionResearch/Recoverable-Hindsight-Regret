from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase10_builder import build_phase10, run_phase10_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 10: selective repair and meta-policy selector")
    p.add_argument("--config", default="config/phase10.yaml")
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
    logger.info("Phase 10 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase10_toy_validation()
    logger.info("Toy validation passed: %s", toy)
    outputs = build_phase10(cfg, root)
    logger.info("Deployment actions written: %s", outputs.deployment_actions.relative_to(root))
    logger.info("Deployment ledger written: %s", outputs.deployment_ledger.relative_to(root))
    logger.info("Strategy performance written: %s", outputs.strategy_performance.relative_to(root))
    logger.info("Selective-repair OOS written: %s", outputs.selective_repair_oos.relative_to(root))
    logger.info("Meta-selection OOS written: %s", outputs.meta_selection_oos.relative_to(root))
    if outputs.continuous_performance:
        logger.info("Continuous pseudo-live performance written: %s", outputs.continuous_performance.relative_to(root))

    if cfg.phase10 and cfg.phase10.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files,
        outputs.deployment_actions, outputs.deployment_ledger, outputs.deployment_fills,
        outputs.strategy_performance, outputs.decision_diagnostics,
        outputs.selective_repair_oos, outputs.selective_repair_summary,
        outputs.meta_selection_oos, outputs.meta_selection_summary,
        outputs.audit_json, *outputs.csv_files,
    ]
    for x in [outputs.continuous_actions, outputs.continuous_ledger, outputs.continuous_fills, outputs.continuous_performance]:
        if x is not None:
            data_files.append(x)

    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase10.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(ctx, status="PASSED", extra={
        "phase10_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}
    })

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=10,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "deployment_actions": str(outputs.deployment_actions.relative_to(root)),
            "deployment_ledger": str(outputs.deployment_ledger.relative_to(root)),
            "strategy_performance": str(outputs.strategy_performance.relative_to(root)),
            "selective_repair_oos": str(outputs.selective_repair_oos.relative_to(root)),
            "meta_selection_oos": str(outputs.meta_selection_oos.relative_to(root)),
            "continuous_performance": str(outputs.continuous_performance.relative_to(root)) if outputs.continuous_performance else None,
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 11,
        },
    )

    print("\nPHASE 10 PASSED")
    print(f"experiment_id:             {ctx.experiment_id}")
    print(f"deployment_actions:         {outputs.deployment_actions.relative_to(root)}")
    print(f"deployment_ledger:          {outputs.deployment_ledger.relative_to(root)}")
    print(f"deployment_fills:           {outputs.deployment_fills.relative_to(root)}")
    print(f"strategy_performance:       {outputs.strategy_performance.relative_to(root)}")
    print(f"decision_diagnostics:       {outputs.decision_diagnostics.relative_to(root)}")
    print(f"selective_repair_oos:       {outputs.selective_repair_oos.relative_to(root)}")
    print(f"selective_repair_summary:   {outputs.selective_repair_summary.relative_to(root)}")
    print(f"meta_selection_oos:         {outputs.meta_selection_oos.relative_to(root)}")
    print(f"meta_selection_summary:     {outputs.meta_selection_summary.relative_to(root)}")
    if outputs.continuous_performance:
        print(f"continuous_performance:     {outputs.continuous_performance.relative_to(root)}")
    print(f"audit:                      {outputs.audit_json.relative_to(root)}")
    print(f"manifest:                   {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:                 {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
