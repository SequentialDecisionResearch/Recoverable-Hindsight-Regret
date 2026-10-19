from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase7_builder import build_phase7, run_phase7_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 7: walk-forward splits, purge/embargo, asset-OOS and preprocessing")
    p.add_argument("--config", default="config/phase7.yaml")
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
    logger.info("Phase 7 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase7_toy_validation()
    logger.info("Toy validation passed: %s", toy)
    outputs = build_phase7(cfg, root)
    logger.info("Feature table written: %s", outputs.feature_table.relative_to(root))
    logger.info("Model dataset written: %s", outputs.model_dataset.relative_to(root))
    logger.info("Fold membership written: %s", outputs.fold_membership.relative_to(root))
    logger.info("Walk-forward audit written: %s", outputs.fold_audit.relative_to(root))
    logger.info("Frozen preprocessors written: %s", outputs.preprocessors_json.relative_to(root))
    logger.info("Asset split written: %s", outputs.asset_split_json.relative_to(root))

    if cfg.phase7 and cfg.phase7.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [*outputs.input_files, outputs.feature_table, outputs.model_dataset, outputs.fold_membership,
                  outputs.fold_audit, outputs.preprocessor_parameters, outputs.preprocessors_json,
                  outputs.asset_split_json, outputs.audit_json, *outputs.csv_files]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase7.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(ctx, status="PASSED", extra={
        "phase7_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}
    })

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=7,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "feature_table": str(outputs.feature_table.relative_to(root)),
            "model_dataset": str(outputs.model_dataset.relative_to(root)),
            "fold_membership": str(outputs.fold_membership.relative_to(root)),
            "walk_forward_folds": str(outputs.fold_audit.relative_to(root)),
            "preprocessors": str(outputs.preprocessors_json.relative_to(root)),
            "asset_split": str(outputs.asset_split_json.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 8,
        },
    )

    print("\nPHASE 7 PASSED")
    print(f"experiment_id:        {ctx.experiment_id}")
    print(f"feature_table:         {outputs.feature_table.relative_to(root)}")
    print(f"model_dataset:         {outputs.model_dataset.relative_to(root)}")
    print(f"fold_membership:       {outputs.fold_membership.relative_to(root)}")
    print(f"walk_forward_folds:    {outputs.fold_audit.relative_to(root)}")
    print(f"preprocessors:         {outputs.preprocessors_json.relative_to(root)}")
    print(f"asset_split:           {outputs.asset_split_json.relative_to(root)}")
    print(f"audit:                 {outputs.audit_json.relative_to(root)}")
    print(f"manifest:              {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:            {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
