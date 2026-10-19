from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.phase8_builder import build_phase8, run_phase8_toy_validation
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 8: train-only predictability models and OOS predictions")
    p.add_argument("--config", default="config/phase8.yaml")
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
    logger.info("Phase 8 start")
    logger.info("Config validated: %s", cfg_path)

    toy = run_phase8_toy_validation()
    logger.info("Toy validation passed: %s", toy)
    outputs = build_phase8(cfg, root)
    logger.info("Predictions written: %s", outputs.predictions.relative_to(root))
    logger.info("Predictability metrics written: %s", outputs.metrics.relative_to(root))
    logger.info("Predictability summary written: %s", outputs.summary_metrics.relative_to(root))
    logger.info("Prediction deciles written: %s", outputs.prediction_deciles.relative_to(root))
    logger.info("Model index written: %s", outputs.model_index.relative_to(root))

    if cfg.phase8 and cfg.phase8.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files = [
        *outputs.input_files, outputs.predictions, outputs.metrics, outputs.summary_metrics,
        outputs.prediction_deciles, outputs.high_regret_thresholds, outputs.model_index,
        outputs.audit_json, *outputs.csv_files,
    ]
    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase8.log")
    audit = json.loads(outputs.audit_json.read_text(encoding="utf-8"))
    run_logger.info("Experiment created: %s", ctx.experiment_id)
    registry.finalize_run(ctx, status="PASSED", extra={
        "phase8_validation": {"toy_validation": toy, "pytest_passed": not args.skip_tests, "audit": audit}
    })

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=8,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "predictions": str(outputs.predictions.relative_to(root)),
            "predictability_metrics": str(outputs.metrics.relative_to(root)),
            "predictability_summary": str(outputs.summary_metrics.relative_to(root)),
            "prediction_deciles": str(outputs.prediction_deciles.relative_to(root)),
            "high_regret_thresholds": str(outputs.high_regret_thresholds.relative_to(root)),
            "model_index": str(outputs.model_index.relative_to(root)),
            "audit_json": str(outputs.audit_json.relative_to(root)),
            "next_phase": 9,
        },
    )

    print("\nPHASE 8 PASSED")
    print(f"experiment_id:             {ctx.experiment_id}")
    print(f"predictions:               {outputs.predictions.relative_to(root)}")
    print(f"predictability_metrics:    {outputs.metrics.relative_to(root)}")
    print(f"predictability_summary:    {outputs.summary_metrics.relative_to(root)}")
    print(f"prediction_deciles:        {outputs.prediction_deciles.relative_to(root)}")
    print(f"high_regret_thresholds:    {outputs.high_regret_thresholds.relative_to(root)}")
    print(f"model_index:               {outputs.model_index.relative_to(root)}")
    print(f"audit:                     {outputs.audit_json.relative_to(root)}")
    print(f"manifest:                  {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:                {checkpoint.relative_to(root)}")


if __name__ == "__main__":
    main()
