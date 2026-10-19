from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trading_policy_oracle_repair.checkpoint import write_checkpoint
from trading_policy_oracle_repair.config import load_config
from trading_policy_oracle_repair.logging_utils import configure_logging
from trading_policy_oracle_repair.raw_download import audit_downloads, download_all
from trading_policy_oracle_repair.registry import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 0: skeleton/config/registry/raw demo download")
    p.add_argument("--config", default="config/phase0.yaml")
    p.add_argument("--download", action="store_true", help="Download exploratory raw market data")
    p.add_argument("--skip-tests", action="store_true")
    return p.parse_args()


def ensure_dirs(root: Path, cfg) -> None:
    for rel in [
        cfg.paths.data_root,
        cfg.paths.raw_dir,
        cfg.paths.interim_dir,
        cfg.paths.processed_dir,
        cfg.paths.outputs_dir,
        cfg.paths.checkpoints_dir,
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_pytest(root: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_path = (root / args.config).resolve()
    cfg = load_config(cfg_path)
    ensure_dirs(root, cfg)

    logger = configure_logging()
    logger.info("Phase 0 start")
    logger.info("Config validated: %s", cfg_path)

    downloaded = []
    do_download = args.download or cfg.phase0.create_demo_raw_data
    if do_download:
        logger.info("Downloading exploratory raw data from %s", cfg.raw_data.provider)
        downloaded = download_all(cfg, root)
        audit = audit_downloads(downloaded)
        logger.info("Downloaded %s symbols", audit["n_symbols"])
        for row in audit["symbols"]:
            logger.info(
                "%s rows=%s range=%s..%s",
                row["symbol"], row["rows"], row["first_date"][:10], row["last_date"][:10]
            )
    else:
        audit = {"n_symbols": 0, "symbols": []}
        logger.info("Raw download skipped")

    if cfg.phase0.run_pytest and not args.skip_tests:
        logger.info("Running pytest")
        run_pytest(root)
        logger.info("pytest passed")

    data_files: list[Path] = []
    for item in downloaded:
        data_files.append(item.parquet_path)
        if item.csv_path is not None:
            data_files.append(item.csv_path)
        data_files.append(item.metadata_path)

    registry = ExperimentRegistry(root, cfg)
    ctx = registry.start_run(cfg_path, data_files=data_files)
    run_logger = configure_logging(ctx.run_dir / "logs" / "phase0.log")
    run_logger.info("Experiment created: %s", ctx.experiment_id)

    registry.finalize_run(
        ctx,
        status="PASSED",
        extra={
            "phase0_validation": {
                "config_valid": True,
                "pytest_passed": bool(cfg.phase0.run_pytest and not args.skip_tests),
                "raw_download_audit": audit,
            }
        },
    )

    checkpoint = write_checkpoint(
        root / cfg.paths.checkpoints_dir,
        experiment_id=ctx.experiment_id,
        phase=0,
        payload={
            "manifest": str(ctx.manifest_path.relative_to(root)),
            "raw_download_audit": audit,
            "next_phase": 1,
        },
    )

    print("\nPHASE 0 PASSED")
    print(f"experiment_id: {ctx.experiment_id}")
    print(f"manifest:      {ctx.manifest_path.relative_to(root)}")
    print(f"checkpoint:    {checkpoint.relative_to(root)}")
    print("\nSend me the terminal output plus manifest.json/checkpoint if anything looks wrong.")


if __name__ == "__main__":
    main()
