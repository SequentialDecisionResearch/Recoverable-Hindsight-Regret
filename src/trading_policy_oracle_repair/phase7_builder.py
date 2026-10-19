from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .features import MODEL_FEATURE_COLUMNS, assert_feature_time_wall, build_market_features
from .hashing import sha256_file
from .preprocessing import fit_preprocessor
from .splits import build_fold_membership, deterministic_asset_split, make_annual_folds


@dataclass(frozen=True)
class Phase7Outputs:
    feature_table: Path
    model_dataset: Path
    fold_membership: Path
    fold_audit: Path
    preprocessor_parameters: Path
    preprocessors_json: Path
    asset_split_json: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase6_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase6_*.json"))
    return files[-1] if files else None


def require_phase6_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase7 is not None
    cp = _latest_phase6_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase7.require_phase6_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 6 checkpoint found. Run Phase 0 through Phase 6 first.")
    return cp


def _toy_feature_label_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2010-01-04", "2022-12-30")
    markets: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    for j, asset in enumerate(["AAA", "BBB", "CCC"]):
        t = np.arange(len(dates), dtype=float)
        p = 100.0 * np.exp(0.0002 * t + 0.03 * np.sin(t / (19.0 + j)))
        markets.append(pd.DataFrame({"date": dates, "asset_id": asset, "signal_price": p}))
        horizon = 5
        d = dates[:-horizon]
        a = dates[horizon:]
        for e in [0.0, 0.5, 1.0]:
            labels.append(pd.DataFrame({
                "date": d, "label_available_date": a, "asset_id": asset,
                "start_exposure": e, "policy_id": "p1", "utility": np.sin(np.arange(len(d))/23.0) * 0.01,
                "policy_class_regret": 0.0,
            }))
    return pd.concat(markets, ignore_index=True), pd.concat(labels, ignore_index=True)


def run_phase7_toy_validation() -> dict[str, Any]:
    market, labels = _toy_feature_label_data()
    features = build_market_features(market)
    assert_feature_time_wall(features)
    dataset = labels.merge(features, on=["asset_id", "date"], how="left", validate="many_to_one")
    train_assets, holdout_assets = deterministic_asset_split(["AAA", "BBB", "CCC"], 0.30, 7)
    folds = make_annual_folds(
        market["date"], initial_train_years=5, calibration_years=1, test_years=1,
        require_complete_test_year=True,
    )
    membership, audit = build_fold_membership(
        labels, folds, market["date"], train_assets, holdout_assets, embargo_days=5,
    )
    if membership.empty or audit.empty:
        raise AssertionError("toy walk-forward produced no folds")
    fold_id = str(audit.iloc[0]["fold_id"])
    train_keys = membership.loc[(membership.fold_id == fold_id) & (membership.split == "train"), ["asset_id", "date"]]
    state = dataset[["asset_id", "date", "start_exposure", *MODEL_FEATURE_COLUMNS[:-1]]].drop_duplicates()
    train = state.merge(train_keys.drop_duplicates(), on=["asset_id", "date"], how="inner")
    pre = fit_preprocessor(train, MODEL_FEATURE_COLUMNS)

    # Adversarial isolation: extreme test values must not change train-fit params.
    mutated = dataset.copy()
    test_keys = membership.loc[(membership.fold_id == fold_id) & membership.split.str.contains("test"), ["asset_id", "date"]].drop_duplicates()
    idx = mutated.reset_index().merge(test_keys, on=["asset_id", "date"], how="inner")["index"]
    mutated.loc[idx, "mom_5"] = 1e12
    state2 = mutated[["asset_id", "date", "start_exposure", *MODEL_FEATURE_COLUMNS[:-1]]].drop_duplicates()
    train2 = state2.merge(train_keys.drop_duplicates(), on=["asset_id", "date"], how="inner")
    pre2 = fit_preprocessor(train2, MODEL_FEATURE_COLUMNS)
    if pre.to_dict() != pre2.to_dict():
        raise AssertionError("test-only perturbation changed train preprocessor")

    first = audit.iloc[0]
    train_mem = membership.loc[(membership.fold_id == fold_id) & (membership.split == "train")]
    cal_mem = membership.loc[(membership.fold_id == fold_id) & (membership.split == "calibration")]
    if train_mem["label_available_date"].max() > pd.Timestamp(first["train_label_cutoff"]):
        raise AssertionError("train purge failed")
    if cal_mem["label_available_date"].max() > pd.Timestamp(first["calibration_label_cutoff"]):
        raise AssertionError("calibration purge failed")

    return {
        "status": "PASSED",
        "folds": int(len(folds)),
        "first_fold": fold_id,
        "train_assets": train_assets,
        "asset_oos_assets": holdout_assets,
        "train_fit_rows": int(pre.fit_rows),
        "test_perturbation_changed_preprocessor": False,
    }


def _fit_fold_preprocessors(
    dataset: pd.DataFrame,
    membership: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    # Remove policy repetition while preserving all start_exposure states.
    state_cols = ["asset_id", "date", "start_exposure", *[c for c in feature_columns if c != "start_exposure"]]
    states = dataset[state_cols].drop_duplicates().copy()
    bundles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for fold_id in sorted(membership["fold_id"].unique()):
        keys = membership.loc[(membership["fold_id"] == fold_id) & (membership["split"] == "train"), ["asset_id", "date"]].drop_duplicates()
        train = states.merge(keys, on=["asset_id", "date"], how="inner", validate="many_to_many")
        pre = fit_preprocessor(train, feature_columns)
        bundles[str(fold_id)] = pre.to_dict()
        for col in feature_columns:
            rows.append({
                "fold_id": str(fold_id), "feature": col,
                "median": pre.medians[col], "mean": pre.means[col], "scale": pre.scales[col],
                "fit_rows": pre.fit_rows,
            })
    return bundles, pd.DataFrame(rows)


def build_phase7(cfg: AppConfig, project_root: str | Path) -> Phase7Outputs:
    if cfg.phase != 7 or cfg.phase7 is None:
        raise ValueError("build_phase7 requires phase=7 config")
    root = Path(project_root).resolve()
    require_phase6_checkpoint(root, cfg)
    p7 = cfg.phase7
    rc = cfg.research_contract
    if p7.initial_train_years != rc.initial_train_years:
        raise ValueError("phase7 initial_train_years must match research_contract")
    if p7.calibration_years != rc.calibration_years or p7.test_years != rc.test_years:
        raise ValueError("phase7 calibration/test years must match research_contract")
    if p7.embargo_days != rc.embargo_days:
        raise ValueError("phase7 embargo_days must match research_contract")

    canonical_path = root / cfg.paths.processed_dir / p7.phase1_processed_subdir / "canonical_market.parquet"
    labels_path = root / cfg.paths.processed_dir / p7.phase6_processed_subdir / "continuation_labels.parquet"
    for path in (canonical_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase-7 input: {path}")

    logger = logging.getLogger("trading_policy_oracle_repair")
    market = pd.read_parquet(canonical_path)
    labels = pd.read_parquet(labels_path)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["label_available_date"] = pd.to_datetime(labels["label_available_date"]).dt.normalize()
    labels["asset_id"] = labels["asset_id"].astype(str).str.upper()

    logger.info("Building past-only Phase 7 feature table")
    features = build_market_features(market, annualization_days=p7.annualization_days)
    assert_feature_time_wall(features)
    model = labels.merge(features, on=["asset_id", "date"], how="left", validate="many_to_one")
    if model["feature_source_date"].isna().any():
        raise AssertionError("some labels have no matching feature row")
    if (pd.to_datetime(model["feature_source_date"]) > pd.to_datetime(model["date"])).any():
        raise AssertionError("future feature source detected")

    assets = [x.upper() for x in cfg.raw_data.tickers if x.upper() in set(labels.asset_id.unique())]
    train_assets, holdout_assets = deterministic_asset_split(assets, p7.asset_oos_fraction, cfg.seed)
    asset_split = {
        "seed": int(cfg.seed), "holdout_fraction": float(p7.asset_oos_fraction),
        "train_assets": train_assets, "asset_oos_assets": holdout_assets,
        "rule": "SHA256(seed|asset), take ceil(N*fraction) lowest hashes; independent of returns",
    }

    calendar = pd.DatetimeIndex(market["date"].drop_duplicates().sort_values())
    folds = make_annual_folds(
        calendar,
        initial_train_years=p7.initial_train_years,
        calibration_years=p7.calibration_years,
        test_years=p7.test_years,
        require_complete_test_year=p7.require_complete_test_year,
    )
    if not folds:
        raise ValueError("No valid annual walk-forward folds")
    logger.info("Phase 7 walk-forward folds: %d", len(folds))

    membership, fold_audit = build_fold_membership(
        labels, folds, calendar, train_assets, holdout_assets, p7.embargo_days,
    )
    if membership.empty:
        raise AssertionError("walk-forward membership is empty")
    if set(membership.loc[membership.split.isin(["train", "calibration"]), "asset_id"]) & set(holdout_assets):
        raise AssertionError("asset-OOS ticker leaked into train/calibration")

    bundles, prep_params = _fit_fold_preprocessors(model, membership, MODEL_FEATURE_COLUMNS)

    out_dir = root / cfg.paths.processed_dir / p7.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "feature_table.parquet"
    model_path = out_dir / "model_dataset.parquet"
    membership_path = out_dir / "fold_membership.parquet"
    fold_audit_path = out_dir / "walk_forward_folds.parquet"
    params_path = out_dir / "preprocessor_parameters.parquet"
    prep_json_path = out_dir / "preprocessors.json"
    asset_json_path = out_dir / "asset_split.json"
    audit_path = out_dir / "phase7_audit.json"

    features.to_parquet(feature_path, index=False)
    model.to_parquet(model_path, index=False)
    membership.to_parquet(membership_path, index=False)
    fold_audit.to_parquet(fold_audit_path, index=False)
    prep_params.to_parquet(params_path, index=False)
    prep_json_path.write_text(json.dumps(bundles, indent=2, sort_keys=True), encoding="utf-8")
    asset_json_path.write_text(json.dumps(asset_split, indent=2, sort_keys=True), encoding="utf-8")

    csv_files: list[Path] = []
    if p7.write_csv_summaries:
        fcsv = out_dir / "walk_forward_folds.csv"
        pcsv = out_dir / "preprocessor_parameters.csv"
        fold_audit.to_csv(fcsv, index=False)
        prep_params.to_csv(pcsv, index=False)
        csv_files.extend([fcsv, pcsv])

    split_counts = membership.groupby(["fold_id", "split"], observed=True).size().rename("date_asset_rows").reset_index()
    audit: dict[str, Any] = {
        "phase": 7,
        "scope": "chronological walk-forward, purge/embargo, asset-OOS and train-only preprocessing; no model fitting",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_time_wall": "all source timestamps <= decision date",
        "fold_count": int(len(folds)),
        "first_fold": str(folds[0].fold_id),
        "last_fold": str(folds[-1].fold_id),
        "asset_split": asset_split,
        "embargo_days": int(p7.embargo_days),
        "label_horizon_days": int(rc.horizon_days),
        "require_complete_test_year": bool(p7.require_complete_test_year),
        "rows": {
            "feature_table": int(len(features)), "model_dataset": int(len(model)),
            "fold_membership": int(len(membership)), "fold_audit": int(len(fold_audit)),
            "preprocessor_parameters": int(len(prep_params)),
        },
        "split_counts": split_counts.to_dict(orient="records"),
        "inputs": {
            "canonical_market_sha256": sha256_file(canonical_path),
            "continuation_labels_sha256": sha256_file(labels_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return Phase7Outputs(
        feature_table=feature_path, model_dataset=model_path, fold_membership=membership_path,
        fold_audit=fold_audit_path, preprocessor_parameters=params_path,
        preprocessors_json=prep_json_path, asset_split_json=asset_json_path,
        audit_json=audit_path, csv_files=tuple(csv_files), input_files=(canonical_path, labels_path),
    )
