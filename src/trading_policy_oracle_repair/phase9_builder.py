from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .hashing import sha256_file
from .uncertainty import PairwiseCalibrator, evidence_metrics, fit_pairwise_calibrator


STATE_KEYS = ["date", "label_available_date", "asset_id", "start_exposure", "horizon"]


@dataclass(frozen=True)
class Phase9Outputs:
    pairwise_calibrators: Path
    evidence_predictions: Path
    coverage_harm_frontier: Path
    coverage_harm_summary: Path
    calibration_coverage: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase8_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase8_*.json"))
    return files[-1] if files else None


def require_phase8_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase9 is not None
    cp = _latest_phase8_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase9.require_phase8_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 8 checkpoint found. Run Phase 0 through Phase 8 first.")
    return cp


def _utility_wide(predictions: pd.DataFrame, *, fold_id: str, split: str, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = predictions.loc[
        (predictions["fold_id"].astype(str) == str(fold_id))
        & (predictions["split"].astype(str) == str(split))
        & (predictions["task"].astype(str) == "utility")
        & (predictions["model_name"].astype(str) == str(model_name))
    ].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    for c in ["date", "label_available_date"]:
        df[c] = pd.to_datetime(df[c]).dt.normalize()
    if df.duplicated(STATE_KEYS + ["policy_id"]).any():
        raise AssertionError("Phase-9 utility predictions are not unique per state/policy")
    actual = df.pivot(index=STATE_KEYS, columns="policy_id", values="actual_utility").sort_index(axis=1)
    predicted = df.pivot(index=STATE_KEYS, columns="policy_id", values="predicted_value").sort_index(axis=1)
    common_cols = sorted(set(actual.columns) & set(predicted.columns))
    actual = actual[common_cols].dropna(how="any")
    predicted = predicted[common_cols].reindex(actual.index).dropna(how="any")
    actual = actual.reindex(predicted.index)
    if not actual.index.equals(predicted.index):
        raise AssertionError("actual/predicted utility state indexes differ")
    return actual, predicted


def _pair_frame(actual: pd.DataFrame, predicted: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    if baseline == candidate:
        raise ValueError("baseline and candidate must differ")
    if baseline not in actual or candidate not in actual or baseline not in predicted or candidate not in predicted:
        raise KeyError("missing policy column for pair")
    out = actual.index.to_frame(index=False)
    out["baseline_policy_id"] = str(baseline)
    out["candidate_policy_id"] = str(candidate)
    out["actual_advantage"] = actual[candidate].to_numpy(dtype=float) - actual[baseline].to_numpy(dtype=float)
    out["predicted_advantage"] = predicted[candidate].to_numpy(dtype=float) - predicted[baseline].to_numpy(dtype=float)
    return out


def _calibrator_row(
    *, fold_id: str, model_name: str, baseline: str, candidate: str,
    cal: PairwiseCalibrator, alpha: float, block_length: int, bootstrap_resamples: int,
) -> dict[str, Any]:
    return {
        "fold_id": str(fold_id), "model_name": str(model_name),
        "baseline_policy_id": str(baseline), "candidate_policy_id": str(candidate),
        "alpha": float(alpha), "target_lower_bound_coverage": float(1.0 - alpha),
        "calibration_rows": int(cal.calibration_rows),
        "empirical_residual_quantile": float(cal.empirical_residual_quantile),
        "block_bootstrap_residual_quantile": float(cal.bootstrap_residual_quantile),
        "empirical_lower_bound_coverage": float(cal.empirical_coverage),
        "block_bootstrap_lower_bound_coverage": float(cal.bootstrap_coverage),
        "bootstrap_block_length_dates": int(block_length),
        "bootstrap_resamples": int(bootstrap_resamples),
    }


def _summary_over_folds(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    group_cols = ["split", "model_name", "baseline_policy_id", "lcb_method", "deploy_margin"]
    numeric = [c for c in ["coverage", "harm_rate", "mean_actual_advantage", "median_actual_advantage", "positive_rate", "positive_advantage_recall"] if c in frontier]
    rows: list[dict[str, Any]] = []
    for key, g in frontier.groupby(group_cols, dropna=False, observed=True):
        row = dict(zip(group_cols, key))
        row["folds"] = int(g["fold_id"].nunique())
        row["n_total"] = int(g["n"].sum())
        row["deployed_total"] = int(g["deployed"].sum())
        for col in numeric:
            x = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(x.mean()) if x.notna().any() else float("nan")
            row[f"{col}_std"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _choose_best_candidate(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    *, baseline: str, calibrators: dict[tuple[str, str], PairwiseCalibrator], method: str,
) -> pd.DataFrame:
    candidates = [c for c in actual.columns if c != baseline]
    if not candidates:
        raise ValueError("need at least two policies")
    pred_adv = np.column_stack([predicted[c].to_numpy(dtype=float) - predicted[baseline].to_numpy(dtype=float) for c in candidates])
    act_adv = np.column_stack([actual[c].to_numpy(dtype=float) - actual[baseline].to_numpy(dtype=float) for c in candidates])
    lcb = np.empty_like(pred_adv)
    for j, candidate in enumerate(candidates):
        lcb[:, j] = calibrators[(baseline, candidate)].lower_bound(pred_adv[:, j], method=method)
    best_j = np.argmax(lcb, axis=1)
    rows = actual.index.to_frame(index=False)
    rows["baseline_policy_id"] = baseline
    rows["selected_candidate_policy_id"] = np.asarray(candidates, dtype=object)[best_j]
    idx = np.arange(len(rows))
    rows["predicted_advantage"] = pred_adv[idx, best_j]
    rows["actual_advantage"] = act_adv[idx, best_j]
    rows["lower_bound"] = lcb[idx, best_j]
    rows["lcb_method"] = str(method)
    return rows


def run_phase9_toy_validation(seed: int = 19) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cal_n, test_n = 1200, 1200
    dates_cal = pd.bdate_range("2018-01-02", periods=cal_n)
    dates_test = pd.bdate_range("2023-01-02", periods=test_n)
    signal_cal = rng.normal(0.010, 0.020, size=cal_n)
    signal_test = rng.normal(0.010, 0.020, size=test_n)
    noise_cal = rng.normal(0.0, 0.020, size=cal_n)
    noise_test = rng.normal(0.0, 0.020, size=test_n)
    pred_cal = signal_cal
    actual_cal = signal_cal + noise_cal
    pred_test = signal_test
    actual_test = signal_test + noise_test
    cal_df = pd.DataFrame({"date": dates_cal, "actual_advantage": actual_cal, "predicted_advantage": pred_cal})
    cal = fit_pairwise_calibrator(
        cal_df, alpha=0.10, block_length_dates=20, n_resamples=40,
        conservative_quantile=0.10, seed=seed,
    )
    lcb = cal.lower_bound(pred_test, "empirical")
    gated = evidence_metrics(actual_test, lcb, margin=0.0)
    naive = evidence_metrics(actual_test, pred_test, margin=0.0)
    if cal.empirical_coverage < 0.88:
        raise AssertionError("toy empirical LCB calibration coverage is too low")
    if gated["coverage"] >= naive["coverage"]:
        raise AssertionError("toy calibrated gate should abstain more than raw predicted-advantage gate")
    if np.isfinite(gated["harm_rate"]) and np.isfinite(naive["harm_rate"]) and gated["harm_rate"] > naive["harm_rate"] + 0.02:
        raise AssertionError("toy calibrated gate should not materially increase harm")
    return {
        "status": "PASSED",
        "empirical_calibration_coverage": float(cal.empirical_coverage),
        "bootstrap_calibration_coverage": float(cal.bootstrap_coverage),
        "naive_coverage": float(naive["coverage"]),
        "gated_coverage": float(gated["coverage"]),
        "naive_harm_rate": float(naive["harm_rate"]),
        "gated_harm_rate": float(gated["harm_rate"]),
    }


def build_phase9(cfg: AppConfig, project_root: str | Path) -> Phase9Outputs:
    if cfg.phase != 9 or cfg.phase9 is None:
        raise ValueError("build_phase9 requires phase=9 config")
    root = Path(project_root).resolve()
    require_phase8_checkpoint(root, cfg)
    p9 = cfg.phase9
    p8dir = root / cfg.paths.processed_dir / p9.phase8_processed_subdir
    predictions_path = p8dir / "predictions.parquet"
    model_index_path = p8dir / "model_index.parquet"
    for path in (predictions_path, model_index_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase-9 input: {path}")

    logger = logging.getLogger("trading_policy_oracle_repair")
    prediction_columns = [
        "fold_id", "split", *STATE_KEYS, "policy_id", "task", "model_name",
        "actual_utility", "predicted_value",
    ]
    try:
        predictions = pd.read_parquet(
            predictions_path, columns=prediction_columns, filters=[("task", "==", "utility")],
        )
    except Exception:
        # Compatibility fallback for parquet engines/files that cannot push down the filter.
        predictions = pd.read_parquet(predictions_path, columns=prediction_columns)
        predictions = predictions.loc[predictions["task"].astype(str) == "utility"].copy()
    required = set(prediction_columns)
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Phase-8 predictions missing columns: {sorted(missing)}")
    predictions = predictions.loc[predictions["task"].astype(str) == "utility"].copy()
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.normalize()
    predictions["label_available_date"] = pd.to_datetime(predictions["label_available_date"]).dt.normalize()

    out_dir = root / cfg.paths.processed_dir / p9.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    calibrators_path = out_dir / "pairwise_calibrators.parquet"
    evidence_path = out_dir / "evidence_predictions.parquet"
    frontier_path = out_dir / "coverage_harm_frontier.parquet"
    summary_path = out_dir / "coverage_harm_summary.parquet"
    calibration_path = out_dir / "calibration_coverage.parquet"
    audit_path = out_dir / "phase9_audit.json"
    for stale in (calibrators_path, evidence_path, frontier_path, summary_path, calibration_path, audit_path):
        if stale.exists():
            stale.unlink()

    fold_ids = sorted(predictions["fold_id"].astype(str).unique())
    calibrator_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    frontier_rows: list[dict[str, Any]] = []
    policy_ids = sorted(predictions["policy_id"].astype(str).unique())
    pair_count = len(policy_ids) * (len(policy_ids) - 1)

    for fold_no, fold_id in enumerate(fold_ids):
        logger.info("Phase 9 fold %s (%d/%d)", fold_id, fold_no + 1, len(fold_ids))
        for model_no, model_name in enumerate(p9.utility_models):
            cal_actual, cal_pred = _utility_wide(predictions, fold_id=fold_id, split="calibration", model_name=model_name)
            if cal_actual.empty:
                raise AssertionError(f"missing calibration utility predictions for {fold_id}/{model_name}")
            fold_policies = list(cal_actual.columns)
            calibrators: dict[tuple[str, str], PairwiseCalibrator] = {}
            pair_no = 0
            for baseline in fold_policies:
                for candidate in fold_policies:
                    if candidate == baseline:
                        continue
                    pair_no += 1
                    pf = _pair_frame(cal_actual, cal_pred, baseline, candidate)
                    if len(pf) < int(p9.min_calibration_rows):
                        raise AssertionError(f"insufficient calibration rows for {fold_id}/{model_name}/{baseline}->{candidate}")
                    pair_seed = int(cfg.seed) + fold_no * 10000 + model_no * 1000 + pair_no
                    cal = fit_pairwise_calibrator(
                        pf, alpha=p9.alpha, block_length_dates=p9.bootstrap_block_length_dates,
                        n_resamples=p9.bootstrap_resamples,
                        conservative_quantile=p9.bootstrap_conservative_quantile, seed=pair_seed,
                    )
                    calibrators[(baseline, candidate)] = cal
                    row = _calibrator_row(
                        fold_id=fold_id, model_name=model_name, baseline=baseline, candidate=candidate,
                        cal=cal, alpha=p9.alpha, block_length=p9.bootstrap_block_length_dates,
                        bootstrap_resamples=p9.bootstrap_resamples,
                    )
                    calibrator_rows.append(row)
                    for method, coverage in [
                        ("empirical", cal.empirical_coverage),
                        ("block_bootstrap", cal.bootstrap_coverage),
                    ]:
                        calibration_rows.append({
                            "fold_id": fold_id, "model_name": model_name,
                            "baseline_policy_id": baseline, "candidate_policy_id": candidate,
                            "lcb_method": method, "calibration_rows": int(cal.calibration_rows),
                            "alpha": float(p9.alpha), "target_coverage": float(1.0 - p9.alpha),
                            "observed_lower_bound_coverage": float(coverage),
                        })

            if len(calibrators) != pair_count:
                raise AssertionError("unexpected number of pairwise calibrators")

            for split in p9.evaluation_splits:
                actual, pred = _utility_wide(predictions, fold_id=fold_id, split=split, model_name=model_name)
                if actual.empty:
                    continue
                if list(actual.columns) != fold_policies:
                    raise AssertionError("policy set differs between calibration and evaluation")
                for method in ("empirical", "block_bootstrap"):
                    for baseline in fold_policies:
                        ev = _choose_best_candidate(
                            actual, pred, baseline=baseline, calibrators=calibrators, method=method,
                        )
                        ev.insert(0, "fold_id", fold_id)
                        ev.insert(1, "split", split)
                        ev.insert(2, "model_name", model_name)
                        ev["deploy_margin"] = float(p9.deploy_margin)
                        ev["deploy_flag"] = (ev["lower_bound"].to_numpy(dtype=float) > float(p9.deploy_margin)).astype(np.int8)
                        evidence_frames.append(ev)
                        for margin in p9.frontier_margins:
                            mm = evidence_metrics(ev["actual_advantage"].to_numpy(), ev["lower_bound"].to_numpy(), margin=float(margin))
                            frontier_rows.append({
                                "fold_id": fold_id, "split": split, "model_name": model_name,
                                "baseline_policy_id": baseline, "lcb_method": method,
                                "deploy_margin": float(margin), **mm,
                            })
        logger.info("Phase 9 %s complete", fold_id)

    calibrators_df = pd.DataFrame(calibrator_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    evidence_df = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
    frontier_df = pd.DataFrame(frontier_rows)
    summary_df = _summary_over_folds(frontier_df)
    if calibrators_df.empty or evidence_df.empty or frontier_df.empty:
        raise AssertionError("Phase-9 produced empty outputs")

    calibrators_df.to_parquet(calibrators_path, index=False)
    evidence_df.to_parquet(evidence_path, index=False)
    frontier_df.to_parquet(frontier_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    calibration_df.to_parquet(calibration_path, index=False)

    csv_files: list[Path] = []
    if p9.write_csv_summaries:
        for frame, name in [
            (calibrators_df, "pairwise_calibrators.csv"),
            (calibration_df, "calibration_coverage.csv"),
            (frontier_df, "coverage_harm_frontier.csv"),
            (summary_df, "coverage_harm_summary.csv"),
        ]:
            path = out_dir / name
            frame.to_csv(path, index=False)
            csv_files.append(path)
        if p9.evidence_sample_rows > 0:
            sample = out_dir / "evidence_predictions_sample.csv"
            evidence_df.head(int(p9.evidence_sample_rows)).to_csv(sample, index=False)
            csv_files.append(sample)

    primary = summary_df.loc[
        (summary_df["lcb_method"] == p9.primary_lcb_method)
        & np.isclose(summary_df["deploy_margin"].astype(float), float(p9.deploy_margin))
    ].copy()
    audit = {
        "phase": 9,
        "scope": "uncertainty calibration and abstention diagnostics only; no trades are executed",
        "primary_lcb_method": p9.primary_lcb_method,
        "alpha": float(p9.alpha),
        "target_lower_bound_coverage": float(1.0 - p9.alpha),
        "deploy_margin": float(p9.deploy_margin),
        "frontier_margins": list(map(float, p9.frontier_margins)),
        "utility_models": list(p9.utility_models),
        "evaluation_splits": list(p9.evaluation_splits),
        "bootstrap": {
            "block_length_dates": int(p9.bootstrap_block_length_dates),
            "resamples": int(p9.bootstrap_resamples),
            "conservative_quantile": float(p9.bootstrap_conservative_quantile),
            "role": "robustness; primary gate remains empirical unless config changes",
        },
        "calibration_contract": "pairwise advantage residual quantiles use calibration split only; no test labels affect calibrators",
        "selection_contract": "best candidate is argmax calibrated lower bound; deploy flag requires LCB > deploy_margin",
        "fold_count": int(len(fold_ids)),
        "policy_count": int(len(policy_ids)),
        "ordered_pair_count_per_fold_model": int(pair_count),
        "calibrator_rows": int(len(calibrators_df)),
        "evidence_rows": int(len(evidence_df)),
        "frontier_rows": int(len(frontier_df)),
        "primary_summary_rows": int(len(primary)),
        "test_labels_used_for_calibration": False,
        "trading_actions_executed": False,
        "inputs": {
            "phase8_predictions_sha256": sha256_file(predictions_path),
            "phase8_model_index_sha256": sha256_file(model_index_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return Phase9Outputs(
        pairwise_calibrators=calibrators_path,
        evidence_predictions=evidence_path,
        coverage_harm_frontier=frontier_path,
        coverage_harm_summary=summary_path,
        calibration_coverage=calibration_path,
        audit_json=audit_path,
        csv_files=tuple(csv_files),
        input_files=(predictions_path, model_index_path),
    )
