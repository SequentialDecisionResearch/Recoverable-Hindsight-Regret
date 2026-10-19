from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import AppConfig
from .features import MODEL_FEATURE_COLUMNS
from .hashing import sha256_file
from .models import fit_high_regret_classifier, fit_regressor, predict_probability
from .predictability import classification_metrics, prediction_deciles, regression_metrics
from .preprocessing import FrozenPreprocessor, fit_preprocessor


@dataclass(frozen=True)
class Phase8Outputs:
    predictions: Path
    metrics: Path
    summary_metrics: Path
    prediction_deciles: Path
    high_regret_thresholds: Path
    model_index: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    model_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase7_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase7_*.json"))
    return files[-1] if files else None


def require_phase7_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase8 is not None
    cp = _latest_phase7_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase8.require_phase7_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 7 checkpoint found. Run Phase 0 through Phase 7 first.")
    return cp


def _open_parquet_writer(path: Path, frame: pd.DataFrame):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = pq.ParquetWriter(str(path), table.schema, compression="zstd", use_dictionary=True)
    writer.write_table(table)
    return writer


def _append_parquet(writer, frame: pd.DataFrame) -> None:
    import pyarrow as pa

    writer.write_table(pa.Table.from_pandas(frame, preserve_index=False))


def _prediction_frame(
    base: pd.DataFrame,
    *,
    fold_id: str,
    split: str,
    policy_id: str,
    task: str,
    model_name: str,
    predicted: np.ndarray,
    actual_target: np.ndarray,
    high_regret_threshold: float,
) -> pd.DataFrame:
    cols = ["date", "label_available_date", "asset_id", "start_exposure", "horizon", "utility", "policy_class_regret"]
    out = base[cols].copy()
    out.insert(0, "fold_id", str(fold_id))
    out.insert(1, "split", str(split))
    out["policy_id"] = str(policy_id)
    out["task"] = str(task)
    out["model_name"] = str(model_name)
    out["actual_target"] = np.asarray(actual_target, dtype=float)
    out["predicted_value"] = np.asarray(predicted, dtype=float)
    out["actual_utility"] = pd.to_numeric(out["utility"], errors="raise").astype(float)
    out["actual_regret"] = pd.to_numeric(out["policy_class_regret"], errors="raise").astype(float)
    out["high_regret_threshold"] = float(high_regret_threshold)
    out["high_regret_label"] = (out["actual_regret"].to_numpy() > float(high_regret_threshold)).astype(np.int8)
    out["prediction_scope"] = "out_of_train"
    return out.drop(columns=["utility", "policy_class_regret"])


def _metric_row(
    *, fold_id: str, split: str, policy_id: str, task: str, model_name: str,
    high_regret_threshold: float, metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fold_id": str(fold_id), "split": str(split), "policy_id": str(policy_id),
        "task": str(task), "model_name": str(model_name),
        "high_regret_threshold": float(high_regret_threshold), **metrics,
    }


def _decile_rows(
    frame: pd.DataFrame,
    *, fold_id: str, split: str, policy_id: str, task: str, model_name: str,
) -> pd.DataFrame:
    d = prediction_deciles(
        frame["actual_target"].to_numpy(), frame["predicted_value"].to_numpy(),
        actual_regret=frame["actual_regret"].to_numpy(), bins=10,
    )
    d.insert(0, "fold_id", str(fold_id))
    d.insert(1, "split", str(split))
    d.insert(2, "policy_id", str(policy_id))
    d.insert(3, "task", str(task))
    d.insert(4, "model_name", str(model_name))
    return d


def _summarize_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols = ["split", "policy_id", "task", "model_name"]
    numeric = [c for c in metrics.columns if c not in ["fold_id", *group_cols] and pd.api.types.is_numeric_dtype(metrics[c])]
    rows: list[dict[str, Any]] = []
    for key, g in metrics.groupby(group_cols, dropna=False, observed=True):
        row = dict(zip(group_cols, key))
        row["folds"] = int(g["fold_id"].nunique())
        for col in numeric:
            x = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(x.mean()) if x.notna().any() else float("nan")
            row[f"{col}_std"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _toy_data(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_train, n_test = 800, 300
    Xtr = rng.normal(size=(n_train, 5))
    Xte = rng.normal(size=(n_test, 5))
    beta = np.array([0.5, -0.35, 0.20, 0.0, 0.15])
    ytr = Xtr @ beta + rng.normal(scale=0.08, size=n_train)
    yte = Xte @ beta + rng.normal(scale=0.08, size=n_test)
    train = pd.DataFrame(Xtr, columns=[f"x{i}" for i in range(5)])
    test = pd.DataFrame(Xte, columns=[f"x{i}" for i in range(5)])
    train["utility"] = ytr
    test["utility"] = yte
    train["regret"] = np.maximum(0.0, -ytr + 0.1 * Xtr[:, 4])
    test["regret"] = np.maximum(0.0, -yte + 0.1 * Xte[:, 4])
    return train, test


def run_phase8_toy_validation() -> dict[str, Any]:
    train, test = _toy_data()
    features = [f"x{i}" for i in range(5)]
    pre = fit_preprocessor(train, features)
    Xtr = pre.transform(train).to_numpy(dtype=float)
    Xte = pre.transform(test).to_numpy(dtype=float)
    ridge = fit_regressor("ridge", Xtr, train["utility"].to_numpy(), seed=7, ridge_alpha=1.0)
    p1 = ridge.predict(Xte)
    p2 = ridge.predict(Xte)
    m = regression_metrics(test["utility"], p1, target_kind="utility")
    if not np.allclose(p1, p2, atol=0.0, rtol=0.0):
        raise AssertionError("Phase-8 deterministic prediction failed")
    if float(m["r2"]) < 0.85:
        raise AssertionError("Phase-8 toy regression should recover an obvious signal")
    threshold = float(np.quantile(train["regret"], 0.90))
    ybin = (train["regret"].to_numpy() > threshold).astype(int)
    clf, fallback = fit_high_regret_classifier(Xtr, ybin, seed=7)
    prob = predict_probability(clf, Xte)
    cm = classification_metrics((test["regret"].to_numpy() > threshold).astype(int), prob, test["regret"].to_numpy())
    if np.isfinite(cm["auroc"]) and float(cm["auroc"]) < 0.75:
        raise AssertionError("Phase-8 toy classifier should rank high-regret states")
    return {
        "status": "PASSED",
        "ridge_test_r2": float(m["r2"]),
        "ridge_test_spearman": float(m["spearman"]),
        "high_regret_threshold": threshold,
        "logistic_test_auroc": float(cm["auroc"]),
        "classifier_fallback": fallback,
        "deterministic_predictions": True,
    }


def build_phase8(cfg: AppConfig, project_root: str | Path) -> Phase8Outputs:
    if cfg.phase != 8 or cfg.phase8 is None:
        raise ValueError("build_phase8 requires phase=8 config")
    root = Path(project_root).resolve()
    require_phase7_checkpoint(root, cfg)
    p8 = cfg.phase8
    p7dir = root / cfg.paths.processed_dir / p8.phase7_processed_subdir
    model_path = p7dir / "model_dataset.parquet"
    membership_path = p7dir / "fold_membership.parquet"
    preprocessors_path = p7dir / "preprocessors.json"
    asset_split_path = p7dir / "asset_split.json"
    for path in (model_path, membership_path, preprocessors_path, asset_split_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase-8 input: {path}")

    logger = logging.getLogger("trading_policy_oracle_repair")
    dataset = pd.read_parquet(model_path)
    membership = pd.read_parquet(membership_path)
    preprocessors_raw = json.loads(preprocessors_path.read_text(encoding="utf-8"))
    asset_split = json.loads(asset_split_path.read_text(encoding="utf-8"))
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.normalize()
    dataset["label_available_date"] = pd.to_datetime(dataset["label_available_date"]).dt.normalize()
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    dataset["asset_id"] = dataset["asset_id"].astype(str).str.upper()
    membership["asset_id"] = membership["asset_id"].astype(str).str.upper()
    if set(MODEL_FEATURE_COLUMNS) - set(dataset.columns):
        raise ValueError("Phase-7 model dataset is missing registered model features")

    out_dir = root / cfg.paths.processed_dir / p8.processed_subdir
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.parquet"
    metrics_path = out_dir / "predictability_metrics.parquet"
    summary_path = out_dir / "predictability_summary.parquet"
    deciles_path = out_dir / "prediction_deciles.parquet"
    thresholds_path = out_dir / "high_regret_thresholds.parquet"
    model_index_path = out_dir / "model_index.parquet"
    audit_path = out_dir / "phase8_audit.json"

    for stale in (predictions_path, metrics_path, summary_path, deciles_path, thresholds_path, model_index_path, audit_path):
        if stale.exists():
            stale.unlink()

    pred_writer = None
    metric_rows: list[dict[str, Any]] = []
    decile_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    model_files: list[Path] = []
    sample_frames: list[pd.DataFrame] = []
    total_prediction_rows = 0
    policy_ids = sorted(dataset["policy_id"].astype(str).unique())
    fold_ids = sorted(membership["fold_id"].astype(str).unique())

    try:
        for fold_no, fold_id in enumerate(fold_ids):
            logger.info("Phase 8 fold %s (%d/%d)", fold_id, fold_no + 1, len(fold_ids))
            if fold_id not in preprocessors_raw:
                raise KeyError(f"missing frozen preprocessor for {fold_id}")
            pre = FrozenPreprocessor.from_dict(preprocessors_raw[fold_id])
            if list(pre.feature_columns) != list(MODEL_FEATURE_COLUMNS):
                raise AssertionError("Phase-8 feature set differs from frozen Phase-7 feature contract")

            mem = membership.loc[membership["fold_id"].astype(str) == fold_id, ["asset_id", "date", "split"]].drop_duplicates()
            fold_data = dataset.merge(mem, on=["asset_id", "date"], how="inner", validate="many_to_one")
            train_all = fold_data.loc[fold_data["split"] == "train"].copy()
            if train_all.empty:
                raise AssertionError(f"empty train split for {fold_id}")
            holdout_assets = set(str(x).upper() for x in asset_split.get("asset_oos_assets", []))
            if set(train_all["asset_id"].astype(str).str.upper()) & holdout_assets:
                raise AssertionError("asset-OOS ticker leaked into Phase-8 training")

            for policy_no, policy_id in enumerate(policy_ids):
                logger.info("Phase 8 %s policy %s (%d/%d)", fold_id, policy_id, policy_no + 1, len(policy_ids))
                tr = train_all.loc[train_all["policy_id"].astype(str) == policy_id].copy()
                if tr.empty:
                    raise AssertionError(f"empty train data for {fold_id}/{policy_id}")
                Xtr = pre.transform(tr).to_numpy(dtype=float)
                y_util = pd.to_numeric(tr["utility"], errors="raise").to_numpy(dtype=float)
                y_regret = pd.to_numeric(tr["policy_class_regret"], errors="raise").to_numpy(dtype=float)
                high_threshold = float(np.quantile(y_regret, float(p8.high_regret_quantile)))
                y_high = (y_regret > high_threshold).astype(int)
                threshold_rows.append({
                    "fold_id": fold_id, "policy_id": policy_id,
                    "quantile": float(p8.high_regret_quantile), "threshold": high_threshold,
                    "train_rows": int(len(tr)), "train_positive_fraction": float(np.mean(y_high)),
                })

                fitted: dict[tuple[str, str], tuple[Any, str | None]] = {}
                for task, y in (("utility", y_util), ("regret", y_regret)):
                    for model_name in ("ridge", "gradient_boosting"):
                        est = fit_regressor(
                            model_name, Xtr, y, seed=int(cfg.seed), ridge_alpha=p8.ridge_alpha,
                            gbt_learning_rate=p8.gbt_learning_rate, gbt_max_iter=p8.gbt_max_iter,
                            gbt_max_leaf_nodes=p8.gbt_max_leaf_nodes,
                            gbt_min_samples_leaf=p8.gbt_min_samples_leaf,
                            gbt_l2_regularization=p8.gbt_l2_regularization,
                        )
                        fitted[(task, model_name)] = (est, None)
                clf, fallback = fit_high_regret_classifier(
                    Xtr, y_high, seed=int(cfg.seed), logistic_c=p8.logistic_c,
                    logistic_max_iter=p8.logistic_max_iter,
                )
                fitted[("high_regret", "logistic")] = (clf, fallback)

                policy_dir = models_dir / fold_id / policy_id
                policy_dir.mkdir(parents=True, exist_ok=True)
                for (task, model_name), (est, fallback_name) in fitted.items():
                    mp = policy_dir / f"{task}__{model_name}.joblib"
                    joblib.dump(est, mp, compress=3)
                    model_files.append(mp)
                    model_rows.append({
                        "fold_id": fold_id, "policy_id": policy_id, "task": task,
                        "model_name": model_name, "train_rows": int(len(tr)),
                        "feature_count": int(len(MODEL_FEATURE_COLUMNS)),
                        "high_regret_threshold": high_threshold,
                        "fallback": fallback_name,
                        "model_path": str(mp.relative_to(root)),
                        "model_sha256": sha256_file(mp),
                        "seed": int(cfg.seed),
                    })

                for split in p8.prediction_splits:
                    ev = fold_data.loc[
                        (fold_data["split"] == split) & (fold_data["policy_id"].astype(str) == policy_id)
                    ].copy()
                    if ev.empty:
                        continue
                    Xev = pre.transform(ev).to_numpy(dtype=float)
                    actual_util = pd.to_numeric(ev["utility"], errors="raise").to_numpy(dtype=float)
                    actual_regret = pd.to_numeric(ev["policy_class_regret"], errors="raise").to_numpy(dtype=float)
                    actual_high = (actual_regret > high_threshold).astype(int)

                    prediction_specs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
                    for task, actual in (("utility", actual_util), ("regret", actual_regret)):
                        for model_name in ("ridge", "gradient_boosting"):
                            pred = np.asarray(fitted[(task, model_name)][0].predict(Xev), dtype=float)
                            prediction_specs.append((task, model_name, actual, pred))
                    prob = predict_probability(fitted[("high_regret", "logistic")][0], Xev)
                    prediction_specs.append(("high_regret", "logistic", actual_high.astype(float), prob))

                    for task, model_name, actual, pred in prediction_specs:
                        pf = _prediction_frame(
                            ev, fold_id=fold_id, split=split, policy_id=policy_id,
                            task=task, model_name=model_name, predicted=pred,
                            actual_target=actual, high_regret_threshold=high_threshold,
                        )
                        if pred_writer is None:
                            pred_writer = _open_parquet_writer(predictions_path, pf)
                        else:
                            _append_parquet(pred_writer, pf)
                        total_prediction_rows += len(pf)
                        if sum(len(x) for x in sample_frames) < int(p8.prediction_sample_rows):
                            remain = int(p8.prediction_sample_rows) - sum(len(x) for x in sample_frames)
                            sample_frames.append(pf.head(remain).copy())

                        if task in {"utility", "regret"}:
                            mm = regression_metrics(actual, pred, target_kind=task)
                        else:
                            mm = classification_metrics(actual_high, pred, actual_regret)
                        metric_rows.append(_metric_row(
                            fold_id=fold_id, split=split, policy_id=policy_id, task=task,
                            model_name=model_name, high_regret_threshold=high_threshold, metrics=mm,
                        ))
                        decile_frames.append(_decile_rows(
                            pf, fold_id=fold_id, split=split, policy_id=policy_id,
                            task=task, model_name=model_name,
                        ))

            logger.info("Phase 8 %s complete", fold_id)
    finally:
        if pred_writer is not None:
            pred_writer.close()

    if pred_writer is None:
        raise AssertionError("Phase-8 generated no predictions")
    metrics = pd.DataFrame(metric_rows)
    deciles = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    thresholds = pd.DataFrame(threshold_rows)
    model_index = pd.DataFrame(model_rows)
    summary = _summarize_fold_metrics(metrics)
    metrics.to_parquet(metrics_path, index=False)
    summary.to_parquet(summary_path, index=False)
    deciles.to_parquet(deciles_path, index=False)
    thresholds.to_parquet(thresholds_path, index=False)
    model_index.to_parquet(model_index_path, index=False)

    csv_files: list[Path] = []
    if p8.write_csv_summaries:
        for frame, name in [
            (metrics, "predictability_metrics.csv"), (summary, "predictability_summary.csv"),
            (thresholds, "high_regret_thresholds.csv"), (model_index, "model_index.csv"),
        ]:
            path = out_dir / name
            frame.to_csv(path, index=False)
            csv_files.append(path)
        if p8.prediction_sample_rows > 0 and sample_frames:
            sample_path = out_dir / "predictions_sample.csv"
            pd.concat(sample_frames, ignore_index=True).head(p8.prediction_sample_rows).to_csv(sample_path, index=False)
            csv_files.append(sample_path)

    test_metrics = metrics.loc[metrics["split"].isin(["same_asset_test", "asset_oos_test"])].copy()
    audit = {
        "phase": 8,
        "scope": "predictability only; models fit on train only; calibration is predicted but never used to refit; no trading/repair",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_exclusions": ["asset_id", "future returns", "oracle action", "future best policy", "label values"],
        "models": {
            "utility_regression": ["ridge", "gradient_boosting"],
            "regret_regression": ["ridge", "gradient_boosting"],
            "high_regret_classification": ["logistic"],
            "gradient_boosting_estimator": "sklearn HistGradientBoostingRegressor",
        },
        "high_regret_quantile": float(p8.high_regret_quantile),
        "high_regret_threshold_rule": "computed separately for each fold/policy from TRAIN policy_class_regret only",
        "prediction_splits": list(p8.prediction_splits),
        "asset_split": asset_split,
        "fold_count": int(len(fold_ids)),
        "policy_count": int(len(policy_ids)),
        "model_artifacts": int(len(model_files)),
        "prediction_rows": int(total_prediction_rows),
        "metric_rows": int(len(metrics)),
        "test_metric_rows": int(len(test_metrics)),
        "decile_rows": int(len(deciles)),
        "train_only_contract": True,
        "calibration_refit": False,
        "deployment_actions_generated": False,
        "inputs": {
            "model_dataset_sha256": sha256_file(model_path),
            "fold_membership_sha256": sha256_file(membership_path),
            "preprocessors_sha256": sha256_file(preprocessors_path),
            "asset_split_sha256": sha256_file(asset_split_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return Phase8Outputs(
        predictions=predictions_path, metrics=metrics_path, summary_metrics=summary_path,
        prediction_deciles=deciles_path, high_regret_thresholds=thresholds_path,
        model_index=model_index_path, audit_json=audit_path, csv_files=tuple(csv_files),
        model_files=tuple(model_files), input_files=(model_path, membership_path, preprocessors_path, asset_split_path),
    )
