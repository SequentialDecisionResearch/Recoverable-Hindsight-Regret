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
from .deployment import (
    FrozenFoldRuntime,
    build_policy_action_template_from_policy,
    continuation_utilities_for_states,
    run_deployment_path,
)
from .features import BASE_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from .hashing import sha256_file
from .policies import BuyHoldPolicy, MomentumPolicy, load_policies
from .preprocessing import FrozenPreprocessor


@dataclass(frozen=True)
class Phase10Outputs:
    deployment_actions: Path
    deployment_ledger: Path
    deployment_fills: Path
    strategy_performance: Path
    decision_diagnostics: Path
    selective_repair_oos: Path
    selective_repair_summary: Path
    meta_selection_oos: Path
    meta_selection_summary: Path
    continuous_actions: Path | None
    continuous_ledger: Path | None
    continuous_fills: Path | None
    continuous_performance: Path | None
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase9_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase9_*.json"))
    return files[-1] if files else None


def require_phase9_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase10 is not None
    cp = _latest_phase9_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase10.require_phase9_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 9 checkpoint found. Run Phase 0 through Phase 9 first.")
    return cp


def _load_runtime(
    *, root: Path, fold_id: str, model_name: str,
    preprocessors_raw: dict[str, Any], model_index: pd.DataFrame,
    calibrators: pd.DataFrame,
) -> FrozenFoldRuntime:
    if fold_id not in preprocessors_raw:
        raise KeyError(f"missing preprocessor for {fold_id}")
    pre = FrozenPreprocessor.from_dict(preprocessors_raw[fold_id])
    if list(pre.feature_columns) != list(MODEL_FEATURE_COLUMNS):
        raise AssertionError("Phase-10 feature contract differs from Phase-7 preprocessor")
    mi = model_index.loc[
        (model_index["fold_id"].astype(str) == str(fold_id))
        & (model_index["task"].astype(str) == "utility")
        & (model_index["model_name"].astype(str) == str(model_name))
    ].copy()
    if mi.empty:
        raise KeyError(f"missing utility models for {fold_id}/{model_name}")
    models: dict[str, Any] = {}
    for row in mi.itertuples(index=False):
        p = root / str(row.model_path)
        if not p.exists():
            raise FileNotFoundError(f"missing frozen model artifact: {p}")
        models[str(row.policy_id)] = joblib.load(p)

    c = calibrators.loc[
        (calibrators["fold_id"].astype(str) == str(fold_id))
        & (calibrators["model_name"].astype(str) == str(model_name))
    ].copy()
    q: dict[tuple[str, str, str], float] = {}
    for row in c.itertuples(index=False):
        b, j = str(row.baseline_policy_id), str(row.candidate_policy_id)
        q[(b, j, "empirical")] = float(row.empirical_residual_quantile)
        q[(b, j, "block_bootstrap")] = float(row.block_bootstrap_residual_quantile)
    expected = len(models) * (len(models) - 1)
    if len(c) != expected:
        raise AssertionError(f"expected {expected} ordered calibrators for {fold_id}/{model_name}, got {len(c)}")
    return FrozenFoldRuntime(str(fold_id), str(model_name), pre, models, q)


def _tag_result(result, *, fold_id: str, split: str, asset_id: str, strategy_id: str):
    ledger, fills, actions = result.ledger.copy(), result.fills.copy(), result.actions.copy()
    for f in (ledger, fills, actions):
        if "fold_id" in f.columns:
            f["bundle_fold_id"] = f["fold_id"].astype(str)
        f["fold_id"] = str(fold_id)
        f["split"] = str(split)
        f["asset_id"] = str(asset_id)
        f["strategy_id"] = str(strategy_id)
    perf = dict(result.performance)
    perf.update({"fold_id": str(fold_id), "split": str(split), "asset_id": str(asset_id), "strategy_id": str(strategy_id)})
    return ledger, fills, actions, perf


def _strategy_id(kind: str, *, baseline: str | None = None, model: str | None = None, method: str | None = None) -> str:
    parts = [kind]
    if baseline is not None:
        parts.append(str(baseline))
    if model is not None:
        parts.append(str(model))
    if method is not None:
        parts.append(str(method))
    return "__".join(parts)


def _summarize(df: pd.DataFrame, group_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["rows"] = int(len(g))
        row["folds"] = int(g["fold_id"].nunique()) if "fold_id" in g else 0
        row["assets"] = int(g["asset_id"].nunique()) if "asset_id" in g else 0
        for col in numeric_cols:
            if col not in g:
                continue
            x = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(x.mean()) if x.notna().any() else float("nan")
            row[f"{col}_std"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _decision_diagnostics_for_path(
    *, actions: pd.DataFrame, asset_market: pd.DataFrame, policies: dict[str, Any],
    horizon: int, cost_bps_per_side: float, test_start: pd.Timestamp, test_end: pd.Timestamp,
    template_cache: dict[tuple[str, str], Any],
) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    a = actions.copy()
    a["decision_date"] = pd.to_datetime(a["decision_date"]).dt.normalize()
    market = asset_market.copy().sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(market["date"])}
    test_start_n = pd.Timestamp(test_start).normalize()
    test_end_n = pd.Timestamp(test_end).normalize()
    # Fold boundaries are calendar dates (for example 2016-12-31), while the
    # canonical market table contains trading dates only.  Resolve the economic
    # test block to the last realized trading date on or before test_end instead
    # of requiring the calendar boundary itself to exist in the market index.
    market_in_test = market.loc[(market["date"] >= test_start_n) & (market["date"] <= test_end_n)]
    if market_in_test.empty:
        return pd.DataFrame()
    end_idx = int(market_in_test.index[-1])
    eligible = a.loc[(a["decision_date"] >= test_start_n) & (a["decision_date"] <= test_end_n)].copy()
    eligible["decision_index"] = eligible["decision_date"].map(pos)
    eligible = eligible.loc[eligible["decision_index"].notna()].copy()
    eligible["decision_index"] = eligible["decision_index"].astype(int)
    eligible = eligible.loc[eligible["decision_index"] + int(horizon) <= int(end_idx)].copy()
    if eligible.empty:
        return pd.DataFrame()

    kind = str(eligible["strategy_type"].iloc[0])
    if kind == "selective_repair":
        eligible["reference_policy_id"] = eligible["baseline_policy_id"].astype(str)
    elif kind in {"meta_selector", "naive_meta"}:
        eligible["reference_policy_id"] = eligible["current_policy_before"].astype(str)
    else:
        return pd.DataFrame()

    needed_policies = sorted(set(eligible["reference_policy_id"].astype(str)) | set(eligible["selected_policy_id"].astype(str)))
    util_by_policy: dict[str, np.ndarray] = {}
    idx = eligible["decision_index"].to_numpy(dtype=int)
    exp = eligible["current_exposure"].to_numpy(dtype=float)
    asset_id = str(market.iloc[0]["asset_id"])
    for pid in needed_policies:
        key = (asset_id, pid)
        if key not in template_cache:
            template_cache[key] = build_policy_action_template_from_policy(market, policies[pid])
        util_by_policy[pid] = continuation_utilities_for_states(
            asset_market=market, action_template=template_cache[key], decision_indices=idx,
            start_exposures=exp, horizon=int(horizon), cost_bps_per_side=float(cost_bps_per_side),
        )
    ref_u = np.array([util_by_policy[p][i] for i, p in enumerate(eligible["reference_policy_id"].astype(str))], dtype=float)
    sel_u = np.array([util_by_policy[p][i] for i, p in enumerate(eligible["selected_policy_id"].astype(str))], dtype=float)
    eligible["actual_reference_utility"] = ref_u
    eligible["actual_selected_utility"] = sel_u
    eligible["actual_advantage"] = sel_u - ref_u
    eligible["intervention_flag"] = eligible["selected_policy_id"].astype(str) != eligible["reference_policy_id"].astype(str)
    eligible["harm_flag"] = eligible["intervention_flag"] & (eligible["actual_advantage"] < 0.0)
    eligible["positive_advantage"] = np.maximum(eligible["actual_advantage"].to_numpy(dtype=float), 0.0)
    eligible["evaluation_horizon"] = int(horizon)
    eligible["evaluation_only"] = True
    return eligible.drop(columns=["decision_index"])


def _diagnostic_summary(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()
    group_cols = ["fold_id", "split", "asset_id", "strategy_id", "strategy_type", "model_name", "lcb_method", "baseline_policy_id"]
    rows = []
    for key, g in diag.groupby(group_cols, observed=True, dropna=False):
        row = dict(zip(group_cols, key))
        intervene = g["intervention_flag"].to_numpy(dtype=bool)
        row["eligible_decisions"] = int(len(g))
        row["interventions"] = int(intervene.sum())
        row["coverage"] = float(intervene.mean()) if len(g) else float("nan")
        row["harm_rate"] = float(np.mean(g.loc[intervene, "actual_advantage"].to_numpy(dtype=float) < 0.0)) if intervene.any() else float("nan")
        row["mean_actual_advantage"] = float(g.loc[intervene, "actual_advantage"].mean()) if intervene.any() else float("nan")
        row["median_actual_advantage"] = float(g.loc[intervene, "actual_advantage"].median()) if intervene.any() else float("nan")
        row["positive_rate"] = float(np.mean(g.loc[intervene, "actual_advantage"].to_numpy(dtype=float) > 0.0)) if intervene.any() else float("nan")
        row["switching_burden"] = float(intervene.sum() / len(g)) if len(g) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_selective(perf: pd.DataFrame, diag_summary: pd.DataFrame) -> pd.DataFrame:
    rep = perf.loc[perf["strategy_type"] == "selective_repair"].copy()
    fix = perf.loc[perf["strategy_type"] == "fixed_policy"].copy()
    rows = []
    for r in rep.itertuples(index=False):
        b = fix.loc[
            (fix["fold_id"].astype(str) == str(r.fold_id)) & (fix["split"].astype(str) == str(r.split))
            & (fix["asset_id"].astype(str) == str(r.asset_id)) & (fix["baseline_policy_id"].astype(str) == str(r.baseline_policy_id))
        ]
        if len(b) != 1:
            raise AssertionError("missing unique fixed baseline for selective repair")
        b0 = b.iloc[0]
        row = {c: getattr(r, c) for c in rep.columns}
        row.update({
            "baseline_final_wealth": float(b0["final_wealth"]),
            "baseline_CAGR": float(b0["CAGR"]),
            "baseline_MaxDD": float(b0["MaxDD"]),
            "baseline_total_turnover": float(b0["total_turnover"]),
            "baseline_total_cost": float(b0["total_cost"]),
            "log_utility_improvement": float(np.log(float(r.final_wealth) / float(b0["final_wealth"]))),
            "CAGR_delta": float(r.CAGR - b0["CAGR"]),
            "MaxDD_delta": float(r.MaxDD - b0["MaxDD"]),
            "turnover_delta": float(r.total_turnover - b0["total_turnover"]),
            "cost_delta": float(r.total_cost - b0["total_cost"]),
        })
        ds = diag_summary.loc[diag_summary["strategy_id"].astype(str) == str(r.strategy_id)]
        ds = ds.loc[(ds["fold_id"].astype(str) == str(r.fold_id)) & (ds["asset_id"].astype(str) == str(r.asset_id))]
        if len(ds) == 1:
            for c in ["eligible_decisions", "interventions", "coverage", "harm_rate", "mean_actual_advantage", "median_actual_advantage", "positive_rate", "switching_burden"]:
                row[c] = ds.iloc[0][c]
        rows.append(row)
    return pd.DataFrame(rows)


def _meta_table(perf: pd.DataFrame, diag_summary: pd.DataFrame, initial_policy: str) -> pd.DataFrame:
    meta = perf.loc[perf["strategy_type"].isin(["meta_selector", "naive_meta"])].copy()
    fix = perf.loc[perf["strategy_type"] == "fixed_policy"].copy()
    rows = []
    for r in meta.itertuples(index=False):
        scope = fix.loc[
            (fix["fold_id"].astype(str) == str(r.fold_id)) & (fix["split"].astype(str) == str(r.split))
            & (fix["asset_id"].astype(str) == str(r.asset_id))
        ].copy()
        ref = scope.loc[scope["baseline_policy_id"].astype(str) == str(initial_policy)]
        if len(ref) != 1:
            raise AssertionError("missing meta initial-policy fixed comparator")
        best = scope.sort_values(["final_wealth", "baseline_policy_id"], ascending=[False, True]).iloc[0]
        b0 = ref.iloc[0]
        row = {c: getattr(r, c) for c in meta.columns}
        row.update({
            "reference_fixed_policy_id": str(initial_policy),
            "reference_final_wealth": float(b0["final_wealth"]),
            "log_utility_improvement_vs_reference": float(np.log(float(r.final_wealth) / float(b0["final_wealth"]))),
            "best_fixed_test_policy_id_diagnostic": str(best["baseline_policy_id"]),
            "best_fixed_test_final_wealth_diagnostic": float(best["final_wealth"]),
            "log_utility_vs_best_fixed_test_diagnostic": float(np.log(float(r.final_wealth) / float(best["final_wealth"]))),
        })
        ds = diag_summary.loc[diag_summary["strategy_id"].astype(str) == str(r.strategy_id)]
        ds = ds.loc[(ds["fold_id"].astype(str) == str(r.fold_id)) & (ds["asset_id"].astype(str) == str(r.asset_id))]
        if len(ds) == 1:
            for c in ["eligible_decisions", "interventions", "coverage", "harm_rate", "mean_actual_advantage", "median_actual_advantage", "positive_rate", "switching_burden"]:
                row[c] = ds.iloc[0][c]
        rows.append(row)
    return pd.DataFrame(rows)


class _ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)
    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def run_phase10_toy_validation() -> dict[str, Any]:
    dates = pd.bdate_range("2026-01-02", periods=10)
    px = np.linspace(100.0, 118.0, len(dates))
    market = pd.DataFrame({
        "date": dates, "asset_id": "AAA", "open": px, "high": px * 1.001, "low": px * .999,
        "close": px, "signal_price": px, "dividend_cash": 0.0, "split_factor": 1.0, "cash_daily_return": 0.0,
    })
    feat = pd.DataFrame({"date": dates, "asset_id": "AAA", "feature_source_date": dates})
    for c in BASE_FEATURE_COLUMNS:
        feat[c] = 0.0
    pre = FrozenPreprocessor(tuple(MODEL_FEATURE_COLUMNS), {c: 0.0 for c in MODEL_FEATURE_COLUMNS}, {c: 0.0 for c in MODEL_FEATURE_COLUMNS}, {c: 1.0 for c in MODEL_FEATURE_COLUMNS}, 100)
    policies = {"buy_hold": BuyHoldPolicy(), "mom_126": MomentumPolicy(lookback=126)}
    models = {"buy_hold": _ConstantModel(.05), "mom_126": _ConstantModel(0.0)}
    q = {
        ("mom_126", "buy_hold", "empirical"): 0.0,
        ("buy_hold", "mom_126", "empirical"): -0.10,
    }
    runtime = FrozenFoldRuntime("toy", "ridge", pre, models, q)
    exec_dates = list(pd.DatetimeIndex(dates[1:]).normalize())
    rm = {d: runtime for d in exec_dates}
    fixed = run_deployment_path(
        asset_market=market, feature_table_asset=feat, execution_dates=exec_dates,
        runtime_by_execution_date={}, policies=policies, strategy_type="fixed_policy",
        model_name=None, lcb_method=None, baseline_policy_id="mom_126", initial_policy_id="mom_126",
        initial_wealth=1.0, initial_exposure=0.0, cost_bps_per_side=5.0,
        deploy_margin=0.0, meta_switch_penalty=0.0005,
    )
    repaired = run_deployment_path(
        asset_market=market, feature_table_asset=feat, execution_dates=exec_dates,
        runtime_by_execution_date=rm, policies=policies, strategy_type="selective_repair",
        model_name="ridge", lcb_method="empirical", baseline_policy_id="mom_126", initial_policy_id="mom_126",
        initial_wealth=1.0, initial_exposure=0.0, cost_bps_per_side=5.0,
        deploy_margin=0.0, meta_switch_penalty=0.0005,
    )
    if repaired.performance["final_wealth"] <= fixed.performance["final_wealth"]:
        raise AssertionError("toy selective repair should improve the intentionally weak baseline")
    if not repaired.actions["deploy_flag"].all():
        raise AssertionError("toy evidence should select buy_hold on every decision")
    return {
        "status": "PASSED",
        "fixed_final_wealth": float(fixed.performance["final_wealth"]),
        "repaired_final_wealth": float(repaired.performance["final_wealth"]),
        "repair_decisions": int(len(repaired.actions)),
        "oracle_or_label_inputs_required": False,
    }


def build_phase10(cfg: AppConfig, project_root: str | Path) -> Phase10Outputs:
    if cfg.phase != 10 or cfg.phase10 is None:
        raise ValueError("build_phase10 requires phase=10 config")
    root = Path(project_root).resolve()
    require_phase9_checkpoint(root, cfg)
    p10 = cfg.phase10
    if p10.primary_lcb_method not in p10.lcb_methods:
        raise ValueError("primary_lcb_method must be included in lcb_methods")
    if int(p10.evaluation_horizon_days) != int(cfg.research_contract.horizon_days):
        raise ValueError("Phase-10 evaluation horizon must match the registered research horizon")
    p1dir = root / cfg.paths.processed_dir / p10.phase1_processed_subdir
    p7dir = root / cfg.paths.processed_dir / p10.phase7_processed_subdir
    p8dir = root / cfg.paths.processed_dir / p10.phase8_processed_subdir
    p9dir = root / cfg.paths.processed_dir / p10.phase9_processed_subdir
    market_path = p1dir / "canonical_market.parquet"
    feature_path = p7dir / "feature_table.parquet"
    folds_path = p7dir / "walk_forward_folds.parquet"
    asset_split_path = p7dir / "asset_split.json"
    preprocessors_path = p7dir / "preprocessors.json"
    model_index_path = p8dir / "model_index.parquet"
    calibrators_path = p9dir / "pairwise_calibrators.parquet"
    policies_path = root / p10.policies_config
    input_files = (market_path, feature_path, folds_path, asset_split_path, preprocessors_path, model_index_path, calibrators_path, policies_path)
    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase-10 deployment input: {path}")

    logger = logging.getLogger("trading_policy_oracle_repair")
    market = pd.read_parquet(market_path)
    features = pd.read_parquet(feature_path)
    folds = pd.read_parquet(folds_path)
    asset_split = json.loads(asset_split_path.read_text(encoding="utf-8"))
    preprocessors_raw = json.loads(preprocessors_path.read_text(encoding="utf-8"))
    model_index = pd.read_parquet(model_index_path)
    calibrators = pd.read_parquet(calibrators_path)
    policies = load_policies(policies_path)
    policy_ids = sorted(policies)
    if p10.meta_initial_policy_id not in policies:
        raise ValueError("meta_initial_policy_id is not a registered policy")

    for frame in (market, features):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["asset_id"] = frame["asset_id"].astype(str).str.upper()
    for c in ["train_start", "train_end", "calibration_start", "calibration_end", "test_start", "test_end"]:
        if c in folds.columns:
            folds[c] = pd.to_datetime(folds[c]).dt.normalize()
    if "feature_source_date" in features:
        features["feature_source_date"] = pd.to_datetime(features["feature_source_date"]).dt.normalize()
        if (features["feature_source_date"] > features["date"]).any():
            raise AssertionError("Phase-10 detected future-dated features")

    fold_ids = sorted(folds["fold_id"].astype(str).unique())
    train_assets = [str(x).upper() for x in asset_split.get("train_assets", [])]
    holdout_assets = [str(x).upper() for x in asset_split.get("asset_oos_assets", [])]
    if not train_assets or not holdout_assets:
        raise ValueError("Phase-7 asset split is incomplete")
    runtime_cache: dict[tuple[str, str], FrozenFoldRuntime] = {}
    def runtime(fold_id: str, model_name: str) -> FrozenFoldRuntime:
        key = (str(fold_id), str(model_name))
        if key not in runtime_cache:
            runtime_cache[key] = _load_runtime(
                root=root, fold_id=fold_id, model_name=model_name,
                preprocessors_raw=preprocessors_raw, model_index=model_index, calibrators=calibrators,
            )
        return runtime_cache[key]

    out_dir = root / cfg.paths.processed_dir / p10.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "actions": out_dir / "deployment_actions.parquet",
        "ledger": out_dir / "deployment_ledger.parquet",
        "fills": out_dir / "deployment_fills.parquet",
        "performance": out_dir / "strategy_performance.parquet",
        "diagnostics": out_dir / "decision_diagnostics.parquet",
        "repair": out_dir / "selective_repair_oos.parquet",
        "repair_summary": out_dir / "selective_repair_summary.parquet",
        "meta": out_dir / "meta_selection_oos.parquet",
        "meta_summary": out_dir / "meta_selection_summary.parquet",
        "continuous_actions": out_dir / "continuous_actions.parquet",
        "continuous_ledger": out_dir / "continuous_ledger.parquet",
        "continuous_fills": out_dir / "continuous_fills.parquet",
        "continuous_performance": out_dir / "continuous_performance.parquet",
        "audit": out_dir / "phase10_audit.json",
    }
    for p in paths.values():
        if p.exists(): p.unlink()

    ledger_frames, fill_frames, action_frames = [], [], []
    perf_rows: list[dict[str, Any]] = []
    template_cache: dict[tuple[str, str], Any] = {}
    diag_frames: list[pd.DataFrame] = []
    penalty = float(p10.meta_switch_penalty_bps) / 10000.0

    for fold_no, fold_id in enumerate(fold_ids):
        logger.info("Phase 10 fold %s (%d/%d)", fold_id, fold_no + 1, len(fold_ids))
        fr = folds.loc[folds["fold_id"].astype(str) == fold_id]
        if len(fr) != 1:
            raise AssertionError(f"missing unique walk-forward row for {fold_id}")
        test_start = pd.Timestamp(fr.iloc[0]["test_start"]).normalize()
        test_end = pd.Timestamp(fr.iloc[0]["test_end"]).normalize()
        for split in p10.evaluation_splits:
            assets = train_assets if split == "same_asset_test" else holdout_assets
            for asset_id in sorted(assets):
                am = market.loc[market["asset_id"] == asset_id].sort_values("date").reset_index(drop=True)
                af = features.loc[features["asset_id"] == asset_id].sort_values("date").reset_index(drop=True)
                dates = sorted(pd.DatetimeIndex(am.loc[(am["date"] >= test_start) & (am["date"] <= test_end), "date"]).normalize())
                if not dates:
                    continue

                # Fixed comparators: one per registered policy, independent of ML branch.
                for baseline in policy_ids:
                    sid = _strategy_id("fixed", baseline=baseline)
                    res = run_deployment_path(
                        asset_market=am, feature_table_asset=af, execution_dates=dates,
                        runtime_by_execution_date={}, policies=policies, strategy_type="fixed_policy",
                        model_name=None, lcb_method=None, baseline_policy_id=baseline, initial_policy_id=baseline,
                        initial_wealth=p10.initial_wealth, initial_exposure=p10.initial_exposure,
                        cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                        deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                        annualization_days=p10.annualization_days,
                    )
                    le, fi, ac, pe = _tag_result(res, fold_id=fold_id, split=split, asset_id=asset_id, strategy_id=sid)
                    ledger_frames.append(le); fill_frames.append(fi); action_frames.append(ac); perf_rows.append(pe)

                for model_name in p10.utility_models:
                    rt = runtime(fold_id, model_name)
                    rm = {pd.Timestamp(d): rt for d in dates}
                    for method in p10.lcb_methods:
                        for baseline in policy_ids:
                            sid = _strategy_id("repair", baseline=baseline, model=model_name, method=method)
                            res = run_deployment_path(
                                asset_market=am, feature_table_asset=af, execution_dates=dates,
                                runtime_by_execution_date=rm, policies=policies, strategy_type="selective_repair",
                                model_name=model_name, lcb_method=method, baseline_policy_id=baseline,
                                initial_policy_id=baseline, initial_wealth=p10.initial_wealth,
                                initial_exposure=p10.initial_exposure,
                                cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                                deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                                annualization_days=p10.annualization_days,
                            )
                            le, fi, ac, pe = _tag_result(res, fold_id=fold_id, split=split, asset_id=asset_id, strategy_id=sid)
                            ledger_frames.append(le); fill_frames.append(fi); action_frames.append(ac); perf_rows.append(pe)

                        sid = _strategy_id("meta", model=model_name, method=method)
                        res = run_deployment_path(
                            asset_market=am, feature_table_asset=af, execution_dates=dates,
                            runtime_by_execution_date=rm, policies=policies, strategy_type="meta_selector",
                            model_name=model_name, lcb_method=method, baseline_policy_id=None,
                            initial_policy_id=p10.meta_initial_policy_id, initial_wealth=p10.initial_wealth,
                            initial_exposure=p10.initial_exposure,
                            cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                            deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                            annualization_days=p10.annualization_days,
                        )
                        le, fi, ac, pe = _tag_result(res, fold_id=fold_id, split=split, asset_id=asset_id, strategy_id=sid)
                        ledger_frames.append(le); fill_frames.append(fi); action_frames.append(ac); perf_rows.append(pe)

                    if p10.include_naive_meta:
                        sid = _strategy_id("naive_meta", model=model_name)
                        res = run_deployment_path(
                            asset_market=am, feature_table_asset=af, execution_dates=dates,
                            runtime_by_execution_date=rm, policies=policies, strategy_type="naive_meta",
                            model_name=model_name, lcb_method=None, baseline_policy_id=None,
                            initial_policy_id=p10.meta_initial_policy_id, initial_wealth=p10.initial_wealth,
                            initial_exposure=p10.initial_exposure,
                            cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                            deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                            annualization_days=p10.annualization_days,
                        )
                        le, fi, ac, pe = _tag_result(res, fold_id=fold_id, split=split, asset_id=asset_id, strategy_id=sid)
                        ledger_frames.append(le); fill_frames.append(fi); action_frames.append(ac); perf_rows.append(pe)
        logger.info("Phase 10 %s deployment complete", fold_id)

    actions_df = pd.concat(action_frames, ignore_index=True)
    ledger_df = pd.concat(ledger_frames, ignore_index=True)
    nonempty_fill_frames = [f for f in fill_frames if f is not None and not f.empty]
    fills_df = pd.concat(nonempty_fill_frames, ignore_index=True) if nonempty_fill_frames else pd.DataFrame()
    perf_df = pd.DataFrame(perf_rows)
    actions_df.to_parquet(paths["actions"], index=False)
    ledger_df.to_parquet(paths["ledger"], index=False)
    fills_df.to_parquet(paths["fills"], index=False)
    perf_df.to_parquet(paths["performance"], index=False)
    frozen_actions_sha = sha256_file(paths["actions"])

    # Evaluation-only H-day diagnostic utilities are computed AFTER deployment actions are frozen.
    adaptive_actions = actions_df.loc[actions_df["strategy_type"].isin(["selective_repair", "meta_selector", "naive_meta"])].copy()
    for (fold_id, split, asset_id, strategy_id), a in adaptive_actions.groupby(["fold_id", "split", "asset_id", "strategy_id"], observed=True):
        fr = folds.loc[folds["fold_id"].astype(str) == str(fold_id)]
        if len(fr) != 1:
            continue
        am = market.loc[market["asset_id"] == str(asset_id)].sort_values("date").reset_index(drop=True)
        dd = _decision_diagnostics_for_path(
            actions=a, asset_market=am, policies=policies,
            horizon=p10.evaluation_horizon_days,
            cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
            test_start=pd.Timestamp(fr.iloc[0]["test_start"]), test_end=pd.Timestamp(fr.iloc[0]["test_end"]),
            template_cache=template_cache,
        )
        if not dd.empty:
            diag_frames.append(dd)
    diag_df = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    if sha256_file(paths["actions"]) != frozen_actions_sha:
        raise AssertionError("evaluation diagnostics mutated frozen deployment actions")
    diag_df.to_parquet(paths["diagnostics"], index=False)
    diag_summary = _diagnostic_summary(diag_df)

    repair_df = _paired_selective(perf_df, diag_summary)
    repair_summary = _summarize(
        repair_df,
        ["split", "model_name", "lcb_method", "baseline_policy_id"],
        ["log_utility_improvement", "CAGR_delta", "MaxDD_delta", "turnover_delta", "cost_delta", "coverage", "harm_rate", "mean_actual_advantage", "switching_burden"],
    )
    meta_df = _meta_table(perf_df, diag_summary, p10.meta_initial_policy_id)
    meta_summary = _summarize(
        meta_df,
        ["split", "strategy_type", "model_name", "lcb_method"],
        ["log_utility_improvement_vs_reference", "log_utility_vs_best_fixed_test_diagnostic", "coverage", "harm_rate", "mean_actual_advantage", "switching_burden", "total_turnover", "total_cost"],
    )
    repair_df.to_parquet(paths["repair"], index=False)
    repair_summary.to_parquet(paths["repair_summary"], index=False)
    meta_df.to_parquet(paths["meta"], index=False)
    meta_summary.to_parquet(paths["meta_summary"], index=False)

    cont_actions_path = cont_ledger_path = cont_fills_path = cont_perf_path = None
    if p10.run_continuous_pseudo_live:
        ca, cl, cf, cp = [], [], [], []
        all_assets = [(a, "same_asset_test") for a in train_assets] + [(a, "asset_oos_test") for a in holdout_assets]
        for asset_id, split in sorted(all_assets):
            am = market.loc[market["asset_id"] == asset_id].sort_values("date").reset_index(drop=True)
            af = features.loc[features["asset_id"] == asset_id].sort_values("date").reset_index(drop=True)
            map_rows = []
            for fr in folds.itertuples(index=False):
                ds = am.loc[(am["date"] >= pd.Timestamp(fr.test_start)) & (am["date"] <= pd.Timestamp(fr.test_end)), "date"]
                map_rows.extend({"date": pd.Timestamp(d).normalize(), "fold_id": str(fr.fold_id)} for d in ds)
            date_fold = pd.DataFrame(map_rows)
            if date_fold.empty:
                continue
            if date_fold["date"].duplicated().any():
                raise AssertionError("OOS execution date maps to multiple folds")
            dates = sorted(pd.DatetimeIndex(date_fold["date"]).normalize())
            for baseline in policy_ids:
                sid = _strategy_id("continuous_fixed", baseline=baseline)
                res = run_deployment_path(
                    asset_market=am, feature_table_asset=af, execution_dates=dates,
                    runtime_by_execution_date={}, policies=policies, strategy_type="fixed_policy",
                    model_name=None, lcb_method=None, baseline_policy_id=baseline, initial_policy_id=baseline,
                    initial_wealth=p10.initial_wealth, initial_exposure=p10.initial_exposure,
                    cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                    deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                    annualization_days=p10.annualization_days,
                )
                le, fi, ac, pe = _tag_result(res, fold_id="continuous_2016_2025", split=split, asset_id=asset_id, strategy_id=sid)
                cl.append(le); cf.append(fi); ca.append(ac); cp.append(pe)
            for model_name in p10.utility_models:
                rm = {
                    pd.Timestamp(r.date).normalize(): runtime(str(r.fold_id), model_name)
                    for r in date_fold.itertuples(index=False)
                }
                for method in p10.lcb_methods:
                    for baseline in policy_ids:
                        sid = _strategy_id("continuous_repair", baseline=baseline, model=model_name, method=method)
                        res = run_deployment_path(
                            asset_market=am, feature_table_asset=af, execution_dates=dates,
                            runtime_by_execution_date=rm, policies=policies, strategy_type="selective_repair",
                            model_name=model_name, lcb_method=method, baseline_policy_id=baseline, initial_policy_id=baseline,
                            initial_wealth=p10.initial_wealth, initial_exposure=p10.initial_exposure,
                            cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                            deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                            annualization_days=p10.annualization_days,
                        )
                        le, fi, ac, pe = _tag_result(res, fold_id="continuous_2016_2025", split=split, asset_id=asset_id, strategy_id=sid)
                        cl.append(le); cf.append(fi); ca.append(ac); cp.append(pe)
                    sid = _strategy_id("continuous_meta", model=model_name, method=method)
                    res = run_deployment_path(
                        asset_market=am, feature_table_asset=af, execution_dates=dates,
                        runtime_by_execution_date=rm, policies=policies, strategy_type="meta_selector",
                        model_name=model_name, lcb_method=method, baseline_policy_id=None,
                        initial_policy_id=p10.meta_initial_policy_id, initial_wealth=p10.initial_wealth,
                        initial_exposure=p10.initial_exposure,
                        cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                        deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                        annualization_days=p10.annualization_days,
                    )
                    le, fi, ac, pe = _tag_result(res, fold_id="continuous_2016_2025", split=split, asset_id=asset_id, strategy_id=sid)
                    cl.append(le); cf.append(fi); ca.append(ac); cp.append(pe)
                if p10.include_naive_meta:
                    sid = _strategy_id("continuous_naive_meta", model=model_name)
                    res = run_deployment_path(
                        asset_market=am, feature_table_asset=af, execution_dates=dates,
                        runtime_by_execution_date=rm, policies=policies, strategy_type="naive_meta",
                        model_name=model_name, lcb_method=None, baseline_policy_id=None,
                        initial_policy_id=p10.meta_initial_policy_id, initial_wealth=p10.initial_wealth,
                        initial_exposure=p10.initial_exposure,
                        cost_bps_per_side=cfg.research_contract.cost_bps_per_side,
                        deploy_margin=p10.deploy_margin, meta_switch_penalty=penalty,
                        annualization_days=p10.annualization_days,
                    )
                    le, fi, ac, pe = _tag_result(res, fold_id="continuous_2016_2025", split=split, asset_id=asset_id, strategy_id=sid)
                    cl.append(le); cf.append(fi); ca.append(ac); cp.append(pe)
            logger.info("Phase 10 continuous pseudo-live %s complete", asset_id)
        pd.concat(ca, ignore_index=True).to_parquet(paths["continuous_actions"], index=False)
        pd.concat(cl, ignore_index=True).to_parquet(paths["continuous_ledger"], index=False)
        pd.concat(cf, ignore_index=True).to_parquet(paths["continuous_fills"], index=False)
        pd.DataFrame(cp).to_parquet(paths["continuous_performance"], index=False)
        cont_actions_path, cont_ledger_path, cont_fills_path, cont_perf_path = paths["continuous_actions"], paths["continuous_ledger"], paths["continuous_fills"], paths["continuous_performance"]

    csv_files: list[Path] = []
    if p10.write_csv_summaries:
        for frame, name in [
            (repair_summary, "selective_repair_summary.csv"),
            (meta_summary, "meta_selection_summary.csv"),
            (perf_df, "strategy_performance.csv"),
        ]:
            p = out_dir / name; frame.to_csv(p, index=False); csv_files.append(p)
        if p10.daily_sample_rows > 0:
            p = out_dir / "deployment_actions_sample.csv"
            actions_df.head(int(p10.daily_sample_rows)).to_csv(p, index=False); csv_files.append(p)

    audit = {
        "phase": 10,
        "scope": "frozen pseudo-live selective repair and regret-aware meta-policy selection",
        "algorithms": {
            "A": "policy-specific selective repair: best calibrated candidate is used for the next target only when LCB > deploy_margin",
            "B": "stateful meta-selector: predicted-utility argmax with fixed policy-identity switch penalty and best-vs-current LCB hysteresis",
        },
        "utility_models": list(p10.utility_models),
        "lcb_methods": list(p10.lcb_methods),
        "primary_lcb_method": p10.primary_lcb_method,
        "deploy_margin": float(p10.deploy_margin),
        "meta_switch_penalty_bps": float(p10.meta_switch_penalty_bps),
        "meta_initial_policy_id": p10.meta_initial_policy_id,
        "cost_contract": "continuation-utility models were trained on labels with execution costs already included; no second estimated transaction-cost subtraction is applied; the shared execution engine charges realized turnover costs",
        "test_oracle_files_read": False,
        "test_label_files_read": False,
        "phase8_test_predictions_read": False,
        "deployment_input_contract": "Phase-1 market + Phase-7 past-only features/preprocessors/walk-forward folds/asset split + Phase-8 frozen utility model artifacts + Phase-9 calibration-only residual quantiles",
        "actions_frozen_before_hindsight_diagnostics": True,
        "frozen_deployment_actions_sha256": frozen_actions_sha,
        "evaluation_horizon_days": int(p10.evaluation_horizon_days),
        "evaluation_boundary": "decision-level actual advantage is computed only when t+H remains within the same test block",
        "continuous_pseudo_live": bool(p10.run_continuous_pseudo_live),
        "fold_count": int(len(fold_ids)),
        "policy_count": int(len(policy_ids)),
        "deployment_action_rows": int(len(actions_df)),
        "deployment_ledger_rows": int(len(ledger_df)),
        "decision_diagnostic_rows": int(len(diag_df)),
        "inputs": {str(p.relative_to(root)): sha256_file(p) for p in input_files},
    }
    paths["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase10Outputs(
        deployment_actions=paths["actions"], deployment_ledger=paths["ledger"], deployment_fills=paths["fills"],
        strategy_performance=paths["performance"], decision_diagnostics=paths["diagnostics"],
        selective_repair_oos=paths["repair"], selective_repair_summary=paths["repair_summary"],
        meta_selection_oos=paths["meta"], meta_selection_summary=paths["meta_summary"],
        continuous_actions=cont_actions_path, continuous_ledger=cont_ledger_path, continuous_fills=cont_fills_path,
        continuous_performance=cont_perf_path, audit_json=paths["audit"], csv_files=tuple(csv_files), input_files=input_files,
    )
