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
from .inference import aggregate_decision_bootstrap, aggregate_path_bootstrap, paired_path_arrays, stable_seed


@dataclass(frozen=True)
class Phase11Outputs:
    paired_bootstrap_inference: Path
    decision_bootstrap_inference: Path
    continuous_bootstrap_inference: Path
    annual_stability: Path
    asset_stability: Path
    meta_annual_stability: Path
    publication_selective_repair: Path
    publication_meta: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase10_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase10_*.json"))
    return files[-1] if files else None


def require_phase10_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase11 is not None
    cp = _latest_phase10_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase11.require_phase10_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 10 checkpoint found. Run Phase 0 through Phase 10 first.")
    return cp


def _ledger_lookup(ledger: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    out: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    keys = ["fold_id", "split", "asset_id", "strategy_id"]
    for key, g in ledger.groupby(keys, observed=True, dropna=False):
        k = tuple(str(x) for x in key)
        if k in out:
            raise AssertionError(f"duplicate ledger path key: {k}")
        out[k] = g.copy().sort_values("date").reset_index(drop=True)
    return out


def _diag_lookup(diag: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    out: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    if diag.empty:
        return out
    keys = ["fold_id", "split", "asset_id", "strategy_id"]
    for key, g in diag.groupby(keys, observed=True, dropna=False):
        out[tuple(str(x) for x in key)] = g.copy().sort_values("decision_date").reset_index(drop=True)
    return out


def _reference_strategy_id(strategy_type: str, baseline_policy_id: str) -> str:
    if strategy_type == "selective_repair":
        return f"fixed__{baseline_policy_id}"
    if strategy_type in {"meta_selector", "naive_meta"}:
        return "fixed__buy_hold"
    raise ValueError(f"unsupported adaptive strategy_type: {strategy_type}")


def _continuous_reference_strategy_id(strategy_type: str, baseline_policy_id: str) -> str:
    if strategy_type == "selective_repair":
        return f"continuous_fixed__{baseline_policy_id}"
    if strategy_type in {"meta_selector", "naive_meta"}:
        return "continuous_fixed__buy_hold"
    raise ValueError(f"unsupported continuous strategy_type: {strategy_type}")


def _fold_path_inference(
    *, source: pd.DataFrame, ledger_lookup: dict, diag_lookup: dict,
    group_cols: list[str], block_length: int, resamples: int,
    annualization_days: int, seed: int, confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for group_key, g in source.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        meta = dict(zip(group_cols, group_key))
        pairs: list[tuple[str, dict[str, np.ndarray]]] = []
        dpaths: list[tuple[str, pd.DataFrame]] = []
        for r in g.itertuples(index=False):
            fold_id, split, asset_id, sid = str(r.fold_id), str(r.split), str(r.asset_id), str(r.strategy_id)
            strategy_type = str(r.strategy_type)
            baseline = str(r.baseline_policy_id)
            ref_sid = _reference_strategy_id(strategy_type, baseline)
            adp_key = (fold_id, split, asset_id, sid)
            ref_key = (fold_id, split, asset_id, ref_sid)
            if adp_key not in ledger_lookup:
                raise KeyError(f"missing adaptive ledger path {adp_key}")
            if ref_key not in ledger_lookup:
                raise KeyError(f"missing paired reference ledger path {ref_key}")
            pair_key = "|".join(adp_key)
            pairs.append((pair_key, paired_path_arrays(ledger_lookup[ref_key], ledger_lookup[adp_key])))
            if adp_key in diag_lookup:
                dpaths.append((pair_key, diag_lookup[adp_key]))
        group_seed = stable_seed(seed, "group", *group_key)
        prow = dict(meta)
        prow.update(aggregate_path_bootstrap(
            pairs, block_length=block_length, resamples=resamples,
            annualization_days=annualization_days, seed=group_seed,
            confidence_level=confidence_level,
        ))
        path_rows.append(prow)
        if dpaths:
            drow = dict(meta)
            drow.update(aggregate_decision_bootstrap(
                dpaths, block_length=block_length, resamples=resamples,
                seed=stable_seed(seed, "decision_group", *group_key), confidence_level=confidence_level,
            ))
            decision_rows.append(drow)
    return pd.DataFrame(path_rows), pd.DataFrame(decision_rows)


def _continuous_inference(
    *, perf: pd.DataFrame, ledger_lookup: dict,
    block_length: int, resamples: int, annualization_days: int,
    seed: int, confidence_level: float,
) -> pd.DataFrame:
    adaptive = perf.loc[perf["strategy_type"].isin(["selective_repair", "meta_selector", "naive_meta"])].copy()
    if adaptive.empty:
        return pd.DataFrame()
    group_cols = ["split", "strategy_type", "model_name", "lcb_method", "baseline_policy_id"]
    rows: list[dict[str, Any]] = []
    for group_key, g in adaptive.groupby(group_cols, observed=True, dropna=False):
        meta = dict(zip(group_cols, group_key))
        pairs = []
        for r in g.itertuples(index=False):
            fold_id, split, asset_id, sid = str(r.fold_id), str(r.split), str(r.asset_id), str(r.strategy_id)
            baseline = str(r.baseline_policy_id)
            ref_sid = _continuous_reference_strategy_id(str(r.strategy_type), baseline)
            adp_key = (fold_id, split, asset_id, sid)
            ref_key = (fold_id, split, asset_id, ref_sid)
            if adp_key not in ledger_lookup or ref_key not in ledger_lookup:
                raise KeyError(f"missing continuous paired path {adp_key} / {ref_key}")
            pairs.append(("|".join(adp_key), paired_path_arrays(ledger_lookup[ref_key], ledger_lookup[adp_key])))
        row = dict(meta)
        row.update(aggregate_path_bootstrap(
            pairs, block_length=block_length, resamples=resamples,
            annualization_days=annualization_days,
            seed=stable_seed(seed, "continuous", *group_key), confidence_level=confidence_level,
        ))
        rows.append(row)
    return pd.DataFrame(rows)


def _annual_stability(repair: pd.DataFrame) -> pd.DataFrame:
    cols = ["split", "model_name", "lcb_method", "baseline_policy_id", "fold_id"]
    rows = []
    for key, g in repair.groupby(cols, observed=True, dropna=False):
        x = pd.to_numeric(g["log_utility_improvement"], errors="coerce")
        rows.append({
            **dict(zip(cols, key)),
            "assets": int(g["asset_id"].nunique()),
            "mean_log_utility_improvement": float(x.mean()),
            "median_log_utility_improvement": float(x.median()),
            "positive_asset_fraction": float((x > 0).mean()),
            "mean_CAGR_delta": float(pd.to_numeric(g["CAGR_delta"], errors="coerce").mean()),
            "mean_MaxDD_delta": float(pd.to_numeric(g["MaxDD_delta"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def _asset_stability(repair: pd.DataFrame) -> pd.DataFrame:
    cols = ["split", "model_name", "lcb_method", "baseline_policy_id", "asset_id"]
    rows = []
    for key, g in repair.groupby(cols, observed=True, dropna=False):
        x = pd.to_numeric(g["log_utility_improvement"], errors="coerce")
        rows.append({
            **dict(zip(cols, key)),
            "folds": int(g["fold_id"].nunique()),
            "mean_log_utility_improvement": float(x.mean()),
            "median_log_utility_improvement": float(x.median()),
            "positive_fold_fraction": float((x > 0).mean()),
            "mean_CAGR_delta": float(pd.to_numeric(g["CAGR_delta"], errors="coerce").mean()),
            "mean_MaxDD_delta": float(pd.to_numeric(g["MaxDD_delta"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def _meta_annual_stability(meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["split", "strategy_type", "model_name", "lcb_method", "fold_id"]
    rows = []
    for key, g in meta.groupby(cols, observed=True, dropna=False):
        x = pd.to_numeric(g["log_utility_improvement_vs_reference"], errors="coerce")
        rows.append({
            **dict(zip(cols, key)),
            "assets": int(g["asset_id"].nunique()),
            "mean_log_utility_improvement_vs_reference": float(x.mean()),
            "positive_asset_fraction": float((x > 0).mean()),
        })
    return pd.DataFrame(rows)


def _merge_publication_repair(path_inf: pd.DataFrame, decision_inf: pd.DataFrame, annual: pd.DataFrame, assets: pd.DataFrame, continuous: pd.DataFrame) -> pd.DataFrame:
    key = ["split", "model_name", "lcb_method", "baseline_policy_id"]
    out = path_inf.copy()
    if not decision_inf.empty:
        keep = key + [c for c in decision_inf.columns if c not in key and c not in {"bootstrap_resamples", "block_length_days"}]
        out = out.merge(decision_inf[keep], on=key, how="left", suffixes=("", "_decision"))
    if not annual.empty:
        s = annual.groupby(key, observed=True).agg(
            years=("fold_id", "nunique"),
            positive_year_fraction=("mean_log_utility_improvement", lambda x: float(np.mean(np.asarray(x) > 0))),
            worst_year_mean_log_utility=("mean_log_utility_improvement", "min"),
            best_year_mean_log_utility=("mean_log_utility_improvement", "max"),
        ).reset_index()
        out = out.merge(s, on=key, how="left")
    if not assets.empty:
        s = assets.groupby(key, observed=True).agg(
            asset_count=("asset_id", "nunique"),
            positive_asset_fraction=("mean_log_utility_improvement", lambda x: float(np.mean(np.asarray(x) > 0))),
        ).reset_index()
        out = out.merge(s, on=key, how="left")
    if not continuous.empty:
        c = continuous.loc[continuous["strategy_type"] == "selective_repair"].copy()
        rename = {x: f"continuous_{x}" for x in c.columns if x not in key + ["strategy_type"]}
        c = c.drop(columns=["strategy_type"]).rename(columns=rename)
        out = out.merge(c, on=key, how="left")
    return out


def _merge_publication_meta(path_inf: pd.DataFrame, decision_inf: pd.DataFrame, annual: pd.DataFrame, continuous: pd.DataFrame) -> pd.DataFrame:
    key = ["split", "strategy_type", "model_name", "lcb_method", "baseline_policy_id"]
    out = path_inf.copy()
    if not decision_inf.empty:
        keep = key + [c for c in decision_inf.columns if c not in key and c not in {"bootstrap_resamples", "block_length_days"}]
        out = out.merge(decision_inf[keep], on=key, how="left", suffixes=("", "_decision"))
    if not annual.empty:
        akey = ["split", "strategy_type", "model_name", "lcb_method"]
        s = annual.groupby(akey, observed=True).agg(
            years=("fold_id", "nunique"),
            positive_year_fraction=("mean_log_utility_improvement_vs_reference", lambda x: float(np.mean(np.asarray(x) > 0))),
        ).reset_index()
        out = out.merge(s, on=akey, how="left")
    if not continuous.empty:
        c = continuous.loc[continuous["strategy_type"].isin(["meta_selector", "naive_meta"])].copy()
        rename = {x: f"continuous_{x}" for x in c.columns if x not in key}
        c = c.rename(columns=rename)
        out = out.merge(c, on=key, how="left")
    return out


def run_phase11_toy_validation() -> dict[str, Any]:
    from .inference import aggregate_path_bootstrap

    n = 240
    rng = np.random.default_rng(7)
    common = rng.normal(0.0002, 0.01, n)
    # Adaptive has a small but persistent paired advantage. Because the same
    # bootstrap indices are used for both paths, paired inference should recover it.
    ref = common
    adp = common + 0.0008
    arrays = {
        "dates": pd.bdate_range("2020-01-01", periods=n).to_numpy(),
        "reference_log_returns": ref,
        "adaptive_log_returns": adp,
        "reference_turnover": np.zeros(n),
        "adaptive_turnover": np.full(n, 0.0001),
        "reference_cost_fraction": np.zeros(n),
        "adaptive_cost_fraction": np.full(n, 0.00001),
    }
    out = aggregate_path_bootstrap(
        [("toy", arrays)], block_length=20, resamples=300,
        annualization_days=252, seed=11, confidence_level=0.95,
    )
    expected = n * 0.0008
    if abs(out["log_utility_improvement_observed"] - expected) > 1e-12:
        raise AssertionError("toy paired log-utility improvement mismatch")
    if out["log_utility_improvement_ci_low"] <= 0.0:
        raise AssertionError("toy positive paired effect was not recovered")
    return {
        "status": "PASSED",
        "observed_log_utility_improvement": float(out["log_utility_improvement_observed"]),
        "ci_low": float(out["log_utility_improvement_ci_low"]),
        "ci_high": float(out["log_utility_improvement_ci_high"]),
        "bootstrap_prob_gt_zero": float(out["log_utility_improvement_prob_gt_zero"]),
        "block_length_days": 20,
    }


def build_phase11(cfg: AppConfig, root: Path) -> Phase11Outputs:
    assert cfg.phase11 is not None
    p11 = cfg.phase11
    require_phase10_checkpoint(root, cfg)
    logger = logging.getLogger("trading_policy_oracle_repair")

    p10dir = root / cfg.paths.processed_dir / p11.phase10_processed_subdir
    paths_in = {
        "ledger": p10dir / "deployment_ledger.parquet",
        "diagnostics": p10dir / "decision_diagnostics.parquet",
        "repair": p10dir / "selective_repair_oos.parquet",
        "meta": p10dir / "meta_selection_oos.parquet",
        "continuous_ledger": p10dir / "continuous_ledger.parquet",
        "continuous_performance": p10dir / "continuous_performance.parquet",
    }
    required = [paths_in["ledger"], paths_in["diagnostics"], paths_in["repair"], paths_in["meta"]]
    if p11.include_continuous_inference:
        required += [paths_in["continuous_ledger"], paths_in["continuous_performance"]]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing Phase-11 frozen Phase-10 input: {p}")
    input_files = tuple(required)

    ledger = pd.read_parquet(paths_in["ledger"])
    diag = pd.read_parquet(paths_in["diagnostics"])
    repair = pd.read_parquet(paths_in["repair"])
    meta = pd.read_parquet(paths_in["meta"])
    for f in (ledger, diag, repair, meta):
        if "asset_id" in f:
            f["asset_id"] = f["asset_id"].astype(str)
    lookup = _ledger_lookup(ledger)
    dlookup = _diag_lookup(diag)

    logger.info("Phase 11 paired block bootstrap: selective repair")
    repair_path, repair_decision = _fold_path_inference(
        source=repair, ledger_lookup=lookup, diag_lookup=dlookup,
        group_cols=["split", "model_name", "lcb_method", "baseline_policy_id"],
        block_length=p11.bootstrap_block_length_days, resamples=p11.bootstrap_resamples,
        annualization_days=p11.annualization_days, seed=cfg.seed,
        confidence_level=p11.confidence_level,
    )
    logger.info("Phase 11 paired block bootstrap: meta/naive stress tests")
    meta_path, meta_decision = _fold_path_inference(
        source=meta, ledger_lookup=lookup, diag_lookup=dlookup,
        group_cols=["split", "strategy_type", "model_name", "lcb_method", "baseline_policy_id"],
        block_length=p11.bootstrap_block_length_days, resamples=p11.bootstrap_resamples,
        annualization_days=p11.annualization_days, seed=cfg.seed + 1,
        confidence_level=p11.confidence_level,
    )
    paired_inf = pd.concat([
        repair_path.assign(inference_family="selective_repair"),
        meta_path.assign(inference_family="meta_stress_test"),
    ], ignore_index=True, sort=False)
    decision_inf = pd.concat([
        repair_decision.assign(inference_family="selective_repair"),
        meta_decision.assign(inference_family="meta_stress_test"),
    ], ignore_index=True, sort=False)

    annual = _annual_stability(repair)
    assets = _asset_stability(repair)
    meta_annual = _meta_annual_stability(meta)

    continuous_inf = pd.DataFrame()
    if p11.include_continuous_inference:
        logger.info("Phase 11 paired block bootstrap: continuous 2016-2025 pseudo-live")
        cl = pd.read_parquet(paths_in["continuous_ledger"])
        cp = pd.read_parquet(paths_in["continuous_performance"])
        continuous_inf = _continuous_inference(
            perf=cp, ledger_lookup=_ledger_lookup(cl),
            block_length=p11.bootstrap_block_length_days,
            resamples=p11.bootstrap_resamples,
            annualization_days=p11.annualization_days,
            seed=cfg.seed + 2, confidence_level=p11.confidence_level,
        )

    pub_repair = _merge_publication_repair(repair_path, repair_decision, annual, assets, continuous_inf)
    pub_meta = _merge_publication_meta(meta_path, meta_decision, meta_annual, continuous_inf)

    out_dir = root / cfg.paths.processed_dir / p11.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "paired": out_dir / "paired_bootstrap_inference.parquet",
        "decision": out_dir / "decision_bootstrap_inference.parquet",
        "continuous": out_dir / "continuous_bootstrap_inference.parquet",
        "annual": out_dir / "annual_stability.parquet",
        "assets": out_dir / "asset_stability.parquet",
        "meta_annual": out_dir / "meta_annual_stability.parquet",
        "pub_repair": out_dir / "publication_selective_repair.parquet",
        "pub_meta": out_dir / "publication_meta.parquet",
        "audit": out_dir / "phase11_audit.json",
    }
    for p in out_paths.values():
        if p.exists():
            p.unlink()
    paired_inf.to_parquet(out_paths["paired"], index=False)
    decision_inf.to_parquet(out_paths["decision"], index=False)
    continuous_inf.to_parquet(out_paths["continuous"], index=False)
    annual.to_parquet(out_paths["annual"], index=False)
    assets.to_parquet(out_paths["assets"], index=False)
    meta_annual.to_parquet(out_paths["meta_annual"], index=False)
    pub_repair.to_parquet(out_paths["pub_repair"], index=False)
    pub_meta.to_parquet(out_paths["pub_meta"], index=False)

    csv_files: list[Path] = []
    if p11.write_csv_summaries:
        for frame, name in [
            (pub_repair, "publication_selective_repair.csv"),
            (pub_meta, "publication_meta.csv"),
            (annual, "annual_stability.csv"),
            (continuous_inf, "continuous_bootstrap_inference.csv"),
        ]:
            p = out_dir / name
            frame.to_csv(p, index=False)
            csv_files.append(p)

    audit = {
        "phase": 11,
        "scope": "statistical inference and robustness for frozen Phase-10 outcomes",
        "primary_method": "paired circular moving-block bootstrap on synchronized baseline/adaptive daily paths",
        "bootstrap_block_length_days": int(p11.bootstrap_block_length_days),
        "bootstrap_resamples": int(p11.bootstrap_resamples),
        "confidence_level": float(p11.confidence_level),
        "annualization_days": int(p11.annualization_days),
        "primary_strategy": "policy-specific selective repair",
        "meta_status": "secondary stress test; not promoted based on Phase-10 test outcomes",
        "new_models_fit": False,
        "new_hyperparameters_tuned": False,
        "phase0_to_phase9_files_read": False,
        "phase10_outcomes_only": True,
        "paired_path_definition": "same asset, same fold/test dates, synchronized block indices for baseline and adaptive path",
        "decision_inference": "20-day block bootstrap of eligible decision dates for coverage/harm/realized-advantage uncertainty",
        "continuous_inference": bool(p11.include_continuous_inference),
        "selective_repair_groups": int(len(repair_path)),
        "meta_groups": int(len(meta_path)),
        "continuous_groups": int(len(continuous_inf)),
        "inputs": {str(p.relative_to(root)): sha256_file(p) for p in input_files},
    }
    out_paths["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase11Outputs(
        paired_bootstrap_inference=out_paths["paired"],
        decision_bootstrap_inference=out_paths["decision"],
        continuous_bootstrap_inference=out_paths["continuous"],
        annual_stability=out_paths["annual"],
        asset_stability=out_paths["assets"],
        meta_annual_stability=out_paths["meta_annual"],
        publication_selective_repair=out_paths["pub_repair"],
        publication_meta=out_paths["pub_meta"],
        audit_json=out_paths["audit"],
        csv_files=tuple(csv_files),
        input_files=input_files,
    )
