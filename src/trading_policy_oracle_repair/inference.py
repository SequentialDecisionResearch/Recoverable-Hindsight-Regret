from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd


PATH_METRICS = (
    "log_utility_improvement",
    "CAGR_delta",
    "MaxDD_delta",
    "turnover_delta",
    "cost_fraction_delta",
    "Sharpe_delta",
)

DECISION_METRICS = (
    "coverage",
    "harm_rate",
    "mean_actual_advantage",
    "positive_rate",
)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(x) for x in parts)]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little") % (2**32 - 1)


def circular_block_bootstrap_indices(
    n: int,
    *,
    block_length: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block bootstrap indices, shape (resamples, n)."""
    n = int(n)
    block_length = int(block_length)
    resamples = int(resamples)
    if n <= 0:
        raise ValueError("n must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    L = min(block_length, n)
    blocks_needed = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(resamples, blocks_needed), endpoint=False)
    offsets = np.arange(L, dtype=int)
    idx = (starts[..., None] + offsets[None, None, :]) % n
    return idx.reshape(resamples, blocks_needed * L)[:, :n]


def _max_drawdown_from_log_returns(log_returns: np.ndarray) -> np.ndarray:
    x = np.asarray(log_returns, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    cum = np.cumsum(x, axis=1)
    wealth = np.exp(cum)
    wealth = np.concatenate([np.ones((len(wealth), 1)), wealth], axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    dd = wealth / peak - 1.0
    return np.min(dd, axis=1)


def _sharpe_from_log_returns(log_returns: np.ndarray, annualization_days: int) -> np.ndarray:
    x = np.asarray(log_returns, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    simple = np.expm1(x)
    mean = np.mean(simple, axis=1)
    std = np.std(simple, axis=1, ddof=1) if simple.shape[1] > 1 else np.zeros(simple.shape[0])
    out = np.zeros_like(mean)
    mask = std > 0
    out[mask] = mean[mask] / std[mask] * np.sqrt(float(annualization_days))
    return out


def ledger_daily_arrays(ledger: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {"date", "wealth_close", "trade_notional", "wealth_pre_trade", "transaction_cost"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger missing columns: {sorted(missing)}")
    x = ledger.copy().sort_values("date").reset_index(drop=True)
    if len(x) < 2:
        raise ValueError("ledger needs initial row plus at least one realized day")
    dates = pd.to_datetime(x["date"]).dt.normalize().to_numpy()[1:]
    wealth = pd.to_numeric(x["wealth_close"], errors="raise").to_numpy(dtype=float)
    if np.any(wealth <= 0):
        raise ValueError("wealth must remain positive")
    log_returns = np.diff(np.log(wealth))
    wp = pd.to_numeric(x["wealth_pre_trade"], errors="coerce").to_numpy(dtype=float)[1:]
    trade = pd.to_numeric(x["trade_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[1:]
    cost = pd.to_numeric(x["transaction_cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[1:]
    denom = np.where(np.isfinite(wp) & (wp > 0), wp, np.nan)
    turnover = np.nan_to_num(trade / denom, nan=0.0, posinf=0.0, neginf=0.0)
    cost_fraction = np.nan_to_num(cost / denom, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "dates": dates,
        "log_returns": log_returns,
        "turnover": turnover,
        "cost_fraction": cost_fraction,
    }


def paired_path_arrays(reference_ledger: pd.DataFrame, adaptive_ledger: pd.DataFrame) -> dict[str, np.ndarray]:
    ref = ledger_daily_arrays(reference_ledger)
    adp = ledger_daily_arrays(adaptive_ledger)
    if len(ref["dates"]) != len(adp["dates"]) or not np.array_equal(ref["dates"], adp["dates"]):
        raise AssertionError("paired ledgers must have identical realized dates")
    return {
        "dates": ref["dates"],
        "reference_log_returns": ref["log_returns"],
        "adaptive_log_returns": adp["log_returns"],
        "reference_turnover": ref["turnover"],
        "adaptive_turnover": adp["turnover"],
        "reference_cost_fraction": ref["cost_fraction"],
        "adaptive_cost_fraction": adp["cost_fraction"],
    }


def path_delta_metrics_from_arrays(arrays: dict[str, np.ndarray], *, annualization_days: int = 252) -> dict[str, float]:
    ref = np.asarray(arrays["reference_log_returns"], dtype=float)
    adp = np.asarray(arrays["adaptive_log_returns"], dtype=float)
    if ref.shape != adp.shape or ref.ndim != 1 or len(ref) == 0:
        raise ValueError("paired log-return arrays must be aligned nonempty vectors")
    n = len(ref)
    ref_log = float(ref.sum())
    adp_log = float(adp.sum())
    ref_cagr = float(np.exp(ref_log * float(annualization_days) / n) - 1.0)
    adp_cagr = float(np.exp(adp_log * float(annualization_days) / n) - 1.0)
    ref_dd = float(_max_drawdown_from_log_returns(ref)[0])
    adp_dd = float(_max_drawdown_from_log_returns(adp)[0])
    ref_sh = float(_sharpe_from_log_returns(ref, annualization_days)[0])
    adp_sh = float(_sharpe_from_log_returns(adp, annualization_days)[0])
    return {
        "log_utility_improvement": adp_log - ref_log,
        "CAGR_delta": adp_cagr - ref_cagr,
        "MaxDD_delta": adp_dd - ref_dd,
        "turnover_delta": float(np.sum(arrays["adaptive_turnover"]) - np.sum(arrays["reference_turnover"])),
        "cost_fraction_delta": float(np.sum(arrays["adaptive_cost_fraction"]) - np.sum(arrays["reference_cost_fraction"])),
        "Sharpe_delta": adp_sh - ref_sh,
    }


def bootstrap_paired_path(
    arrays: dict[str, np.ndarray],
    *,
    block_length: int,
    resamples: int,
    annualization_days: int,
    seed: int,
) -> dict[str, np.ndarray]:
    ref = np.asarray(arrays["reference_log_returns"], dtype=float)
    adp = np.asarray(arrays["adaptive_log_returns"], dtype=float)
    n = len(ref)
    if n == 0 or adp.shape != ref.shape:
        raise ValueError("paired path arrays must be aligned and nonempty")
    rng = np.random.default_rng(int(seed))
    idx = circular_block_bootstrap_indices(n, block_length=block_length, resamples=resamples, rng=rng)
    rb = ref[idx]
    ab = adp[idx]
    ref_log = rb.sum(axis=1)
    adp_log = ab.sum(axis=1)
    scale = float(annualization_days) / float(n)
    out = {
        "log_utility_improvement": adp_log - ref_log,
        "CAGR_delta": np.exp(adp_log * scale) - np.exp(ref_log * scale),
        "MaxDD_delta": _max_drawdown_from_log_returns(ab) - _max_drawdown_from_log_returns(rb),
        "turnover_delta": np.asarray(arrays["adaptive_turnover"], float)[idx].sum(axis=1) - np.asarray(arrays["reference_turnover"], float)[idx].sum(axis=1),
        "cost_fraction_delta": np.asarray(arrays["adaptive_cost_fraction"], float)[idx].sum(axis=1) - np.asarray(arrays["reference_cost_fraction"], float)[idx].sum(axis=1),
        "Sharpe_delta": _sharpe_from_log_returns(ab, annualization_days) - _sharpe_from_log_returns(rb, annualization_days),
    }
    return out


def decision_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {k: float("nan") for k in DECISION_METRICS}
    intervention = frame["intervention_flag"].astype(bool).to_numpy()
    adv = pd.to_numeric(frame["actual_advantage"], errors="coerce").to_numpy(dtype=float)
    coverage = float(intervention.mean())
    if not intervention.any():
        return {"coverage": coverage, "harm_rate": float("nan"), "mean_actual_advantage": float("nan"), "positive_rate": float("nan")}
    a = adv[intervention]
    return {
        "coverage": coverage,
        "harm_rate": float(np.mean(a < 0.0)),
        "mean_actual_advantage": float(np.mean(a)),
        "positive_rate": float(np.mean(a > 0.0)),
    }


def bootstrap_decision_path(
    frame: pd.DataFrame,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if frame.empty:
        raise ValueError("decision frame may not be empty")
    x = frame.copy().sort_values("decision_date").reset_index(drop=True)
    intervention = x["intervention_flag"].astype(bool).to_numpy()
    adv = pd.to_numeric(x["actual_advantage"], errors="coerce").to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(int(seed))
    idx = circular_block_bootstrap_indices(n, block_length=block_length, resamples=resamples, rng=rng)
    ib = intervention[idx]
    ab = adv[idx]
    count = ib.sum(axis=1)
    coverage = ib.mean(axis=1)
    harmful = (ib & (ab < 0.0)).sum(axis=1)
    positive = (ib & (ab > 0.0)).sum(axis=1)
    adv_sum = np.where(ib, ab, 0.0).sum(axis=1)
    harm = np.full(resamples, np.nan, dtype=float)
    mean_adv = np.full(resamples, np.nan, dtype=float)
    pos_rate = np.full(resamples, np.nan, dtype=float)
    mask = count > 0
    harm[mask] = harmful[mask] / count[mask]
    mean_adv[mask] = adv_sum[mask] / count[mask]
    pos_rate[mask] = positive[mask] / count[mask]
    return {"coverage": coverage, "harm_rate": harm, "mean_actual_advantage": mean_adv, "positive_rate": pos_rate}


def summarize_bootstrap_draws(
    observed: dict[str, float],
    draws: dict[str, np.ndarray],
    *,
    confidence_level: float,
) -> dict[str, float]:
    if not (0.5 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0.5,1)")
    alpha = (1.0 - float(confidence_level)) / 2.0
    out: dict[str, float] = {}
    for metric, arr in draws.items():
        x = np.asarray(arr, dtype=float)
        finite = x[np.isfinite(x)]
        out[f"{metric}_observed"] = float(observed.get(metric, np.nan))
        out[f"{metric}_bootstrap_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
        out[f"{metric}_ci_low"] = float(np.quantile(finite, alpha)) if len(finite) else float("nan")
        out[f"{metric}_ci_high"] = float(np.quantile(finite, 1.0 - alpha)) if len(finite) else float("nan")
        out[f"{metric}_bootstrap_se"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        out[f"{metric}_prob_gt_zero"] = float((np.sum(finite > 0.0) + 1.0) / (len(finite) + 1.0)) if len(finite) else float("nan")
        out[f"{metric}_prob_lt_zero"] = float((np.sum(finite < 0.0) + 1.0) / (len(finite) + 1.0)) if len(finite) else float("nan")
    return out


def aggregate_path_bootstrap(
    pairs: Iterable[tuple[str, dict[str, np.ndarray]]],
    *,
    block_length: int,
    resamples: int,
    annualization_days: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("at least one paired path is required")
    observed_values = {m: [] for m in PATH_METRICS}
    draw_sums = {m: np.zeros(resamples, dtype=float) for m in PATH_METRICS}
    draw_counts = {m: np.zeros(resamples, dtype=int) for m in PATH_METRICS}
    for key, arrays in pairs:
        obs = path_delta_metrics_from_arrays(arrays, annualization_days=annualization_days)
        dr = bootstrap_paired_path(
            arrays,
            block_length=block_length,
            resamples=resamples,
            annualization_days=annualization_days,
            seed=stable_seed(seed, "path", key),
        )
        for m in PATH_METRICS:
            observed_values[m].append(float(obs[m]))
            vals = np.asarray(dr[m], dtype=float)
            mask = np.isfinite(vals)
            draw_sums[m][mask] += vals[mask]
            draw_counts[m][mask] += 1
    observed = {m: float(np.mean(observed_values[m])) for m in PATH_METRICS}
    aggregate_draws = {
        m: np.divide(draw_sums[m], draw_counts[m], out=np.full(resamples, np.nan), where=draw_counts[m] > 0)
        for m in PATH_METRICS
    }
    out = summarize_bootstrap_draws(observed, aggregate_draws, confidence_level=confidence_level)
    out["paired_paths"] = int(len(pairs))
    out["bootstrap_resamples"] = int(resamples)
    out["block_length_days"] = int(block_length)
    return out


def aggregate_decision_bootstrap(
    paths: Iterable[tuple[str, pd.DataFrame]],
    *,
    block_length: int,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    paths = list(paths)
    if not paths:
        return {}
    observed_values = {m: [] for m in DECISION_METRICS}
    draw_sums = {m: np.zeros(resamples, dtype=float) for m in DECISION_METRICS}
    draw_counts = {m: np.zeros(resamples, dtype=int) for m in DECISION_METRICS}
    for key, frame in paths:
        obs = decision_metrics(frame)
        dr = bootstrap_decision_path(frame, block_length=block_length, resamples=resamples, seed=stable_seed(seed, "decision", key))
        for m in DECISION_METRICS:
            if np.isfinite(obs[m]):
                observed_values[m].append(float(obs[m]))
            vals = np.asarray(dr[m], dtype=float)
            mask = np.isfinite(vals)
            draw_sums[m][mask] += vals[mask]
            draw_counts[m][mask] += 1
    observed = {m: (float(np.mean(observed_values[m])) if observed_values[m] else float("nan")) for m in DECISION_METRICS}
    aggregate_draws = {
        m: np.divide(draw_sums[m], draw_counts[m], out=np.full(resamples, np.nan), where=draw_counts[m] > 0)
        for m in DECISION_METRICS
    }
    out = summarize_bootstrap_draws(observed, aggregate_draws, confidence_level=confidence_level)
    out["decision_paths"] = int(len(paths))
    return out
