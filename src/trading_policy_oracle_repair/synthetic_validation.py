from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


REGIME_NAMES = ("trend", "meanrevert", "noise", "jump")


@dataclass(frozen=True)
class SyntheticRun:
    world: str
    signal_strength: float
    replicate: int
    train_rows: int
    calibration_rows: int
    test_rows: int
    prediction_spearman: float
    prediction_r2: float
    lcb_residual_quantile: float
    coverage: float
    abstention_rate: float
    harm_rate: float
    mean_actual_advantage: float
    positive_rate: float
    deployed_gain_per_decision: float
    oracle_positive_gain_per_decision: float
    recoverable_fraction: float


def _transition_matrix() -> np.ndarray:
    # Persistent trend/mean-reversion regimes, a persistent noise regime, and a
    # shorter-lived jump regime. Rows sum exactly to one.
    return np.asarray(
        [
            [0.970, 0.010, 0.015, 0.005],
            [0.010, 0.970, 0.015, 0.005],
            [0.020, 0.020, 0.950, 0.010],
            [0.120, 0.120, 0.160, 0.600],
        ],
        dtype=float,
    )


def simulate_regime_returns(
    *,
    world: str,
    n_days: int,
    signal_strength: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a controlled DGP for recoverability sanity checks.

    The recoverable world follows the manuscript's regime idea:

        r[t+1] = mu[z_t] + phi[z_t] r[t] + sigma[z_t] eps[t+1] + J[t+1]
        x[t]   = signal_strength * g(z_t) + eta[t]

    The clairvoyance-only world deliberately removes predictive dynamics and
    makes future jumps independent of all t-time features. Hindsight still sees
    positive opportunities, but a valid deployable model should mostly abstain.
    """
    if world not in {"recoverable", "clairvoyance_only"}:
        raise ValueError("world must be recoverable or clairvoyance_only")
    if n_days < 500:
        raise ValueError("n_days must be at least 500")
    rng = np.random.default_rng(int(seed))
    z = np.zeros(n_days, dtype=np.int8)
    z[0] = int(rng.integers(0, 4))
    p = _transition_matrix()
    for t in range(1, n_days):
        z[t] = int(rng.choice(4, p=p[int(z[t - 1])]))

    r = np.zeros(n_days + 1, dtype=float)
    jump = np.zeros(n_days, dtype=float)
    if world == "recoverable":
        # Strong enough to produce a visible but not trivial recoverability curve.
        phi = np.asarray([0.75, -0.75, 0.0, 0.0], dtype=float)
        mu = np.asarray([0.0003, 0.0003, 0.0, 0.0], dtype=float)
        sigma = np.asarray([0.0035, 0.0035, 0.0050, 0.0060], dtype=float)
        for t in range(n_days):
            jt = rng.normal(0.0, 0.020) if (int(z[t]) == 3 and rng.random() < 0.10) else 0.0
            jump[t] = jt
            r[t + 1] = mu[int(z[t])] + phi[int(z[t])] * r[t] + sigma[int(z[t])] * rng.normal() + jt
    else:
        # Main regret comes from an unpredictable future innovation/jump.
        # Regime labels are intentionally irrelevant for r[t+1].
        for t in range(n_days):
            jt = rng.normal(0.0, 0.040) if rng.random() < 0.10 else 0.0
            jump[t] = jt
            r[t + 1] = rng.normal(0.0, 0.005) + jt

    regime_code = np.asarray([1.0, -1.0, 0.0, 0.0], dtype=float)[z]
    regime_proxy = float(signal_strength) * regime_code + rng.normal(0.0, 1.0, size=n_days)
    out = pd.DataFrame(
        {
            "date": pd.bdate_range("2000-01-03", periods=n_days),
            "regime": np.asarray(REGIME_NAMES, dtype=object)[z],
            "regime_code": regime_code,
            "regime_proxy": regime_proxy,
            "return_t": r[:n_days],
            "return_t1": r[1:],
            "jump_t1": jump,
        }
    )
    out["signed_return_state"] = np.tanh(out["return_t"].to_numpy(dtype=float) / 0.01)
    out["abs_return_state"] = np.abs(out["return_t"].to_numpy(dtype=float))
    return out


def build_policy_advantage_labels(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    """Create future-only continuation advantage for two frozen synthetic rules.

    Baseline = one-day sign trend rule. Candidate = one-day sign contrarian rule.
    Each rule updates only after observing the current return state. The label is
    the H-day difference in realized net log-return-like utility, so it is future
    information and is never included in the feature columns.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    n = len(frame)
    if n <= horizon_days + 10:
        raise ValueError("synthetic frame is too short for requested horizon")
    r_t = frame["return_t"].to_numpy(dtype=float)
    r1 = frame["return_t1"].to_numpy(dtype=float)
    trend = (r_t >= 0.0).astype(float)
    contrarian = (r_t < 0.0).astype(float)
    prev_trend = np.r_[0.0, trend[:-1]]
    prev_contrarian = np.r_[0.0, contrarian[:-1]]
    cost = float(cost_bps_per_side) / 10000.0
    trend_daily = trend * r1 - cost * np.abs(trend - prev_trend)
    contra_daily = contrarian * r1 - cost * np.abs(contrarian - prev_contrarian)
    daily_adv = contra_daily - trend_daily
    csum = np.r_[0.0, np.cumsum(daily_adv)]
    advantage = np.full(n, np.nan, dtype=float)
    valid = n - int(horizon_days)
    advantage[:valid] = csum[horizon_days : horizon_days + valid] - csum[:valid]

    out = frame.copy()
    out["baseline_policy_id"] = "synthetic_trend"
    out["candidate_policy_id"] = "synthetic_contrarian"
    out["horizon_days"] = int(horizon_days)
    out["actual_advantage"] = advantage
    out["oracle_positive_advantage"] = np.maximum(advantage, 0.0)
    return out


def chronological_purged_splits(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    train_fraction: float,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.4 < train_fraction < 0.8):
        raise ValueError("train_fraction must be in (0.4, 0.8)")
    if not (0.1 <= calibration_fraction < 0.4):
        raise ValueError("calibration_fraction must be in [0.1, 0.4)")
    if train_fraction + calibration_fraction >= 0.9:
        raise ValueError("train + calibration leaves too little test data")
    eligible = np.flatnonzero(np.isfinite(frame["actual_advantage"].to_numpy(dtype=float)))
    n_eff = len(eligible)
    train_end = int(np.floor(train_fraction * n_eff))
    calibration_end = int(np.floor((train_fraction + calibration_fraction) * n_eff))
    # The label at t uses t..t+H. Purge H decision dates between train/cal/test.
    train = eligible[:train_end]
    cal_start = train_end + int(horizon_days)
    calibration = eligible[cal_start:calibration_end]
    test_start = calibration_end + int(horizon_days)
    test = eligible[test_start:]
    if min(len(train), len(calibration), len(test)) < 50:
        raise AssertionError("synthetic purged split is unexpectedly small")
    return train, calibration, test


def _safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.nanstd(y) == 0 or np.nanstd(pred) == 0:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(pred), method="spearman"))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def run_synthetic_replicate(
    *,
    world: str,
    signal_strength: float,
    replicate: int,
    n_days: int,
    horizon_days: int,
    cost_bps_per_side: float,
    train_fraction: float,
    calibration_fraction: float,
    alpha: float,
    ridge_alpha: float,
    seed: int,
) -> SyntheticRun:
    frame = simulate_regime_returns(
        world=world,
        n_days=int(n_days),
        signal_strength=float(signal_strength),
        seed=int(seed) + 100003 * int(replicate) + (0 if world == "recoverable" else 5000003),
    )
    frame = build_policy_advantage_labels(
        frame,
        horizon_days=int(horizon_days),
        cost_bps_per_side=float(cost_bps_per_side),
    )
    train, cal, test = chronological_purged_splits(
        frame,
        horizon_days=int(horizon_days),
        train_fraction=float(train_fraction),
        calibration_fraction=float(calibration_fraction),
    )
    feature_cols = ["regime_proxy", "signed_return_state", "abs_return_state"]
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["actual_advantage"].to_numpy(dtype=float)
    model = Ridge(alpha=float(ridge_alpha))
    model.fit(X[train], y[train])
    pred_cal = model.predict(X[cal])
    pred_test = model.predict(X[test])
    residual = y[cal] - pred_cal
    q = float(np.quantile(residual, float(alpha)))
    lower = pred_test + q
    deploy = lower > 0.0
    yt = y[test]

    coverage = float(np.mean(deploy))
    if np.any(deploy):
        harm_rate = float(np.mean(yt[deploy] < 0.0))
        mean_actual = float(np.mean(yt[deploy]))
        positive_rate = float(np.mean(yt[deploy] > 0.0))
    else:
        harm_rate = float("nan")
        mean_actual = float("nan")
        positive_rate = float("nan")
    student_gain = float(np.mean(np.where(deploy, yt, 0.0)))
    oracle_gain = float(np.mean(np.maximum(yt, 0.0)))
    recoverable_fraction = float(student_gain / oracle_gain) if oracle_gain > 0.0 else float("nan")

    return SyntheticRun(
        world=str(world),
        signal_strength=float(signal_strength),
        replicate=int(replicate),
        train_rows=int(len(train)),
        calibration_rows=int(len(cal)),
        test_rows=int(len(test)),
        prediction_spearman=_safe_spearman(yt, pred_test),
        prediction_r2=_r2(yt, pred_test),
        lcb_residual_quantile=q,
        coverage=coverage,
        abstention_rate=1.0 - coverage,
        harm_rate=harm_rate,
        mean_actual_advantage=mean_actual,
        positive_rate=positive_rate,
        deployed_gain_per_decision=student_gain,
        oracle_positive_gain_per_decision=oracle_gain,
        recoverable_fraction=recoverable_fraction,
    )


def run_synthetic_grid(
    *,
    worlds: Iterable[str],
    signal_strengths: Iterable[float],
    replicates: int,
    n_days: int,
    horizon_days: int,
    cost_bps_per_side: float,
    train_fraction: float,
    calibration_fraction: float,
    alpha: float,
    ridge_alpha: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for world in worlds:
        for signal in signal_strengths:
            for rep in range(int(replicates)):
                rows.append(
                    run_synthetic_replicate(
                        world=str(world), signal_strength=float(signal), replicate=rep,
                        n_days=int(n_days), horizon_days=int(horizon_days),
                        cost_bps_per_side=float(cost_bps_per_side),
                        train_fraction=float(train_fraction), calibration_fraction=float(calibration_fraction),
                        alpha=float(alpha), ridge_alpha=float(ridge_alpha), seed=int(seed),
                    ).__dict__
                )
    return pd.DataFrame(rows)


def summarize_synthetic(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    group = ["world", "signal_strength"]
    metrics = [
        "prediction_spearman", "prediction_r2", "coverage", "abstention_rate", "harm_rate",
        "mean_actual_advantage", "positive_rate", "deployed_gain_per_decision",
        "oracle_positive_gain_per_decision", "recoverable_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for key, g in runs.groupby(group, observed=True, dropna=False):
        row = dict(zip(group, key))
        row["replicates"] = int(g["replicate"].nunique())
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(x.mean()) if x.notna().any() else float("nan")
            row[f"{m}_std"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def synthetic_sanity_report(summary: pd.DataFrame) -> dict[str, Any]:
    rec = summary.loc[summary["world"] == "recoverable"].sort_values("signal_strength")
    non = summary.loc[summary["world"] == "clairvoyance_only"].sort_values("signal_strength")
    if rec.empty or non.empty:
        raise ValueError("summary must contain both synthetic worlds")
    rec_low, rec_high = rec.iloc[0], rec.iloc[-1]
    return {
        "recoverable_low_signal_gain": float(rec_low["deployed_gain_per_decision_mean"]),
        "recoverable_high_signal_gain": float(rec_high["deployed_gain_per_decision_mean"]),
        "recoverable_high_signal_fraction": float(rec_high["recoverable_fraction_mean"]),
        "recoverable_high_signal_coverage": float(rec_high["coverage_mean"]),
        "nonrecoverable_max_abs_gain": float(np.nanmax(np.abs(non["deployed_gain_per_decision_mean"].to_numpy(dtype=float)))),
        "nonrecoverable_max_coverage": float(np.nanmax(non["coverage_mean"].to_numpy(dtype=float))),
        "nonrecoverable_oracle_gain": float(np.nanmean(non["oracle_positive_gain_per_decision_mean"].to_numpy(dtype=float))),
    }


def run_phase12_synthetic_toy_validation() -> dict[str, Any]:
    runs = run_synthetic_grid(
        worlds=["recoverable", "clairvoyance_only"],
        signal_strengths=[0.0, 1.0, 4.0],
        replicates=4,
        n_days=1800,
        horizon_days=10,
        cost_bps_per_side=2.0,
        train_fraction=0.60,
        calibration_fraction=0.20,
        alpha=0.10,
        ridge_alpha=1.0,
        seed=77,
    )
    summary = summarize_synthetic(runs)
    report = synthetic_sanity_report(summary)
    if report["recoverable_high_signal_gain"] <= 0.0:
        raise AssertionError("recoverable synthetic world failed to produce positive gated gain")
    if report["recoverable_high_signal_fraction"] <= 0.03:
        raise AssertionError("recoverable synthetic world recovered too little hindsight opportunity")
    if report["nonrecoverable_oracle_gain"] <= 0.0:
        raise AssertionError("clairvoyance-only world should still contain hindsight opportunities")
    if report["nonrecoverable_max_coverage"] > 0.05:
        raise AssertionError("clairvoyance-only world should mostly abstain")
    if report["nonrecoverable_max_abs_gain"] > 0.001:
        raise AssertionError("clairvoyance-only deployable gain should be near zero")
    return {"status": "PASSED", **report}
