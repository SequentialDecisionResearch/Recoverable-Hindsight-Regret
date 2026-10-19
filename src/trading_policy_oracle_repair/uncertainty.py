from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairwiseCalibrator:
    alpha: float
    empirical_residual_quantile: float
    bootstrap_residual_quantile: float
    calibration_rows: int
    empirical_coverage: float
    bootstrap_coverage: float

    def lower_bound(self, predicted_advantage: np.ndarray, method: str = "empirical") -> np.ndarray:
        pred = np.asarray(predicted_advantage, dtype=float)
        if method == "empirical":
            q = self.empirical_residual_quantile
        elif method == "block_bootstrap":
            q = self.bootstrap_residual_quantile
        else:
            raise ValueError(f"unknown LCB method: {method}")
        return pred + float(q)


def empirical_lower_residual_quantile(residuals: np.ndarray, alpha: float) -> float:
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite calibration residuals")
    if not (0.0 < float(alpha) < 0.5):
        raise ValueError("alpha must lie in (0, 0.5)")
    return float(np.quantile(x, float(alpha), method="linear"))


def _date_blocks(unique_dates: np.ndarray, block_length: int) -> list[np.ndarray]:
    n = len(unique_dates)
    if n == 0:
        return []
    L = max(1, min(int(block_length), n))
    return [np.asarray([unique_dates[(i + j) % n] for j in range(L)], dtype="datetime64[ns]") for i in range(n)]


def block_bootstrap_lower_residual_quantile(
    frame: pd.DataFrame,
    *,
    residual_col: str,
    date_col: str,
    alpha: float,
    block_length_dates: int,
    n_resamples: int,
    conservative_quantile: float,
    seed: int,
    empirical_quantile: float | None = None,
) -> float:
    if int(n_resamples) <= 0:
        if empirical_quantile is None:
            empirical_quantile = empirical_lower_residual_quantile(frame[residual_col].to_numpy(), alpha)
        return float(empirical_quantile)
    df = frame[[date_col, residual_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[residual_col] = pd.to_numeric(df[residual_col], errors="coerce")
    df = df[np.isfinite(df[residual_col])]
    if df.empty:
        raise ValueError("no finite residuals for block bootstrap")
    unique_dates = np.array(sorted(df[date_col].unique()), dtype="datetime64[ns]")
    blocks = _date_blocks(unique_dates, block_length_dates)
    by_date = {np.datetime64(d): df.loc[df[date_col].to_numpy(dtype="datetime64[ns]") == np.datetime64(d), residual_col].to_numpy(dtype=float) for d in unique_dates}
    rng = np.random.default_rng(int(seed))
    qvals = np.empty(int(n_resamples), dtype=float)
    n_dates = len(unique_dates)
    L = max(1, min(int(block_length_dates), n_dates))
    n_blocks = int(np.ceil(n_dates / L))
    for b in range(int(n_resamples)):
        picked_dates: list[np.datetime64] = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, len(blocks)))
            picked_dates.extend(list(blocks[start]))
            if len(picked_dates) >= n_dates:
                break
        picked_dates = picked_dates[:n_dates]
        sample_parts = [by_date[np.datetime64(d)] for d in picked_dates]
        sample = np.concatenate(sample_parts) if sample_parts else np.array([], dtype=float)
        qvals[b] = empirical_lower_residual_quantile(sample, alpha)
    robust = float(np.quantile(qvals, float(conservative_quantile), method="linear"))
    if empirical_quantile is not None:
        robust = min(float(empirical_quantile), robust)
    return robust


def fit_pairwise_calibrator(
    calibration_frame: pd.DataFrame,
    *,
    actual_advantage_col: str = "actual_advantage",
    predicted_advantage_col: str = "predicted_advantage",
    date_col: str = "date",
    alpha: float = 0.10,
    block_length_dates: int = 20,
    n_resamples: int = 100,
    conservative_quantile: float = 0.10,
    seed: int = 0,
) -> PairwiseCalibrator:
    df = calibration_frame[[date_col, actual_advantage_col, predicted_advantage_col]].copy()
    df["residual"] = pd.to_numeric(df[actual_advantage_col], errors="coerce") - pd.to_numeric(df[predicted_advantage_col], errors="coerce")
    df = df[np.isfinite(df["residual"])]
    if df.empty:
        raise ValueError("empty pairwise calibration frame")
    q_emp = empirical_lower_residual_quantile(df["residual"].to_numpy(), alpha)
    q_boot = block_bootstrap_lower_residual_quantile(
        df, residual_col="residual", date_col=date_col, alpha=alpha,
        block_length_dates=block_length_dates, n_resamples=n_resamples,
        conservative_quantile=conservative_quantile, seed=seed,
        empirical_quantile=q_emp,
    )
    pred = pd.to_numeric(df[predicted_advantage_col], errors="raise").to_numpy(dtype=float)
    actual = pd.to_numeric(df[actual_advantage_col], errors="raise").to_numpy(dtype=float)
    cov_emp = float(np.mean(actual >= pred + q_emp))
    cov_boot = float(np.mean(actual >= pred + q_boot))
    return PairwiseCalibrator(
        alpha=float(alpha),
        empirical_residual_quantile=q_emp,
        bootstrap_residual_quantile=q_boot,
        calibration_rows=int(len(df)),
        empirical_coverage=cov_emp,
        bootstrap_coverage=cov_boot,
    )


def evidence_metrics(actual_advantage: np.ndarray, lower_bound: np.ndarray, margin: float = 0.0) -> dict[str, float | int]:
    actual = np.asarray(actual_advantage, dtype=float)
    lcb = np.asarray(lower_bound, dtype=float)
    if actual.shape != lcb.shape or actual.ndim != 1:
        raise ValueError("actual_advantage and lower_bound must be aligned 1D arrays")
    finite = np.isfinite(actual) & np.isfinite(lcb)
    actual, lcb = actual[finite], lcb[finite]
    n = len(actual)
    if n == 0:
        return {"n": 0, "deployed": 0, "coverage": float("nan"), "harm_rate": float("nan"), "mean_actual_advantage": float("nan"), "median_actual_advantage": float("nan"), "positive_rate": float("nan"), "positive_advantage_recall": float("nan")}
    deploy = lcb > float(margin)
    dep_n = int(deploy.sum())
    pos_total = float(np.maximum(actual, 0.0).sum())
    captured = float(np.maximum(actual[deploy], 0.0).sum()) if dep_n else 0.0
    return {
        "n": int(n),
        "deployed": dep_n,
        "coverage": float(dep_n / n),
        "harm_rate": float(np.mean(actual[deploy] < 0.0)) if dep_n else float("nan"),
        "mean_actual_advantage": float(np.mean(actual[deploy])) if dep_n else float("nan"),
        "median_actual_advantage": float(np.median(actual[deploy])) if dep_n else float("nan"),
        "positive_rate": float(np.mean(actual[deploy] > 0.0)) if dep_n else float("nan"),
        "positive_advantage_recall": float(captured / pos_total) if pos_total > 0 else float("nan"),
    }
