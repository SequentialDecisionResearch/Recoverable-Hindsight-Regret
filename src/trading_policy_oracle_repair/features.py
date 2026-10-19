from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "mom_5", "mom_20", "mom_60", "mom_126",
    "vol_10", "vol_20", "vol_60",
    "drawdown_20", "drawdown_60", "drawdown_252",
    "ma_20_100", "ma_50_200", "rsi_14",
]
MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ["start_exposure"]


def _rsi_series(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return rsi


def build_market_features(canonical_market: pd.DataFrame, annualization_days: int = 252) -> pd.DataFrame:
    """Build past-only market features from signal_price.

    Each feature at date t is a deterministic function of rows with date <= t.
    The output keeps feature_source_date so later phases can audit the time wall.
    """
    required = {"date", "asset_id", "signal_price"}
    missing = required - set(canonical_market.columns)
    if missing:
        raise ValueError(f"canonical market missing feature columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    for asset_id, g0 in canonical_market.groupby("asset_id", sort=False):
        g = g0.copy().sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"]).dt.normalize()
        p = pd.to_numeric(g["signal_price"], errors="raise").astype(float)
        if (p <= 0).any() or (~np.isfinite(p)).any():
            raise ValueError(f"invalid signal_price for {asset_id}")
        logp = np.log(p)
        r = logp.diff()
        out = pd.DataFrame({"date": g["date"], "asset_id": str(asset_id)})
        for k in (5, 20, 60, 126):
            out[f"mom_{k}"] = logp - logp.shift(k)
        for k in (10, 20, 60):
            out[f"vol_{k}"] = r.rolling(k, min_periods=k).std(ddof=1) * np.sqrt(float(annualization_days))
        for k in (20, 60, 252):
            out[f"drawdown_{k}"] = p / p.rolling(k, min_periods=k).max() - 1.0
        ma20 = p.rolling(20, min_periods=20).mean()
        ma50 = p.rolling(50, min_periods=50).mean()
        ma100 = p.rolling(100, min_periods=100).mean()
        ma200 = p.rolling(200, min_periods=200).mean()
        out["ma_20_100"] = ma20 / ma100 - 1.0
        out["ma_50_200"] = ma50 / ma200 - 1.0
        out["rsi_14"] = _rsi_series(p, 14)
        out["feature_source_date"] = out["date"]
        parts.append(out)

    features = pd.concat(parts, ignore_index=True).sort_values(["asset_id", "date"]).reset_index(drop=True)
    return features


def assert_feature_time_wall(features: pd.DataFrame) -> None:
    if not {"date", "feature_source_date"}.issubset(features.columns):
        raise ValueError("features require date and feature_source_date")
    d = pd.to_datetime(features["date"]).dt.normalize()
    s = pd.to_datetime(features["feature_source_date"]).dt.normalize()
    if (s > d).any():
        raise AssertionError("feature source timestamp exceeds decision date")
