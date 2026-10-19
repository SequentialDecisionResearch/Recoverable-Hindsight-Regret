from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def deterministic_asset_split(assets: list[str], holdout_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    clean = sorted({str(x).upper() for x in assets})
    if len(clean) < 2:
        raise ValueError("asset-OOS split requires at least two assets")
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must lie in (0,1)")
    n_hold = max(1, min(len(clean) - 1, int(np.ceil(len(clean) * holdout_fraction))))
    ranked = sorted(clean, key=lambda a: hashlib.sha256(f"{seed}|{a}".encode()).hexdigest())
    holdout = sorted(ranked[:n_hold])
    train = sorted([a for a in clean if a not in set(holdout)])
    return train, holdout


def make_annual_folds(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    initial_train_years: int,
    calibration_years: int = 1,
    test_years: int = 1,
    require_complete_test_year: bool = True,
) -> list[WalkForwardFold]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().drop_duplicates().sort_values()
    if idx.empty:
        raise ValueError("empty calendar")
    min_year = int(idx.min().year)
    max_date = pd.Timestamp(idx.max()).normalize()
    first_cal_year = min_year + int(initial_train_years)
    folds: list[WalkForwardFold] = []
    cal_year = first_cal_year
    while True:
        test_start_year = cal_year + int(calibration_years)
        test_end_year = test_start_year + int(test_years) - 1
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")
        if test_start > max_date:
            break
        if require_complete_test_year and test_end > max_date:
            break
        train_start = pd.Timestamp(f"{min_year}-01-01")
        train_end = pd.Timestamp(f"{cal_year-1}-12-31")
        cal_start = pd.Timestamp(f"{cal_year}-01-01")
        cal_end = pd.Timestamp(f"{test_start_year-1}-12-31")
        folds.append(WalkForwardFold(
            fold_id=f"test_{test_start_year}",
            train_start=train_start,
            train_end=train_end,
            calibration_start=cal_start,
            calibration_end=cal_end,
            test_start=test_start,
            test_end=test_end,
        ))
        cal_year += int(test_years)
    return folds


def embargo_cutoff(trading_calendar: pd.Series | pd.DatetimeIndex, boundary_start, embargo_days: int) -> pd.Timestamp:
    cal = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize().drop_duplicates().sort_values()
    b = pd.Timestamp(boundary_start).normalize()
    prior = cal[cal < b]
    need = int(embargo_days) + 1
    if len(prior) < need:
        raise ValueError("not enough trading days before boundary for embargo")
    return pd.Timestamp(prior[-need]).normalize()


def build_fold_membership(
    labels: pd.DataFrame,
    folds: list[WalkForwardFold],
    trading_calendar: pd.Series | pd.DatetimeIndex,
    train_assets: list[str],
    holdout_assets: list[str],
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "label_available_date", "asset_id"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    base = labels[["date", "label_available_date", "asset_id"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base["label_available_date"] = pd.to_datetime(base["label_available_date"]).dt.normalize()
    base["asset_id"] = base["asset_id"].astype(str).str.upper()
    train_set, hold_set = set(train_assets), set(holdout_assets)
    members: list[pd.DataFrame] = []
    audits: list[dict] = []

    for fold in folds:
        train_cutoff = embargo_cutoff(trading_calendar, fold.calibration_start, embargo_days)
        cal_cutoff = embargo_cutoff(trading_calendar, fold.test_start, embargo_days)

        tr = base.loc[
            base["asset_id"].isin(train_set)
            & (base["date"] >= fold.train_start) & (base["date"] <= fold.train_end)
            & (base["label_available_date"] <= train_cutoff)
        ].copy()
        tr["split"] = "train"

        ca = base.loc[
            base["asset_id"].isin(train_set)
            & (base["date"] >= fold.calibration_start) & (base["date"] <= fold.calibration_end)
            & (base["label_available_date"] <= cal_cutoff)
        ].copy()
        ca["split"] = "calibration"

        te = base.loc[
            base["asset_id"].isin(train_set)
            & (base["date"] >= fold.test_start) & (base["date"] <= fold.test_end)
            & (base["label_available_date"] <= fold.test_end)
        ].copy()
        te["split"] = "same_asset_test"

        ao = base.loc[
            base["asset_id"].isin(hold_set)
            & (base["date"] >= fold.test_start) & (base["date"] <= fold.test_end)
            & (base["label_available_date"] <= fold.test_end)
        ].copy()
        ao["split"] = "asset_oos_test"

        frames = [tr, ca, te, ao]
        for x in frames:
            x["fold_id"] = fold.fold_id
            x["train_label_cutoff"] = train_cutoff
            x["calibration_label_cutoff"] = cal_cutoff
        members.extend(frames)
        audits.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start, "train_end": fold.train_end,
            "calibration_start": fold.calibration_start, "calibration_end": fold.calibration_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "train_label_cutoff": train_cutoff,
            "calibration_label_cutoff": cal_cutoff,
            "train_dates": int(len(tr)), "calibration_dates": int(len(ca)),
            "same_asset_test_dates": int(len(te)), "asset_oos_test_dates": int(len(ao)),
        })
    membership = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    if not membership.empty:
        membership = membership.sort_values(["fold_id", "split", "asset_id", "date"]).reset_index(drop=True)
    audit = pd.DataFrame(audits)
    return membership, audit
