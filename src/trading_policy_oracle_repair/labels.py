from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def make_label_frame(
    *,
    asset_id: str,
    decision_dates: np.ndarray,
    label_available_dates: np.ndarray,
    start_exposures: list[float] | np.ndarray,
    policy_ids: list[str],
    utility_cube: np.ndarray,
    bounded_oracle_utility: np.ndarray,
    horizon: int,
    oracle_max_switches: int,
    cost_bps_per_side: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create long continuation labels and one-row-per-state summary.

    utility_cube shape: (n_dates, n_exposures, n_policies)
    bounded_oracle_utility shape: (n_dates, n_exposures)
    """
    u = np.asarray(utility_cube, dtype=float)
    bo = np.asarray(bounded_oracle_utility, dtype=float)
    e = np.asarray(start_exposures, dtype=float)
    n, ne, npol = u.shape
    if ne != len(e) or npol != len(policy_ids):
        raise ValueError("utility_cube shape does not match exposures/policies")
    if bo.shape != (n, ne):
        raise ValueError("bounded_oracle_utility shape mismatch")
    if len(decision_dates) != n or len(label_available_dates) != n:
        raise ValueError("date vector length mismatch")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(bo)):
        raise ValueError("utilities must be finite")

    best_idx = np.argmax(u, axis=2)
    best_u = np.max(u, axis=2)
    best_id = np.asarray(policy_ids, dtype=object)[best_idx]

    # Flatten in date -> exposure -> policy order.
    date3 = np.broadcast_to(np.asarray(decision_dates)[:, None, None], (n, ne, npol)).ravel()
    avail3 = np.broadcast_to(np.asarray(label_available_dates)[:, None, None], (n, ne, npol)).ravel()
    exp3 = np.broadcast_to(e[None, :, None], (n, ne, npol)).ravel()
    pid3 = np.broadcast_to(np.asarray(policy_ids, dtype=object)[None, None, :], (n, ne, npol)).ravel()
    uflat = u.ravel()
    bestu3 = np.broadcast_to(best_u[:, :, None], (n, ne, npol)).ravel()
    bestid3 = np.broadcast_to(best_id[:, :, None], (n, ne, npol)).ravel()
    bo3 = np.broadcast_to(bo[:, :, None], (n, ne, npol)).ravel()
    gap = bo3 - uflat

    labels = pd.DataFrame({
        "date": pd.to_datetime(date3),
        "label_available_date": pd.to_datetime(avail3),
        "asset_id": str(asset_id),
        "start_exposure": exp3.astype(float),
        "policy_id": pid3,
        "horizon": int(horizon),
        "utility": uflat,
        "final_wealth_from_unit": np.exp(uflat),
        "policy_class_best_policy_id": bestid3,
        "policy_class_best_utility": bestu3,
        "policy_class_regret": bestu3 - uflat,
        "bounded_oracle_value": bo3,
        "bounded_oracle_gap": gap,
        # A constrained bounded-switch comparator need not dominate every
        # continuous-exposure policy.  Keep the signed gap and separately expose
        # its nonnegative diagnostic component rather than silently hiding this.
        "bounded_oracle_regret": np.maximum(gap, 0.0),
        "oracle_max_switches": int(oracle_max_switches),
        "cost_bps_per_side": float(cost_bps_per_side),
        "label_scope": "offline_only",
    })
    if (labels["policy_class_regret"] < -1e-12).any():
        raise AssertionError("policy_class_regret must be nonnegative")
    if not (labels["label_available_date"] > labels["date"]).all():
        raise AssertionError("label_available_date must be strictly after decision date")

    state = pd.DataFrame({
        "date": pd.to_datetime(np.broadcast_to(np.asarray(decision_dates)[:, None], (n, ne)).ravel()),
        "label_available_date": pd.to_datetime(np.broadcast_to(np.asarray(label_available_dates)[:, None], (n, ne)).ravel()),
        "asset_id": str(asset_id),
        "start_exposure": np.broadcast_to(e[None, :], (n, ne)).ravel(),
        "horizon": int(horizon),
        "policy_class_best_policy_id": best_id.ravel(),
        "policy_class_best_utility": best_u.ravel(),
        "bounded_oracle_value": bo.ravel(),
        "oracle_max_switches": int(oracle_max_switches),
        "cost_bps_per_side": float(cost_bps_per_side),
    })
    return labels, state


def make_pairwise_frame(
    *,
    asset_id: str,
    decision_dates: np.ndarray,
    label_available_dates: np.ndarray,
    start_exposures: list[float] | np.ndarray,
    policy_ids: list[str],
    utility_cube: np.ndarray,
    horizon: int,
) -> pd.DataFrame:
    """Ordered policy-to-policy advantages A^{i->j}=U_j-U_i."""
    u = np.asarray(utility_cube, dtype=float)
    e = np.asarray(start_exposures, dtype=float)
    n, ne, npol = u.shape
    if npol != len(policy_ids) or ne != len(e):
        raise ValueError("utility cube shape mismatch")
    frames: list[pd.DataFrame] = []
    base_date = np.broadcast_to(np.asarray(decision_dates)[:, None], (n, ne)).ravel()
    base_avail = np.broadcast_to(np.asarray(label_available_dates)[:, None], (n, ne)).ravel()
    base_exp = np.broadcast_to(e[None, :], (n, ne)).ravel()
    for i, j in itertools.permutations(range(npol), 2):
        ui = u[:, :, i].ravel()
        uj = u[:, :, j].ravel()
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(base_date),
            "label_available_date": pd.to_datetime(base_avail),
            "asset_id": str(asset_id),
            "start_exposure": base_exp,
            "horizon": int(horizon),
            "from_policy_id": policy_ids[i],
            "to_policy_id": policy_ids[j],
            "from_utility": ui,
            "to_utility": uj,
            "pairwise_advantage": uj - ui,
        }))
    return pd.concat(frames, ignore_index=True)
