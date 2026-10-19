from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score


def _finite_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError("prediction/target shape mismatch")
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def _top_fraction_mask(pred: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    if len(pred) == 0:
        return np.zeros(0, dtype=bool)
    n_top = max(1, int(math.ceil(len(pred) * float(fraction))))
    order = np.argsort(pred, kind="mergesort")
    mask = np.zeros(len(pred), dtype=bool)
    mask[order[-n_top:]] = True
    return mask


def regression_metrics(y_true, y_pred, *, target_kind: str) -> dict[str, float | int]:
    y, p = _finite_pair(y_true, y_pred)
    if len(y) == 0:
        raise ValueError("no finite regression rows")
    r2 = float(r2_score(y, p)) if len(y) >= 2 else float("nan")
    rho = float(spearmanr(y, p).statistic) if len(y) >= 2 else float("nan")
    mae = float(mean_absolute_error(y, p))
    top = _top_fraction_mask(p, 0.10)
    overall = float(np.mean(y))
    top_mean = float(np.mean(y[top]))
    out: dict[str, float | int] = {
        "n": int(len(y)),
        "r2": r2,
        "spearman": rho,
        "mae": mae,
        "mean_actual": overall,
        "mean_predicted": float(np.mean(p)),
        "top_decile_mean_actual": top_mean,
        "top_decile_mean_minus_overall": float(top_mean - overall),
        "top_decile_lift_ratio": float(top_mean / overall) if abs(overall) > 1e-15 else float("nan"),
        "regret_weighted_recall_top_decile": float("nan"),
    }
    if target_kind == "regret":
        denom = float(np.sum(np.clip(y, 0.0, None)))
        numer = float(np.sum(np.clip(y[top], 0.0, None)))
        out["regret_weighted_recall_top_decile"] = numer / denom if denom > 0 else float("nan")
    return out


def classification_metrics(y_true_binary, p_high, actual_regret) -> dict[str, float | int]:
    y = np.asarray(y_true_binary, dtype=int)
    p = np.asarray(p_high, dtype=float)
    r = np.asarray(actual_regret, dtype=float)
    if not (len(y) == len(p) == len(r)) or len(y) == 0:
        raise ValueError("classification arrays must be nonempty and aligned")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("classification predictions/regret must be finite")
    classes = np.unique(y)
    auroc = float(roc_auc_score(y, p)) if len(classes) == 2 else float("nan")
    pr_auc = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    brier = float(brier_score_loss(y, p))
    deploy = p >= 0.5
    denom = float(np.sum(np.clip(r, 0.0, None)))
    rwr_p50 = float(np.sum(np.clip(r[deploy], 0.0, None)) / denom) if denom > 0 else float("nan")
    top = _top_fraction_mask(p, 0.10)
    rwr_top = float(np.sum(np.clip(r[top], 0.0, None)) / denom) if denom > 0 else float("nan")
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "brier": brier,
        "mean_predicted_probability": float(np.mean(p)),
        "positive_rate_at_0_5": float(np.mean(deploy)),
        "regret_weighted_recall_at_0_5": rwr_p50,
        "regret_weighted_recall_top_decile": rwr_top,
    }


def prediction_deciles(
    actual_target,
    predicted,
    *,
    actual_regret=None,
    bins: int = 10,
) -> pd.DataFrame:
    y, p = _finite_pair(actual_target, predicted)
    if len(y) == 0:
        return pd.DataFrame(columns=["decile", "n", "mean_predicted", "mean_actual", "mean_actual_regret"])
    if actual_regret is None:
        r = np.full(len(y), np.nan)
    else:
        rr = np.asarray(actual_regret, dtype=float)
        mask = np.isfinite(np.asarray(actual_target, dtype=float)) & np.isfinite(np.asarray(predicted, dtype=float))
        r = rr[mask]
    ranks = pd.Series(p).rank(method="first", pct=True).to_numpy()
    decile = np.minimum(int(bins), np.maximum(1, np.ceil(ranks * bins).astype(int)))
    frame = pd.DataFrame({"decile": decile, "predicted": p, "actual": y, "actual_regret": r})
    out = frame.groupby("decile", observed=True).agg(
        n=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        mean_actual=("actual", "mean"),
        mean_actual_regret=("actual_regret", "mean"),
    ).reset_index()
    return out
