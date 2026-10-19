from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .costs import ProportionalCostModel
from .execution import execute_next_day, initial_ledger_row, prepare_execution_market
from .features import MODEL_FEATURE_COLUMNS
from .metrics import summarize_replay
from .policies.base import BasePolicy
from .preprocessing import FrozenPreprocessor
from .state import make_initial_state


@dataclass(frozen=True)
class Choice:
    selected_policy_id: str
    deploy_flag: bool
    predicted_current_utility: float
    predicted_selected_utility: float
    predicted_advantage: float
    lower_bound: float
    switch_penalty: float
    reason: str


@dataclass(frozen=True)
class FrozenFoldRuntime:
    fold_id: str
    model_name: str
    preprocessor: FrozenPreprocessor
    utility_models: Mapping[str, Any]
    residual_quantiles: Mapping[tuple[str, str, str], float]

    def predict_utilities(self, feature_row: Mapping[str, Any] | pd.Series, current_exposure: float) -> dict[str, float]:
        row = {c: feature_row[c] for c in MODEL_FEATURE_COLUMNS if c != "start_exposure"}
        row["start_exposure"] = float(current_exposure)
        X = self.preprocessor.transform(pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)).to_numpy(dtype=float)
        out: dict[str, float] = {}
        for policy_id, model in self.utility_models.items():
            pred = np.asarray(model.predict(X), dtype=float)
            if pred.shape != (1,) or not np.isfinite(pred[0]):
                raise ValueError(f"invalid utility prediction for {policy_id}")
            out[str(policy_id)] = float(pred[0])
        return out

    def lower_bound(self, baseline: str, candidate: str, predicted_advantage: float, method: str) -> float:
        key = (str(baseline), str(candidate), str(method))
        if key not in self.residual_quantiles:
            raise KeyError(f"missing calibrated residual quantile for {key}")
        return float(predicted_advantage) + float(self.residual_quantiles[key])


def selective_repair_choice(
    predicted_utilities: Mapping[str, float],
    *,
    baseline_policy_id: str,
    runtime: FrozenFoldRuntime,
    lcb_method: str,
    deploy_margin: float,
) -> Choice:
    baseline = str(baseline_policy_id)
    if baseline not in predicted_utilities:
        raise KeyError(f"baseline {baseline} missing from predictions")
    base_u = float(predicted_utilities[baseline])
    candidates = [p for p in sorted(predicted_utilities) if p != baseline]
    if not candidates:
        raise ValueError("selective repair requires at least two policies")
    scored: list[tuple[float, str, float]] = []
    for candidate in candidates:
        adv = float(predicted_utilities[candidate]) - base_u
        lcb = runtime.lower_bound(baseline, candidate, adv, lcb_method)
        scored.append((lcb, candidate, adv))
    best_lcb, best_id, best_adv = max(scored, key=lambda x: (x[0], x[1]))
    deploy = bool(best_lcb > float(deploy_margin))
    selected = best_id if deploy else baseline
    selected_u = float(predicted_utilities[selected])
    return Choice(
        selected_policy_id=selected,
        deploy_flag=deploy,
        predicted_current_utility=base_u,
        predicted_selected_utility=selected_u,
        predicted_advantage=float(best_adv if deploy else 0.0),
        lower_bound=float(best_lcb),
        switch_penalty=0.0,
        reason="lcb_repair" if deploy else "abstain_keep_baseline",
    )


def meta_selector_choice(
    predicted_utilities: Mapping[str, float],
    *,
    current_policy_id: str,
    runtime: FrozenFoldRuntime,
    lcb_method: str,
    deploy_margin: float,
    switch_penalty: float,
) -> Choice:
    current = str(current_policy_id)
    if current not in predicted_utilities:
        raise KeyError(f"current policy {current} missing from predictions")
    penalty = float(switch_penalty)
    if penalty < 0.0:
        raise ValueError("switch_penalty must be nonnegative")
    adjusted = {
        p: float(u) - (penalty if str(p) != current else 0.0)
        for p, u in predicted_utilities.items()
    }
    best = max(sorted(adjusted), key=lambda p: (adjusted[p], p))
    current_u = float(predicted_utilities[current])
    if best == current:
        return Choice(
            selected_policy_id=current, deploy_flag=False,
            predicted_current_utility=current_u,
            predicted_selected_utility=current_u,
            predicted_advantage=0.0, lower_bound=0.0,
            switch_penalty=0.0, reason="current_is_predicted_best",
        )
    raw_adv = float(predicted_utilities[best]) - current_u
    lcb = runtime.lower_bound(current, best, raw_adv, lcb_method) - penalty
    switch = bool(lcb > float(deploy_margin))
    selected = best if switch else current
    return Choice(
        selected_policy_id=selected,
        deploy_flag=switch,
        predicted_current_utility=current_u,
        predicted_selected_utility=float(predicted_utilities[selected]),
        predicted_advantage=float(raw_adv - penalty if switch else 0.0),
        lower_bound=float(lcb),
        switch_penalty=float(penalty if best != current else 0.0),
        reason="lcb_meta_switch" if switch else "abstain_keep_current",
    )


def naive_meta_choice(
    predicted_utilities: Mapping[str, float],
    *,
    current_policy_id: str,
    switch_penalty: float,
) -> Choice:
    current = str(current_policy_id)
    penalty = float(switch_penalty)
    adjusted = {
        p: float(u) - (penalty if str(p) != current else 0.0)
        for p, u in predicted_utilities.items()
    }
    best = max(sorted(adjusted), key=lambda p: (adjusted[p], p))
    raw_adv = float(predicted_utilities[best]) - float(predicted_utilities[current])
    switch = best != current
    return Choice(
        selected_policy_id=str(best), deploy_flag=bool(switch),
        predicted_current_utility=float(predicted_utilities[current]),
        predicted_selected_utility=float(predicted_utilities[best]),
        predicted_advantage=float(raw_adv - (penalty if switch else 0.0)),
        lower_bound=float("nan"),
        switch_penalty=float(penalty if switch else 0.0),
        reason="naive_predicted_argmax_switch" if switch else "naive_keep_current",
    )


@dataclass(frozen=True)
class DeploymentResult:
    ledger: pd.DataFrame
    fills: pd.DataFrame
    actions: pd.DataFrame
    performance: dict[str, Any]


def _prepare_market(asset_market: pd.DataFrame) -> pd.DataFrame:
    market = prepare_execution_market(asset_market)
    for col in ("high", "low", "signal_price"):
        if col not in market.columns:
            raise ValueError(f"deployment market missing {col}")
        x = pd.to_numeric(market[col], errors="raise").astype(float)
        if (~np.isfinite(x)).any() or (x <= 0.0).any():
            raise ValueError(f"invalid deployment {col}")
        market[col] = x
    market.attrs["_policy_history_prevalidated"] = True
    return market


def _history(market: pd.DataFrame, end_inclusive: int) -> pd.DataFrame:
    h = market.iloc[: end_inclusive + 1]
    h.attrs["_policy_history_prevalidated"] = True
    return h


def _feature_lookup(feature_table_asset: pd.DataFrame) -> pd.DataFrame:
    f = feature_table_asset.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    if f["date"].duplicated().any():
        raise ValueError("feature table contains duplicate dates for asset")
    if "feature_source_date" in f:
        src = pd.to_datetime(f["feature_source_date"]).dt.normalize()
        if (src > f["date"]).any():
            raise AssertionError("future feature_source_date detected")
    return f.set_index("date", drop=False)


def run_deployment_path(
    *,
    asset_market: pd.DataFrame,
    feature_table_asset: pd.DataFrame,
    execution_dates: list[pd.Timestamp] | pd.DatetimeIndex,
    runtime_by_execution_date: Mapping[pd.Timestamp, FrozenFoldRuntime],
    policies: Mapping[str, BasePolicy],
    strategy_type: str,
    model_name: str | None,
    lcb_method: str | None,
    baseline_policy_id: str | None,
    initial_policy_id: str,
    initial_wealth: float,
    initial_exposure: float,
    cost_bps_per_side: float,
    deploy_margin: float,
    meta_switch_penalty: float,
    annualization_days: int = 252,
) -> DeploymentResult:
    """Run a true close-t -> open-(t+1) pseudo-live deployment path.

    This function has deliberately no label/oracle argument.  All choices are
    functions only of past market history, current account state, frozen train
    models/preprocessor, and calibration-only residual quantiles.
    """
    allowed = {"fixed_policy", "selective_repair", "meta_selector", "naive_meta"}
    if strategy_type not in allowed:
        raise ValueError(f"unknown strategy_type: {strategy_type}")
    market = _prepare_market(asset_market)
    features = _feature_lookup(feature_table_asset)
    dates = pd.DatetimeIndex(market["date"]).normalize()
    pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    exec_dates = [pd.Timestamp(d).normalize() for d in execution_dates]
    if not exec_dates:
        raise ValueError("execution_dates may not be empty")
    if exec_dates != sorted(exec_dates):
        raise ValueError("execution_dates must be sorted")
    idxs = [pos[d] for d in exec_dates if d in pos]
    if len(idxs) != len(exec_dates):
        raise ValueError("some execution_dates are absent from market")
    if idxs[0] <= 0:
        raise ValueError("first execution date needs a preceding market close")
    if any(b != a + 1 for a, b in zip(idxs[:-1], idxs[1:])):
        raise ValueError("execution_dates must be consecutive market rows")

    first_decision_idx = idxs[0] - 1
    first_row = market.iloc[first_decision_idx]
    state = make_initial_state(
        date=first_row["date"], asset_id=str(first_row["asset_id"]),
        close_price=float(first_row["close"]), initial_wealth=float(initial_wealth),
        initial_exposure=float(initial_exposure),
    )
    cost_model = ProportionalCostModel.from_bps(cost_bps_per_side)
    current_policy = str(initial_policy_id)
    if current_policy not in policies:
        raise KeyError(f"unknown initial policy {current_policy}")
    if strategy_type in {"fixed_policy", "selective_repair"}:
        if baseline_policy_id is None or str(baseline_policy_id) not in policies:
            raise ValueError("fixed/selective strategy requires a valid baseline_policy_id")

    first_ledger = initial_ledger_row(state, first_row)
    first_ledger.update({
        "strategy_type": strategy_type,
        "model_name": model_name or "none",
        "lcb_method": lcb_method or "none",
        "baseline_policy_id": baseline_policy_id or "none",
        "selected_policy_id": current_policy if strategy_type != "fixed_policy" else str(baseline_policy_id),
        "fold_id": "pretest_initial",
    })
    ledger_rows: list[dict[str, Any]] = [first_ledger]
    fill_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    for execution_date, i in zip(exec_dates, idxs):
        prev_i = i - 1
        prev = market.iloc[prev_i]
        row = market.iloc[i]
        decision_date = pd.Timestamp(prev["date"]).normalize()
        runtime = runtime_by_execution_date.get(execution_date)
        if strategy_type != "fixed_policy" and runtime is None:
            raise KeyError(f"missing frozen runtime for execution date {execution_date.date()}")
        if decision_date not in features.index:
            raise KeyError(f"missing past-only feature row for {decision_date.date()}")
        history = _history(market, prev_i)
        close_price = float(prev["close"])
        current_exposure = float(state.exposure(close_price))

        if strategy_type == "fixed_policy":
            selected = str(baseline_policy_id)
            choice = Choice(
                selected_policy_id=selected, deploy_flag=False,
                predicted_current_utility=float("nan"), predicted_selected_utility=float("nan"),
                predicted_advantage=float("nan"), lower_bound=float("nan"),
                switch_penalty=0.0, reason="fixed_policy",
            )
            current_before = selected
        else:
            assert runtime is not None
            if str(runtime.model_name) != str(model_name):
                raise AssertionError("runtime model branch mismatch")
            uhat = runtime.predict_utilities(features.loc[decision_date], current_exposure)
            current_before = current_policy if strategy_type in {"meta_selector", "naive_meta"} else str(baseline_policy_id)
            if strategy_type == "selective_repair":
                choice = selective_repair_choice(
                    uhat, baseline_policy_id=str(baseline_policy_id), runtime=runtime,
                    lcb_method=str(lcb_method), deploy_margin=float(deploy_margin),
                )
            elif strategy_type == "meta_selector":
                choice = meta_selector_choice(
                    uhat, current_policy_id=current_policy, runtime=runtime,
                    lcb_method=str(lcb_method), deploy_margin=float(deploy_margin),
                    switch_penalty=float(meta_switch_penalty),
                )
                current_policy = choice.selected_policy_id
            else:
                choice = naive_meta_choice(
                    uhat, current_policy_id=current_policy,
                    switch_penalty=float(meta_switch_penalty),
                )
                current_policy = choice.selected_policy_id
            selected = choice.selected_policy_id

        action = policies[selected].act(history, state)
        target = float(action.target_exposure)
        step = execute_next_day(
            state=state, previous_row=prev, row=row,
            target_exposure=target, cost_model=cost_model,
        )
        state = step.state
        fold_id = runtime.fold_id if runtime is not None else "fixed"
        action_rows.append({
            "fold_id": fold_id,
            "strategy_type": strategy_type,
            "model_name": model_name or "none",
            "lcb_method": lcb_method or "none",
            "baseline_policy_id": baseline_policy_id or "none",
            "asset_id": str(prev["asset_id"]),
            "decision_date": decision_date,
            "execution_date": execution_date,
            "current_policy_before": current_before,
            "selected_policy_id": selected,
            "current_policy_after": current_policy if strategy_type in {"meta_selector", "naive_meta"} else current_before,
            "deploy_flag": bool(choice.deploy_flag),
            "current_exposure": current_exposure,
            "target_exposure": target,
            "policy_ready": bool(action.ready),
            "policy_reason": str(action.reason),
            "decision_reason": choice.reason,
            "predicted_current_utility": choice.predicted_current_utility,
            "predicted_selected_utility": choice.predicted_selected_utility,
            "predicted_advantage": choice.predicted_advantage,
            "lower_bound": choice.lower_bound,
            "selector_switch_penalty": choice.switch_penalty,
        })
        lr = dict(step.ledger_row)
        lr.update({
            "fold_id": fold_id,
            "strategy_type": strategy_type,
            "model_name": model_name or "none",
            "lcb_method": lcb_method or "none",
            "baseline_policy_id": baseline_policy_id or "none",
            "selected_policy_id": selected,
        })
        ledger_rows.append(lr)
        if step.fill_row is not None:
            fr = dict(step.fill_row)
            fr.update({
                "fold_id": fold_id,
                "strategy_type": strategy_type,
                "model_name": model_name or "none",
                "lcb_method": lcb_method or "none",
                "baseline_policy_id": baseline_policy_id or "none",
                "selected_policy_id": selected,
            })
            fill_rows.append(fr)

    ledger = pd.DataFrame(ledger_rows)
    actions = pd.DataFrame(action_rows)
    fills = pd.DataFrame(fill_rows)
    if fills.empty:
        fills = pd.DataFrame(columns=[
            "asset_id", "decision_date", "execution_date", "execution_price", "target_exposure",
            "shares_before", "shares_after", "trade_shares", "trade_notional", "transaction_cost",
            "wealth_pre_trade", "wealth_post_cost", "fold_id", "strategy_type", "model_name",
            "lcb_method", "baseline_policy_id", "selected_policy_id",
        ])
    # summarize_replay expects a policy_id-like field on first row and readiness columns in actions.
    metric_ledger = ledger.copy()
    metric_ledger["policy_id"] = strategy_type
    metric_actions = actions.rename(columns={"policy_ready": "ready"}).copy()
    performance = summarize_replay(metric_ledger, fills, metric_actions, annualization_days=annualization_days)
    performance.update({
        "strategy_type": strategy_type,
        "model_name": model_name or "none",
        "lcb_method": lcb_method or "none",
        "baseline_policy_id": baseline_policy_id or "none",
        "deploy_count": int(actions["deploy_flag"].sum()) if not actions.empty else 0,
        "decision_count": int(len(actions)),
        "deployment_coverage": float(actions["deploy_flag"].mean()) if len(actions) else 0.0,
        "policy_identity_switches": int((actions["current_policy_after"].astype(str) != actions["current_policy_before"].astype(str)).sum()) if not actions.empty else 0,
    })
    return DeploymentResult(ledger=ledger, fills=fills, actions=actions, performance=performance)


def build_policy_action_template_from_policy(asset_market: pd.DataFrame, policy: BasePolicy):
    """Build the Phase-6 style fixed/hold template without using hindsight labels."""
    from .continuation import build_action_template

    market = _prepare_market(asset_market)
    rows: list[dict[str, Any]] = []
    for i in range(len(market)):
        h = _history(market, i)
        row = market.iloc[i]
        # 0.5 only probes whether the policy emits a fixed target or hold-current.
        # For the six registered policies that distinction depends on market history,
        # while the held exposure value itself is supplied later by the evaluator.
        state = make_initial_state(
            date=row["date"], asset_id=str(row["asset_id"]), close_price=float(row["close"]),
            initial_wealth=1.0, initial_exposure=0.5,
        )
        action = policy.act(h, state)
        rows.append({"date": row["date"], "target_exposure": float(action.target_exposure), "reason": str(action.reason)})
    return build_action_template(market_dates=market["date"], replay_actions=pd.DataFrame(rows))


def continuation_utilities_for_states(
    *,
    asset_market: pd.DataFrame,
    action_template,
    decision_indices: np.ndarray,
    start_exposures: np.ndarray,
    horizon: int,
    cost_bps_per_side: float,
) -> np.ndarray:
    """Evaluation-only H-day utilities for arbitrary realized deployment states."""
    from .continuation import vectorized_account_step

    market = prepare_execution_market(asset_market)
    idx0 = np.asarray(decision_indices, dtype=int)
    e0 = np.asarray(start_exposures, dtype=float)
    if idx0.ndim != 1 or e0.ndim != 1 or len(idx0) != len(e0):
        raise ValueError("decision_indices/start_exposures must be aligned vectors")
    if len(idx0) == 0:
        return np.array([], dtype=float)
    if np.any(idx0 < 0) or np.any(idx0 + int(horizon) >= len(market)):
        raise ValueError("evaluation continuation extends beyond market")
    if np.any((e0 < 0.0) | (e0 > 1.0)):
        raise ValueError("start exposures must lie in [0,1]")

    wealth = np.ones(len(idx0), dtype=float)
    exposure = e0.copy()
    close = market["close"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    cash_r = market["cash_daily_return"].to_numpy(dtype=float)
    split = market["split_factor"].to_numpy(dtype=float)
    div = market["dividend_cash"].to_numpy(dtype=float)
    for h in range(int(horizon)):
        idx = idx0 + h
        nxt = idx + 1
        hold = action_template.hold_current[idx]
        fixed = action_template.fixed_target[idx]
        target = np.where(hold, exposure, fixed)
        wealth, exposure, _ = vectorized_account_step(
            wealth=wealth, exposure=exposure,
            previous_close=close[idx], next_open=open_[nxt], next_close=close[nxt],
            cash_daily_return=cash_r[nxt], split_factor=split[nxt], dividend_cash=div[nxt],
            target_exposure=target, cost_bps_per_side=float(cost_bps_per_side),
        )
    return np.log(wealth)
