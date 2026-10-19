from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .hashing import sha256_file
from .oracle import (
    brute_force_bounded_switch,
    build_binary_transition_table,
    equivalent_switch_budget,
    oracle_actions_frame,
    select_policy_class_oracle,
    solve_bounded_switch_from_transitions,
    solve_local_continuation_from_transitions,
)


@dataclass(frozen=True)
class Phase5Outputs:
    bounded_switch_oracle: Path
    bounded_switch_actions: Path
    policy_class_oracle: Path
    local_continuation_smoke: Path
    audit_json: Path
    csv_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _latest_phase4_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase4_*.json"))
    return files[-1] if files else None


def require_phase4_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase5 is not None
    cp = _latest_phase4_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase5.require_phase4_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 4 checkpoint found. Run Phase 0 through Phase 4 first.")
    return cp


def _toy_market() -> pd.DataFrame:
    # Six closes => five decision/execution intervals.  Alternating price moves
    # make the hindsight switching problem nontrivial while remaining tiny
    # enough for exhaustive enumeration.
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-02", "2026-01-05", "2026-01-06",
            "2026-01-07", "2026-01-08", "2026-01-09",
        ]),
        "asset_id": ["TOY"] * 6,
        "open": [100.0, 100.0, 120.0, 90.0, 120.0, 85.0],
        "high": [101.0, 121.0, 121.0, 121.0, 121.0, 111.0],
        "low": [99.0, 99.0, 89.0, 89.0, 84.0, 84.0],
        "close": [100.0, 120.0, 90.0, 120.0, 85.0, 110.0],
        "signal_price": [100.0, 120.0, 90.0, 120.0, 85.0, 110.0],
        "dividend_cash": [0.0] * 6,
        "split_factor": [1.0] * 6,
        "cash_daily_return": [0.0] * 6,
    })


def run_phase5_toy_validation() -> dict[str, float | int | str]:
    market = _toy_market()
    transitions = build_binary_transition_table(market, cost_bps_per_side=5.0)
    dp = solve_bounded_switch_from_transitions(
        transitions, max_switches=2, initial_position=0
    )
    brute = brute_force_bounded_switch(
        transitions, max_switches=2, initial_position=0
    )
    if abs(dp.log_utility - brute.log_utility) > 1e-12:
        raise AssertionError("Bounded-switch DP does not match brute force on toy path")
    if dp.switch_count > 2:
        raise AssertionError("Bounded-switch DP violated switch budget")

    local = solve_local_continuation_from_transitions(
        transitions,
        decision_index=0,
        horizon=3,
        initial_position=0,
        max_switches=2,
    )
    if local.action_regret(0) < -1e-15 or local.action_regret(1) < -1e-15:
        raise AssertionError("Local action regret must be nonnegative")
    if abs(local.best_log_utility - max(local.q_action_0.log_utility, local.q_action_1.log_utility)) > 1e-12:
        raise AssertionError("Local oracle first-action decomposition failed")

    perf = pd.DataFrame([
        {"asset_id": "TOY", "policy_id": "p1", "initial_wealth": 100.0, "final_wealth": 110.0},
        {"asset_id": "TOY", "policy_id": "p2", "initial_wealth": 100.0, "final_wealth": 125.0},
    ])
    policy_oracle = select_policy_class_oracle(perf)
    if str(policy_oracle.iloc[0]["oracle_policy_id"]) != "p2":
        raise AssertionError("Policy-class oracle failed toy selection")

    return {
        "status": "PASSED",
        "dp_log_utility": float(dp.log_utility),
        "bruteforce_log_utility": float(brute.log_utility),
        "dp_switch_count": int(dp.switch_count),
        "local_best_log_utility": float(local.best_log_utility),
        "local_regret_action_0": float(local.action_regret(0)),
        "local_regret_action_1": float(local.action_regret(1)),
        "policy_class_winner": str(policy_oracle.iloc[0]["oracle_policy_id"]),
    }


def _local_smoke_indices(n_intervals: int, horizon: int, points: int) -> list[int]:
    last = n_intervals - horizon
    if last < 0:
        raise ValueError("Local smoke horizon exceeds available history")
    if points == 1:
        return [last // 2]
    values = np.linspace(0, last, num=min(points, last + 1), dtype=int)
    return sorted(set(int(x) for x in values))


def build_phase5(cfg: AppConfig, project_root: str | Path) -> Phase5Outputs:
    if cfg.phase != 5 or cfg.phase5 is None:
        raise ValueError("build_phase5 requires phase=5 config")
    root = Path(project_root).resolve()
    require_phase4_checkpoint(root, cfg)
    p5 = cfg.phase5
    if abs(float(p5.primary_cost_bps) - float(cfg.research_contract.cost_bps_per_side)) > 1e-12:
        raise ValueError("phase5.primary_cost_bps must match research_contract.cost_bps_per_side")

    canonical_path = root / cfg.paths.processed_dir / p5.phase1_processed_subdir / "canonical_market.parquet"
    performance_path = root / cfg.paths.processed_dir / p5.phase4_processed_subdir / "replay_performance.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 canonical market: {canonical_path}")
    if not performance_path.exists():
        raise FileNotFoundError(f"Missing Phase 4 performance table: {performance_path}")

    market = pd.read_parquet(canonical_path)
    performance = pd.read_parquet(performance_path)
    available_assets = set(market["asset_id"].astype(str).str.upper().unique())
    assets = [x for x in cfg.raw_data.tickers if x.upper() in available_assets]
    if not assets:
        raise ValueError("No configured risky assets found in canonical market")

    oracle_rows: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    transition_cache: dict[str, Any] = {}
    k_per_year = int(cfg.research_contract.bounded_switch_k_per_year)

    for asset_id in assets:
        asset_market = (
            market.loc[market["asset_id"].astype(str).str.upper() == asset_id.upper()]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )
        transitions = build_binary_transition_table(
            asset_market, cost_bps_per_side=p5.primary_cost_bps
        )
        transition_cache[asset_id.upper()] = transitions
        k = equivalent_switch_budget(
            transitions.n_intervals,
            k_per_year=k_per_year,
            annualization_days=p5.annualization_days,
        )
        result = solve_bounded_switch_from_transitions(
            transitions,
            max_switches=k,
            initial_position=p5.initial_position,
        )
        oracle_rows.append({
            "asset_id": asset_id,
            "oracle_type": "bounded_switch_clairvoyant",
            "start_date": transitions.decision_dates[0],
            "end_date": transitions.execution_dates[-1],
            "intervals": int(transitions.n_intervals),
            "initial_position": int(p5.initial_position),
            "max_switches": int(k),
            "actual_switches": int(result.switch_count),
            "cost_bps_per_side": float(p5.primary_cost_bps),
            "initial_wealth": float(p5.initial_wealth),
            "log_utility": float(result.log_utility),
            "final_wealth": float(result.final_wealth(p5.initial_wealth)),
        })
        af = oracle_actions_frame(transitions, result, asset_id=asset_id)
        af["max_switches"] = int(k)
        af["cost_bps_per_side"] = float(p5.primary_cost_bps)
        action_frames.append(af)

    bounded_summary = pd.DataFrame(oracle_rows).sort_values("asset_id").reset_index(drop=True)
    bounded_actions = pd.concat(action_frames, ignore_index=True).sort_values(
        ["asset_id", "execution_date"]
    ).reset_index(drop=True)

    policy_class = select_policy_class_oracle(performance)

    smoke_asset = p5.local_smoke_asset.strip().upper()
    if smoke_asset not in transition_cache:
        raise ValueError(f"Local smoke asset {smoke_asset} not found among configured assets")
    transitions = transition_cache[smoke_asset]
    h = int(p5.local_horizon_days)
    local_k = equivalent_switch_budget(
        h,
        k_per_year=k_per_year,
        annualization_days=p5.annualization_days,
    )
    indices = _local_smoke_indices(transitions.n_intervals, h, p5.local_smoke_points)
    local_rows: list[dict[str, Any]] = []
    for t in indices:
        for initial_position in (0, 1):
            result = solve_local_continuation_from_transitions(
                transitions,
                decision_index=t,
                horizon=h,
                initial_position=initial_position,
                max_switches=local_k,
            )
            q0 = float(result.q_action_0.log_utility)
            q1 = float(result.q_action_1.log_utility)
            local_rows.append({
                "asset_id": smoke_asset,
                "decision_index": int(t),
                "decision_date": result.decision_date,
                "horizon": h,
                "initial_position": int(initial_position),
                "max_switches": int(local_k),
                "best_log_utility": float(result.best_log_utility),
                "optimal_first_action": int(result.best.actions[0]),
                "q_action_0": q0,
                "q_action_1": q1,
                "action_regret_0": float(result.action_regret(0)),
                "action_regret_1": float(result.action_regret(1)),
            })
    local_smoke = pd.DataFrame(local_rows).sort_values(
        ["decision_date", "initial_position"]
    ).reset_index(drop=True)

    out_dir = root / cfg.paths.processed_dir / p5.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    bounded_path = out_dir / "bounded_switch_oracle.parquet"
    actions_path = out_dir / "bounded_switch_actions.parquet"
    policy_class_path = out_dir / "policy_class_oracle.parquet"
    local_path = out_dir / "local_continuation_smoke.parquet"
    audit_path = out_dir / "phase5_audit.json"

    bounded_summary.to_parquet(bounded_path, index=False)
    bounded_actions.to_parquet(actions_path, index=False)
    policy_class.to_parquet(policy_class_path, index=False)
    local_smoke.to_parquet(local_path, index=False)

    csv_files: list[Path] = []
    if p5.write_csv_summary:
        for frame, name in [
            (bounded_summary, "bounded_switch_oracle.csv"),
            (policy_class, "policy_class_oracle.csv"),
            (local_smoke, "local_continuation_smoke.csv"),
        ]:
            path = out_dir / name
            frame.to_csv(path, index=False)
            csv_files.append(path)

    audit = {
        "phase": 5,
        "scope": "hindsight oracle infrastructure only; no Phase-6 training labels and no deployable ML",
        "assets": assets,
        "cost_bps_per_side": float(p5.primary_cost_bps),
        "bounded_switch_k_per_year": k_per_year,
        "switch_budget_rule": "round(n_intervals * k_per_year / annualization_days), min 1 when positive",
        "initial_position": int(p5.initial_position),
        "oracle_accounting": "binary transition growth generated exclusively by Phase-2 execute_next_day",
        "bounded_switch_rows": int(len(bounded_summary)),
        "bounded_action_rows": int(len(bounded_actions)),
        "policy_class_rows": int(len(policy_class)),
        "local_smoke_rows": int(len(local_smoke)),
        "local_smoke_asset": smoke_asset,
        "local_horizon_days": h,
        "local_switch_budget": int(local_k),
        "canonical_market_sha256": sha256_file(canonical_path),
        "phase4_performance_sha256": sha256_file(performance_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase5Outputs(
        bounded_switch_oracle=bounded_path,
        bounded_switch_actions=actions_path,
        policy_class_oracle=policy_class_path,
        local_continuation_smoke=local_path,
        audit_json=audit_path,
        csv_files=tuple(csv_files),
        input_files=(canonical_path, performance_path),
    )
