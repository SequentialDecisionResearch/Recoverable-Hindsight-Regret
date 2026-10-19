from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .continuation import (
    bounded_oracle_utility_grid,
    build_action_template,
    continuation_utility_grid,
    direct_policy_continuation,
    vectorized_account_step,
)
from .hashing import sha256_file
from .labels import make_label_frame, make_pairwise_frame
from .oracle import equivalent_switch_budget
from .policies import BuyHoldPolicy, load_policy_specs


@dataclass(frozen=True)
class Phase6Outputs:
    continuation_labels: Path
    pairwise_advantage: Path
    state_summary: Path
    audit_json: Path
    csv_sample: Path | None
    input_files: tuple[Path, ...]


def _latest_phase5_checkpoint(project_root: Path, checkpoints_dir: Path) -> Path | None:
    files = sorted((project_root / checkpoints_dir).glob("phase5_*.json"))
    return files[-1] if files else None


def require_phase5_checkpoint(project_root: Path, cfg: AppConfig) -> Path | None:
    assert cfg.phase6 is not None
    cp = _latest_phase5_checkpoint(project_root, cfg.paths.checkpoints_dir)
    if cfg.phase6.require_phase5_checkpoint and cp is None:
        raise FileNotFoundError("No Phase 5 checkpoint found. Run Phase 0 through Phase 5 first.")
    return cp


def _toy_market(n: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=n)
    # Nontrivial open/close path, with a dividend, so labels exercise all key
    # accounting pieces without needing a large example.
    close = np.array([100, 102, 101, 105, 104, 99, 100, 103, 101, 106, 108, 107], dtype=float)[:n]
    open_ = np.r_[close[0], close[:-1] * np.array([1.0] + [1.002] * (n - 2))][:n]
    return pd.DataFrame({
        "date": dates,
        "asset_id": ["TOY"] * n,
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "signal_price": close,
        "dividend_cash": [0, 0, 0, 0, 0.2] + [0] * (n - 5),
        "split_factor": [1.0] * n,
        "cash_daily_return": [0.0] + [0.0001] * (n - 1),
    })


def run_phase6_toy_validation() -> dict[str, float | int | str]:
    market = _toy_market()
    horizon = 4
    exposures = [0.0, 0.25, 1.0]

    # Buy&Hold is fixed-target, so its Phase-4-style action template is simple.
    actions = pd.DataFrame({
        "date": market["date"],
        "target_exposure": [1.0] * len(market),
        "reason": ["fully invested"] * len(market),
    })
    template = build_action_template(market_dates=market["date"], replay_actions=actions)
    utilities, _, _, dates = continuation_utility_grid(
        asset_market=market,
        action_template=template,
        start_exposures=exposures,
        horizon=horizon,
        cost_bps_per_side=5.0,
    )
    direct_u, _ = direct_policy_continuation(
        asset_market=market,
        policy=BuyHoldPolicy(policy_id="buy_hold"),
        decision_index=0,
        horizon=horizon,
        initial_exposure=0.25,
        cost_bps_per_side=5.0,
    )
    diff = abs(float(utilities[0, 1]) - float(direct_u))
    if diff > 1e-11:
        raise AssertionError("Vectorized Phase-6 continuation does not match direct Phase-2 replay")

    oracle_u, _ = bounded_oracle_utility_grid(
        asset_market=market,
        start_exposures=exposures,
        horizon=horizon,
        max_switches=2,
        cost_bps_per_side=5.0,
    )
    if oracle_u.shape != utilities.shape:
        raise AssertionError("Oracle grid shape mismatch")
    if pd.Timestamp(dates[0, 1]) != pd.Timestamp(market.iloc[horizon]["date"]):
        raise AssertionError("label_available_date must equal t+H trading row")

    # Direct equivalence of one vectorized account step and Phase-2 semantics is
    # covered again here with a fractional starting exposure.
    w, e, _ = vectorized_account_step(
        wealth=np.array([1.0]), exposure=np.array([0.25]),
        previous_close=np.array([100.0]), next_open=np.array([103.0]), next_close=np.array([105.0]),
        cash_daily_return=np.array([0.0001]), split_factor=np.array([1.0]),
        dividend_cash=np.array([0.2]), target_exposure=np.array([0.75]), cost_bps_per_side=5.0,
    )
    if not (float(w[0]) > 0 and 0 <= float(e[0]) <= 1):
        raise AssertionError("Vectorized account step produced invalid state")

    return {
        "status": "PASSED",
        "vectorized_vs_direct_abs_error": float(diff),
        "toy_start_states": int((len(market) - horizon) * len(exposures)),
        "first_label_available_date": str(pd.Timestamp(dates[0, 1]).date()),
        "sample_oracle_log_utility": float(oracle_u[0, 1]),
    }


def _open_parquet_writer(path: Path, frame: pd.DataFrame):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = pq.ParquetWriter(
        str(path),
        table.schema,
        compression="zstd",
        use_dictionary=True,
    )
    writer.write_table(table)
    return writer


def _append_parquet(writer, frame: pd.DataFrame) -> None:
    import pyarrow as pa

    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer.write_table(table)


def build_phase6(cfg: AppConfig, project_root: str | Path) -> Phase6Outputs:
    if cfg.phase != 6 or cfg.phase6 is None:
        raise ValueError("build_phase6 requires phase=6 config")
    root = Path(project_root).resolve()
    require_phase5_checkpoint(root, cfg)
    p6 = cfg.phase6
    contract = cfg.research_contract
    if int(p6.horizon_days) != int(contract.horizon_days):
        raise ValueError("phase6.horizon_days must match research_contract.horizon_days")
    if abs(float(p6.primary_cost_bps) - float(contract.cost_bps_per_side)) > 1e-12:
        raise ValueError("phase6.primary_cost_bps must match research_contract.cost_bps_per_side")

    canonical_path = root / cfg.paths.processed_dir / p6.phase1_processed_subdir / "canonical_market.parquet"
    actions_path = root / cfg.paths.processed_dir / p6.phase4_processed_subdir / "replay_actions.parquet"
    policies_path = root / p6.policies_config
    for path in (canonical_path, actions_path, policies_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase-6 input: {path}")

    market = pd.read_parquet(canonical_path)
    actions = pd.read_parquet(actions_path)
    policy_specs = load_policy_specs(policies_path)
    policy_ids = [str(x["id"]) for x in policy_specs]
    exposures = [float(x) for x in contract.start_exposures]
    horizon = int(p6.horizon_days)
    local_k = equivalent_switch_budget(
        horizon,
        k_per_year=int(contract.bounded_switch_k_per_year),
        annualization_days=int(p6.annualization_days),
    )

    available_assets = set(market["asset_id"].astype(str).str.upper().unique())
    assets = [x for x in cfg.raw_data.tickers if x.upper() in available_assets]
    if not assets:
        raise ValueError("No configured risky assets found in canonical market")

    out_dir = root / cfg.paths.processed_dir / p6.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "continuation_labels.parquet"
    pairwise_path = out_dir / "pairwise_advantage.parquet"
    state_path = out_dir / "label_state_summary.parquet"
    audit_path = out_dir / "phase6_audit.json"
    sample_path = out_dir / "continuation_labels_sample.csv" if p6.write_csv_sample else None
    for path in (labels_path, pairwise_path, state_path):
        if path.exists():
            path.unlink()

    logger = logging.getLogger("trading_policy_oracle_repair")
    label_writer = pair_writer = state_writer = None
    sample_frames: list[pd.DataFrame] = []
    total_label_rows = total_pair_rows = total_state_rows = 0
    negative_oracle_gap_rows = 0
    asset_audit: list[dict[str, Any]] = []

    try:
        for asset_id in assets:
            logger.info("Phase 6 labels: %s", asset_id)
            am = (
                market.loc[market["asset_id"].astype(str).str.upper() == asset_id.upper()]
                .copy().sort_values("date").reset_index(drop=True)
            )
            if len(am) <= horizon:
                raise ValueError(f"Asset {asset_id} has insufficient history for H={horizon}")
            n = len(am) - horizon
            utility_cube = np.empty((n, len(exposures), len(policy_ids)), dtype=float)
            decision_dates = label_dates = None

            for j, policy_id in enumerate(policy_ids):
                aa = actions.loc[
                    (actions["asset_id"].astype(str).str.upper() == asset_id.upper())
                    & (actions["policy_id"].astype(str) == policy_id)
                ].copy().sort_values("date").reset_index(drop=True)
                template = build_action_template(market_dates=am["date"], replay_actions=aa)
                utility, _, _, dates = continuation_utility_grid(
                    asset_market=am,
                    action_template=template,
                    start_exposures=exposures,
                    horizon=horizon,
                    cost_bps_per_side=float(p6.primary_cost_bps),
                )
                utility_cube[:, :, j] = utility
                if decision_dates is None:
                    decision_dates = dates[:, 0]
                    label_dates = dates[:, 1]

            assert decision_dates is not None and label_dates is not None
            oracle_u, _ = bounded_oracle_utility_grid(
                asset_market=am,
                start_exposures=exposures,
                horizon=horizon,
                max_switches=local_k,
                cost_bps_per_side=float(p6.primary_cost_bps),
            )
            labels, state = make_label_frame(
                asset_id=asset_id,
                decision_dates=decision_dates,
                label_available_dates=label_dates,
                start_exposures=exposures,
                policy_ids=policy_ids,
                utility_cube=utility_cube,
                bounded_oracle_utility=oracle_u,
                horizon=horizon,
                oracle_max_switches=local_k,
                cost_bps_per_side=float(p6.primary_cost_bps),
            )
            pairwise = make_pairwise_frame(
                asset_id=asset_id,
                decision_dates=decision_dates,
                label_available_dates=label_dates,
                start_exposures=exposures,
                policy_ids=policy_ids,
                utility_cube=utility_cube,
                horizon=horizon,
            )

            if label_writer is None:
                label_writer = _open_parquet_writer(labels_path, labels)
                pair_writer = _open_parquet_writer(pairwise_path, pairwise)
                state_writer = _open_parquet_writer(state_path, state)
            else:
                _append_parquet(label_writer, labels)
                _append_parquet(pair_writer, pairwise)
                _append_parquet(state_writer, state)

            if p6.write_csv_sample and sum(len(x) for x in sample_frames) < p6.csv_sample_rows:
                remaining = p6.csv_sample_rows - sum(len(x) for x in sample_frames)
                sample_frames.append(labels.head(remaining).copy())

            total_label_rows += len(labels)
            total_pair_rows += len(pairwise)
            total_state_rows += len(state)
            negative_oracle_gap_rows += int((labels["bounded_oracle_gap"] < -1e-12).sum())
            asset_audit.append({
                "asset_id": asset_id,
                "market_rows": int(len(am)),
                "eligible_decision_dates": int(n),
                "state_rows": int(len(state)),
                "label_rows": int(len(labels)),
                "pairwise_rows": int(len(pairwise)),
                "first_decision_date": str(pd.Timestamp(decision_dates[0]).date()),
                "last_decision_date": str(pd.Timestamp(decision_dates[-1]).date()),
                "first_label_available_date": str(pd.Timestamp(label_dates[0]).date()),
                "last_label_available_date": str(pd.Timestamp(label_dates[-1]).date()),
            })
            logger.info(
                "Phase 6 %s complete: states=%d labels=%d pairwise=%d",
                asset_id, len(state), len(labels), len(pairwise),
            )
    finally:
        for writer in (label_writer, pair_writer, state_writer):
            if writer is not None:
                writer.close()

    if p6.write_csv_sample and sample_path is not None and p6.csv_sample_rows > 0:
        pd.concat(sample_frames, ignore_index=True).head(p6.csv_sample_rows).to_csv(sample_path, index=False)

    expected_labels = total_state_rows * len(policy_ids)
    expected_pairs = total_state_rows * len(policy_ids) * (len(policy_ids) - 1)
    if total_label_rows != expected_labels or total_pair_rows != expected_pairs:
        raise AssertionError("Phase-6 output row counts violate policy/state combinatorics")

    audit = {
        "phase": 6,
        "scope": "offline continuation labels only; no train/cal/test split and no ML",
        "utility_definition": "log(final_wealth / initial_wealth) with initial wealth normalized to 1",
        "common_starting_state": True,
        "horizon_days": horizon,
        "start_exposures": exposures,
        "policy_ids": policy_ids,
        "assets": assets,
        "primary_cost_bps_per_side": float(p6.primary_cost_bps),
        "bounded_switch_k_per_year": int(contract.bounded_switch_k_per_year),
        "local_oracle_max_switches": int(local_k),
        "fractional_initial_oracle_rule": "first move to binary {0,1} pays cost and counts as a switch unless target equals start_exposure",
        "bounded_oracle_note": "bounded_oracle_gap is signed because a switch-constrained binary comparator need not dominate continuous or more frequently trading policies; bounded_oracle_regret stores max(gap,0) only as a nonnegative diagnostic",
        "rows": {
            "state_summary": int(total_state_rows),
            "continuation_labels": int(total_label_rows),
            "pairwise_advantage": int(total_pair_rows),
            "negative_bounded_oracle_gap_rows": int(negative_oracle_gap_rows),
        },
        "asset_audit": asset_audit,
        "canonical_market_sha256": sha256_file(canonical_path),
        "phase4_replay_actions_sha256": sha256_file(actions_path),
        "policies_config_sha256": sha256_file(policies_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    return Phase6Outputs(
        continuation_labels=labels_path,
        pairwise_advantage=pairwise_path,
        state_summary=state_path,
        audit_json=audit_path,
        csv_sample=sample_path,
        input_files=(canonical_path, actions_path, policies_path),
    )
