from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BasePolicy
from .breakout import DonchianBreakoutPolicy
from .buy_hold import BuyHoldPolicy
from .mean_reversion import RSIMeanReversionPolicy
from .momentum import MomentumPolicy
from .sma_cross import SMACrossPolicy
from .vol_target import VolTargetPolicy


def load_policy_specs(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    specs = raw.get("policies") if isinstance(raw, dict) else None
    if not isinstance(specs, list) or not specs:
        raise ValueError("Policy YAML must contain a nonempty 'policies' list")
    ids = [str(x.get("id", "")).strip() for x in specs]
    if any(not x for x in ids):
        raise ValueError("Every policy requires a nonempty id")
    if len(ids) != len(set(ids)):
        raise ValueError("Policy ids must be unique")
    return specs


def build_policy(spec: dict[str, Any]) -> BasePolicy:
    policy_id = str(spec["id"])
    ptype = str(spec["type"])
    if ptype == "buy_hold":
        return BuyHoldPolicy(policy_id=policy_id)
    if ptype == "sma_cross":
        return SMACrossPolicy(
            short_window=int(spec["short_window"]),
            long_window=int(spec["long_window"]),
            policy_id=policy_id,
        )
    if ptype == "momentum":
        return MomentumPolicy(lookback=int(spec["lookback"]), policy_id=policy_id)
    if ptype == "breakout":
        return DonchianBreakoutPolicy(
            entry_window=int(spec["entry_window"]),
            exit_window=int(spec["exit_window"]),
            policy_id=policy_id,
        )
    if ptype == "rsi_mean_reversion":
        return RSIMeanReversionPolicy(
            window=int(spec["window"]),
            entry=float(spec["entry"]),
            exit=float(spec["exit"]),
            policy_id=policy_id,
        )
    if ptype == "vol_target":
        return VolTargetPolicy(
            target_ann_vol=float(spec["target_ann_vol"]),
            lookback=int(spec["lookback"]),
            max_exposure=float(spec.get("max_exposure", 1.0)),
            annualization_days=int(spec.get("annualization_days", 252)),
            policy_id=policy_id,
        )
    raise ValueError(f"Unsupported policy type: {ptype}")


def load_policies(path: str | Path) -> dict[str, BasePolicy]:
    policies = [build_policy(spec) for spec in load_policy_specs(path)]
    return {p.policy_id: p for p in policies}
