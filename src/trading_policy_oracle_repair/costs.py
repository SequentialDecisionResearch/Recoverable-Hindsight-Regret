from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionalCostModel:
    """Single-sided proportional transaction-cost model."""

    cost_rate: float

    @classmethod
    def from_bps(cls, bps_per_side: float) -> "ProportionalCostModel":
        if bps_per_side < 0:
            raise ValueError("bps_per_side must be non-negative")
        return cls(cost_rate=float(bps_per_side) / 10_000.0)

    def cost(self, trade_notional: float) -> float:
        if trade_notional < -1e-12:
            raise ValueError("trade_notional must be non-negative")
        return self.cost_rate * max(float(trade_notional), 0.0)


def solve_post_cost_wealth(
    *,
    v_pre: float,
    current_risky_value: float,
    target_exposure: float,
    cost_rate: float,
    iterations: int = 100,
) -> float:
    """Solve the post-cost target-weight fixed point by bisection.

    For one risky asset plus cash, solve

        V+ + c * |w V+ - v_risky^-| = V^-

    so the target risky exposure is exact after costs are deducted.
    """
    v_pre = float(v_pre)
    current_risky_value = float(current_risky_value)
    target_exposure = float(target_exposure)
    cost_rate = float(cost_rate)

    if v_pre < 0:
        raise ValueError("v_pre must be non-negative")
    if current_risky_value < -1e-12:
        raise ValueError("current_risky_value must be non-negative")
    if not 0.0 <= target_exposure <= 1.0:
        raise ValueError("target_exposure must lie in [0, 1]")
    if not 0.0 <= cost_rate < 1.0:
        raise ValueError("cost_rate must lie in [0, 1)")
    if v_pre == 0.0 or cost_rate == 0.0:
        return v_pre

    def f(v_net: float) -> float:
        turnover = abs(target_exposure * v_net - current_risky_value)
        return v_net + cost_rate * turnover - v_pre

    # If no rebalancing is required, V+ = V^- exactly.
    if abs(f(v_pre)) <= 1e-14 * max(v_pre, 1.0):
        return v_pre

    lo, hi = 0.0, v_pre
    if f(lo) > 1e-12:
        raise ValueError("No feasible post-cost wealth under supplied inputs")

    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
