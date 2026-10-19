from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccountState:
    """Single-risky-asset + cash account state at a daily close.

    Shares are expressed in the execution price basis used by the market table.
    The state itself never owns policy logic; it is mutated only by the execution
    engine through creation of a new immutable state.
    """

    date: object
    asset_id: str
    shares: float
    cash: float

    def risky_value(self, price: float) -> float:
        return float(self.shares) * float(price)

    def wealth(self, price: float) -> float:
        return self.risky_value(price) + float(self.cash)

    def exposure(self, price: float) -> float:
        w = self.wealth(price)
        if w <= 0.0:
            return 0.0
        return self.risky_value(price) / w


def make_initial_state(
    *,
    date,
    asset_id: str,
    close_price: float,
    initial_wealth: float,
    initial_exposure: float,
) -> AccountState:
    if close_price <= 0:
        raise ValueError("close_price must be positive")
    if initial_wealth <= 0:
        raise ValueError("initial_wealth must be positive")
    if not 0.0 <= initial_exposure <= 1.0:
        raise ValueError("initial_exposure must lie in [0, 1]")

    risky_value = float(initial_wealth) * float(initial_exposure)
    shares = risky_value / float(close_price)
    cash = float(initial_wealth) - risky_value
    return AccountState(date=date, asset_id=asset_id, shares=shares, cash=cash)
