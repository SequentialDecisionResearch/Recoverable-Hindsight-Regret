from __future__ import annotations

import pandas as pd

from ..state import AccountState
from .base import Action, BasePolicy


class BuyHoldPolicy(BasePolicy):
    def __init__(self, policy_id: str = "buy_hold") -> None:
        self.policy_id = policy_id

    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        h = self._validate_history(history, state)
        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(h.iloc[-1]["date"]).normalize(),
            target_exposure=1.0,
            ready=True,
            reason="fully invested",
            diagnostics={},
        )

    def definition(self) -> dict:
        return {"id": self.policy_id, "type": "buy_hold"}
