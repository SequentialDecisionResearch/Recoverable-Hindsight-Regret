from __future__ import annotations

import numpy as np
import pandas as pd

from ..state import AccountState
from .base import Action, BasePolicy


class VolTargetPolicy(BasePolicy):
    def __init__(
        self,
        target_ann_vol: float,
        lookback: int,
        max_exposure: float = 1.0,
        annualization_days: int = 252,
        policy_id: str = "vol_target",
    ) -> None:
        if target_ann_vol <= 0:
            raise ValueError("target_ann_vol must be positive")
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0.0 < max_exposure <= 1.0:
            raise ValueError("max_exposure must lie in (0, 1]")
        if annualization_days < 1:
            raise ValueError("annualization_days must be positive")
        self.target_ann_vol = float(target_ann_vol)
        self.lookback = int(lookback)
        self.max_exposure = float(max_exposure)
        self.annualization_days = int(annualization_days)
        self.policy_id = policy_id

    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        h = self._validate_history(history, state)
        needed = self.lookback + 1
        if len(h) < needed:
            return self._warmup_action(h, state, needed_rows=needed)

        p = h["signal_price"].astype(float)
        returns = p.pct_change().dropna().iloc[-self.lookback :]
        daily_vol = float(returns.std(ddof=1))
        ann_vol = daily_vol * np.sqrt(float(self.annualization_days))
        if not np.isfinite(ann_vol) or ann_vol <= 1e-12:
            target = self.max_exposure
        else:
            target = min(self.max_exposure, self.target_ann_vol / ann_vol)
        target = float(np.clip(target, 0.0, self.max_exposure))

        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(h.iloc[-1]["date"]).normalize(),
            target_exposure=target,
            ready=True,
            reason="volatility_target",
            diagnostics={"realized_ann_vol": ann_vol, "target_ann_vol": self.target_ann_vol},
        )

    def definition(self) -> dict:
        return {
            "id": self.policy_id,
            "type": "vol_target",
            "target_ann_vol": self.target_ann_vol,
            "lookback": self.lookback,
            "max_exposure": self.max_exposure,
            "annualization_days": self.annualization_days,
        }
