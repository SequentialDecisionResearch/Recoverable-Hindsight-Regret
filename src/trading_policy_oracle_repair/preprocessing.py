from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrozenPreprocessor:
    feature_columns: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    fit_rows: int

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing features: {sorted(missing)}")
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_columns:
            x = pd.to_numeric(frame[col], errors="coerce").astype(float)
            x = x.replace([np.inf, -np.inf], np.nan).fillna(self.medians[col])
            out[col] = (x - self.means[col]) / self.scales[col]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "fit_rows": int(self.fit_rows),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FrozenPreprocessor":
        return cls(
            feature_columns=tuple(raw["feature_columns"]),
            medians={k: float(v) for k, v in raw["medians"].items()},
            means={k: float(v) for k, v in raw["means"].items()},
            scales={k: float(v) for k, v in raw["scales"].items()},
            fit_rows=int(raw["fit_rows"]),
        )


def fit_preprocessor(frame: pd.DataFrame, feature_columns: list[str] | tuple[str, ...]) -> FrozenPreprocessor:
    cols = tuple(feature_columns)
    if frame.empty:
        raise ValueError("cannot fit preprocessor on empty frame")
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"missing features: {sorted(missing)}")
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        finite = x.dropna()
        if finite.empty:
            raise ValueError(f"feature {col} has no finite train values")
        median = float(finite.median())
        xi = x.fillna(median)
        mean = float(xi.mean())
        scale = float(xi.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        medians[col] = median
        means[col] = mean
        scales[col] = scale
    return FrozenPreprocessor(cols, medians, means, scales, int(len(frame)))
