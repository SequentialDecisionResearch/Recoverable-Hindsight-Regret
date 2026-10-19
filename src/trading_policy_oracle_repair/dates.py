from __future__ import annotations

import pandas as pd


def normalize_daily_dates(values) -> pd.DatetimeIndex:
    """Return timezone-naive midnight calendar dates for daily market bars.

    Daily bars are keyed by exchange calendar date, not by an intraday UTC clock.
    This also cleans legacy files where a New York midnight daily bar was stored
    as a naive 04:00/05:00 timestamp after a UTC conversion.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if idx.tz is not None:
        # Preserve the vendor's displayed calendar date; do not convert timezone.
        idx = idx.tz_localize(None)
    return idx.normalize()
