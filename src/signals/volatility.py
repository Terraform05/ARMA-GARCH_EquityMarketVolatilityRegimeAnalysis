from __future__ import annotations

import pandas as pd


def compute_vol_score(cond_vol: pd.Series, z_window: int) -> pd.Series:
    mean = cond_vol.rolling(z_window, min_periods=z_window).mean()
    std = cond_vol.rolling(z_window, min_periods=z_window).std()
    return (cond_vol - mean) / std


def recent_regime_change(regimes: pd.Series, window: int) -> pd.Series:
    changed = regimes != regimes.shift(1)
    return changed.rolling(window, min_periods=1).max().fillna(0).astype(bool)
