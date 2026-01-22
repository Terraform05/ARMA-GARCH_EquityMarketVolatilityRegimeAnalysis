from __future__ import annotations

import numpy as np
import pandas as pd


def matrix_exposure(
    regimes: pd.Series,
    trend_states: pd.Series,
    matrix: dict[str, dict[str, float]],
) -> pd.Series:
    lookup = {
        (regime, state): exposure
        for regime, row in matrix.items()
        for state, exposure in row.items()
    }
    exposures = [
        lookup.get((regime, state), 0.0)
        for regime, state in zip(regimes, trend_states)
    ]
    return pd.Series(exposures, index=regimes.index)


def continuous_exposure(
    *,
    trend_score: pd.Series,
    vol_score: pd.Series,
    base_exposure: float,
    trend_coef: float,
    vol_coef: float,
) -> pd.Series:
    return base_exposure + trend_coef * trend_score - vol_coef * vol_score


def apply_rebalance(
    frame: pd.DataFrame, exposure_col: str, rebalance: str
) -> pd.Series:
    rebalance_mode = (rebalance or "daily").lower()
    if rebalance_mode in {"daily", "none"}:
        return frame[exposure_col]

    dates = pd.to_datetime(frame["date"])
    if rebalance_mode == "monthly":
        key = dates.dt.to_period("M")
    elif rebalance_mode == "weekly":
        key = dates.dt.to_period("W-FRI")
    else:
        raise ValueError("rebalance must be daily, weekly, monthly, or none")

    rebalance_dates = dates.groupby(key).max()
    mask = dates.isin(rebalance_dates)
    exposures = pd.Series(np.nan, index=frame.index)
    exposures[mask] = frame.loc[mask, exposure_col]
    exposures = exposures.ffill()
    return exposures.fillna(frame[exposure_col])
