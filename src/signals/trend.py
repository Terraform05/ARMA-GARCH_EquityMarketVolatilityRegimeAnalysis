from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_score(
    price: pd.Series, trend_window: int, z_window: int
) -> pd.Series:
    log_price = np.log(price)
    trend_raw = (log_price - log_price.shift(trend_window)) / float(trend_window)
    mean = trend_raw.rolling(z_window, min_periods=z_window).mean()
    std = trend_raw.rolling(z_window, min_periods=z_window).std()
    return (trend_raw - mean) / std


def bucket_trend_state(trend_score: pd.Series, threshold: float) -> pd.Series:
    states = pd.Series("neutral", index=trend_score.index)
    states = states.mask(trend_score >= threshold, "strong_up")
    states = states.mask(trend_score <= -threshold, "strong_down")
    return states


def confirm_states(states: pd.Series, confirm: int) -> pd.Series:
    if confirm <= 1:
        return states
    confirmed: list[str] = []
    last_state: str | None = None
    pending_state: str | None = None
    pending_count = 0

    for state in states:
        if last_state is None:
            last_state = state
            confirmed.append(state)
            continue

        if state == last_state:
            pending_state = None
            pending_count = 0
            confirmed.append(last_state)
            continue

        if pending_state != state:
            pending_state = state
            pending_count = 1
        else:
            pending_count += 1

        if pending_count >= confirm:
            last_state = pending_state
            pending_state = None
            pending_count = 0

        confirmed.append(last_state)

    return pd.Series(confirmed, index=states.index)
