from __future__ import annotations

import numpy as np
import pandas as pd


def _select_spx_price(spx_df: pd.DataFrame) -> pd.DataFrame:
    if "Adj Close" in spx_df.columns:
        price_col = "Adj Close"
    elif "Close" in spx_df.columns:
        price_col = "Close"
    else:
        raise ValueError("SPX data must include 'Adj Close' or 'Close'.")

    spx = spx_df[["date", price_col]].copy()
    spx = spx.rename(columns={price_col: "spx_adj_close"})
    return spx


def _select_vix_close(vix_df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in vix_df.columns:
        raise ValueError("VIX data must include 'Close'.")
    vix = vix_df[["date", "Close"]].copy()
    vix = vix.rename(columns={"Close": "vix_close"})
    return vix


def prepare_aligned_dataset(spx_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    spx = _select_spx_price(spx_df)
    vix = _select_vix_close(vix_df)

    aligned = spx.merge(vix, on="date", how="left")
    aligned = aligned.dropna(subset=["spx_adj_close", "vix_close"]).copy()

    aligned["log_return"] = np.log(
        aligned["spx_adj_close"] / aligned["spx_adj_close"].shift(1)
    )
    aligned["sq_return"] = aligned["log_return"] ** 2

    aligned = aligned.dropna(
        subset=["log_return", "sq_return"]).reset_index(drop=True)
    return aligned
