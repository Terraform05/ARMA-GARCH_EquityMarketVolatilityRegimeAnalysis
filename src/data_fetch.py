from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


def _to_date(value: str) -> date:
    return pd.to_datetime(value).date()


def download_yahoo_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    start_date = _to_date(start)
    end_date = _to_date(end)
    inclusive_end = end_date + timedelta(days=1)

    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=inclusive_end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError(
            f"No data returned for {ticker} between {start} and {end}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= end_date]
    return df
