from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.garch_utils import fit_garch_variant
from tqdm import tqdm


@dataclass
class OOSResults:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_oos_check(
    data: pd.DataFrame,
    output_dir: Path,
    split_date: str = "2024-01-01",
    arma_order: tuple[int, int] = (2, 0),
    variant: str = "GARCH",
) -> OOSResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    split_mask = data["date"] < split_date
    train = data[split_mask].copy()
    test = data[~split_mask].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty set.")

    returns_train = train["log_return"].dropna()
    arma_result = ARIMA(returns_train, order=(arma_order[0], 0, arma_order[1])).fit()

    resid = arma_result.resid.dropna()
    garch_result = fit_garch_variant(resid, variant)

    data["realized_vol"] = data["log_return"].rolling(21).std() * (252**0.5) * 100
    test = data.loc[train.index.max() + 1 :].copy()

    static_available = True
    try:
        horizon = len(test)
        forecasts = garch_result.forecast(horizon=horizon, reindex=False)
        forecast_var = forecasts.variance.iloc[-1]
        forecast_vol = (forecast_var**0.5) * (252**0.5) * 100
        test = test.iloc[: len(forecast_vol)].copy()
        test["forecast_vol"] = forecast_vol.to_numpy()
    except ValueError:
        static_available = False
        test["forecast_vol"] = pd.NA

    test = test.dropna(subset=["realized_vol"])

    if static_available:
        _plot_oos(
            test.dropna(subset=["forecast_vol"]),
            plots_dir,
            variant=variant,
        )

    rolling = _rolling_forecast(data, test.index, arma_order, variant)
    test["forecast_vol_rolling"] = pd.to_numeric(rolling, errors="coerce")
    test["realized_vol"] = pd.to_numeric(test["realized_vol"], errors="coerce")
    rolling_plot = (
        test[["date", "forecast_vol_rolling", "realized_vol"]]
        .dropna()
        .rename(columns={"forecast_vol_rolling": "forecast_vol"})
    )
    _plot_oos(
        rolling_plot,
        plots_dir,
        suffix="_rolling",
        title="Rolling 1-Step Forecast vs Realized",
        variant=variant,
    )

    summary_path = data_dir / "summary.txt"
    static_corr = (
        test["forecast_vol"].corr(test["realized_vol"]) if static_available else float("nan")
    )
    rolling_corr = test["forecast_vol_rolling"].corr(test["realized_vol"])
    static_rmse = (
        ((test["forecast_vol"] - test["realized_vol"]) ** 2).mean() ** 0.5
        if static_available
        else float("nan")
    )
    rolling_rmse = (
        (test["forecast_vol_rolling"] - test["realized_vol"]) ** 2
    ).mean() ** 0.5
    summary_path.write_text(
        "\n".join(
            [
                "Out-of-sample check:",
                f"  split date: {split_date}",
                f"  train rows: {len(train):,}",
                f"  test rows:  {len(test):,}",
                f"  ARMA order: ({arma_order[0]}, {arma_order[1]})",
                "",
                "OOS metrics (static forecast vs realized):",
                f"  corr:  {static_corr:.4f}",
                f"  rmse:  {static_rmse:.4f}",
                f"  available: {static_available}",
                "",
                "OOS metrics (rolling 1-step vs realized):",
                f"  corr:  {rolling_corr:.4f}",
                f"  rmse:  {rolling_rmse:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    test[["date", "forecast_vol", "forecast_vol_rolling", "realized_vol"]].to_csv(
        data_dir / "forecast_vs_realized.csv", index=False
    )

    pd.DataFrame(
        [
            {"forecast": "static", "corr": static_corr, "rmse": static_rmse},
            {"forecast": "rolling", "corr": rolling_corr, "rmse": rolling_rmse},
        ]
    ).to_csv(data_dir / "oos_metrics.csv", index=False)

    return OOSResults(
        train_start=train["date"].min().date().isoformat(),
        train_end=train["date"].max().date().isoformat(),
        test_start=test["date"].min().date().isoformat(),
        test_end=test["date"].max().date().isoformat(),
    )


def _plot_oos(
    data: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
    title: str = "Out-of-Sample Volatility Forecast vs Realized",
    variant: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        data["date"],
        data["forecast_vol"],
        color="royalblue",
        label="Forecast Vol (annualized %)",
        linewidth=0.8,
    )
    ax.plot(
        data["date"],
        data["realized_vol"],
        color="darkorange",
        label="21D Realized Vol (annualized %)",
        linewidth=0.8,
        alpha=0.7,
    )
    if variant:
        ax.set_title(f"{title} ({variant})")
    else:
        ax.set_title(title)
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"forecast_vs_realized{suffix}.png", dpi=150)
    plt.close(fig)


def _rolling_forecast(
    data: pd.DataFrame,
    test_index: pd.Index,
    arma_order: tuple[int, int],
    variant: str,
) -> pd.Series:
    forecasts: list[float] = []
    for pos in tqdm(test_index, desc="Rolling OOS forecasts"):
        window = data.loc[:pos - 1]
        returns = window["log_return"].dropna()
        arma_result = ARIMA(returns, order=(arma_order[0], 0, arma_order[1])).fit()
        resid = arma_result.resid.dropna()
        garch_result = fit_garch_variant(resid, variant)
        forecast = garch_result.forecast(horizon=1, reindex=False)
        var = forecast.variance.iloc[-1].iloc[0]
        vol = (var**0.5) * (252**0.5) * 100
        forecasts.append(float(vol))
    return pd.Series(forecasts, index=test_index)
