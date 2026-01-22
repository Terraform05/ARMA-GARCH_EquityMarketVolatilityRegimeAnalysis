from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller

from src.plotting import format_date_axis, save_fig

@dataclass
class DiagnosticResults:
    adf_stat: float
    adf_pvalue: float
    arch_stat: float
    arch_pvalue: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_diagnostics(
    data: pd.DataFrame,
    output_dir: Path,
    lags: int = 20,
) -> DiagnosticResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)
    returns = data["log_return"].dropna()

    adf_result = adfuller(returns)
    arch_result = het_arch(returns, nlags=lags)

    _plot_series(data, plots_dir)
    _plot_acf_pacf(returns, plots_dir, lags=lags)

    summary_path = data_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "ADF test on log_return:",
                f"  statistic: {adf_result[0]:.6f}",
                f"  p-value:   {adf_result[1]:.6f}",
                "",
                "ARCH test on log_return:",
                f"  statistic: {arch_result[0]:.6f}",
                f"  p-value:   {arch_result[1]:.6f}",
                "",
                f"Lags used: {lags}",
            ]
        ),
        encoding="utf-8",
    )

    return DiagnosticResults(
        adf_stat=float(adf_result[0]),
        adf_pvalue=float(adf_result[1]),
        arch_stat=float(arch_result[0]),
        arch_pvalue=float(arch_result[1]),
    )


def _plot_series(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(data["date"], data["log_return"], color="steelblue", linewidth=0.8)
    axes[0].set_title("SPX Log Returns")
    axes[0].set_ylabel("Log Return")
    axes[0].set_xlabel("Date")

    axes[1].plot(data["date"], data["sq_return"], color="firebrick", linewidth=0.8)
    axes[1].set_title("Squared Returns")
    axes[1].set_ylabel("Squared Return")
    axes[1].set_xlabel("Date")

    format_date_axis(axes[-1])
    save_fig(fig, output_dir / "returns_series.png")


def _plot_acf_pacf(returns: pd.Series, output_dir: Path, lags: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(returns, lags=lags, ax=axes[0])
    axes[0].set_title("ACF: Log Returns")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    plot_pacf(returns, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("PACF: Log Returns")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    save_fig(fig, output_dir / "acf_pacf.png")
