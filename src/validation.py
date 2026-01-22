from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.plotting import save_fig


@dataclass
class ValidationResults:
    lb_stat: float
    lb_pvalue: float
    lb_sq_stat: float
    lb_sq_pvalue: float
    arch_stat: float
    arch_pvalue: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_validation(
    residuals: pd.Series | np.ndarray,
    output_dir: Path,
    lags: int = 20,
) -> ValidationResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)
    residuals = pd.Series(residuals).dropna()

    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(residuals**2, lags=[lags], return_df=True)
    arch = het_arch(residuals, nlags=lags)

    summary_path = data_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Ljung-Box on standardized residuals:",
                f"  statistic: {float(lb['lb_stat'].iloc[0]):.6f}",
                f"  p-value:   {float(lb['lb_pvalue'].iloc[0]):.6f}",
                "",
                "Ljung-Box on squared standardized residuals:",
                f"  statistic: {float(lb_sq['lb_stat'].iloc[0]):.6f}",
                f"  p-value:   {float(lb_sq['lb_pvalue'].iloc[0]):.6f}",
                "",
                "ARCH test on standardized residuals:",
                f"  statistic: {float(arch[0]):.6f}",
                f"  p-value:   {float(arch[1]):.6f}",
                "",
                f"Lags used: {lags}",
            ]
        ),
        encoding="utf-8",
    )

    _plot_residuals(residuals, plots_dir)
    _plot_residual_acf(residuals, plots_dir, lags)
    _plot_residual_qq(residuals, plots_dir)

    return ValidationResults(
        lb_stat=float(lb["lb_stat"].iloc[0]),
        lb_pvalue=float(lb["lb_pvalue"].iloc[0]),
        lb_sq_stat=float(lb_sq["lb_stat"].iloc[0]),
        lb_sq_pvalue=float(lb_sq["lb_pvalue"].iloc[0]),
        arch_stat=float(arch[0]),
        arch_pvalue=float(arch[1]),
    )


def _plot_residuals(residuals: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(residuals.index, residuals.values, color="slategray", linewidth=0.7)
    ax.set_title("Standardized Residuals")
    ax.set_ylabel("Residual")
    ax.set_xlabel("Index")
    save_fig(fig, output_dir / "residuals_series.png")


def _plot_residual_acf(residuals: pd.Series, output_dir: Path, lags: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(residuals, lags=lags, ax=axes[0])
    axes[0].set_title("ACF: Standardized Residuals")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    plot_acf(residuals**2, lags=lags, ax=axes[1])
    axes[1].set_title("ACF: Squared Residuals")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")
    save_fig(fig, output_dir / "residuals_acf.png")


def _plot_residual_qq(residuals: pd.Series, output_dir: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    qqplot(residuals, line="s", ax=fig.gca())
    plt.title("Q-Q Plot: Standardized Residuals")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    save_fig(fig, output_dir / "residuals_qq.png")
