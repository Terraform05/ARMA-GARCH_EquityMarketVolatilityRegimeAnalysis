from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class RegimeResults:
    low_thresh: float
    high_thresh: float
    realized_window: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _assign_regimes(series: pd.Series) -> tuple[pd.Series, float, float]:
    low_thresh = series.quantile(0.33)
    high_thresh = series.quantile(0.66)

    def _label(value: float) -> str:
        if value <= low_thresh:
            return "low"
        if value >= high_thresh:
            return "high"
        return "mid"

    regimes = series.apply(_label)
    return regimes, float(low_thresh), float(high_thresh)


def run_regime_analysis(
    data: pd.DataFrame,
    conditional_vol: pd.DataFrame,
    output_dir: Path,
) -> RegimeResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)

    merged = data.merge(conditional_vol, on="date", how="inner")
    merged = merged.dropna(subset=["cond_vol", "vix_close", "log_return"])

    realized_window, realized_vol = _select_realized_window(
        merged["log_return"],
        merged["vix_close"],
        windows=[10, 21, 63],
        data_dir=data_dir,
        plots_dir=plots_dir,
    )
    merged["realized_vol"] = realized_vol
    merged = merged.dropna(subset=["realized_vol"])

    regimes, low_thresh, high_thresh = _assign_regimes(merged["cond_vol"])
    merged["regime"] = regimes

    _plot_regimes(merged, plots_dir)
    _plot_vix_vs_realized(merged, plots_dir, realized_window)

    summary_path = data_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Regime thresholds based on conditional volatility quantiles:",
                f"  low <= {low_thresh:.6f}",
                f"  high >= {high_thresh:.6f}",
                "",
                "Realized volatility window selection:",
                f"  window (days): {realized_window}",
            ]
        ),
        encoding="utf-8",
    )

    merged = merged.assign(realized_window=realized_window)
    merged.to_csv(data_dir / "regime_series.csv", index=False)
    _write_regime_outcomes(merged, data_dir, plots_dir)

    return RegimeResults(
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        realized_window=realized_window,
    )


def _plot_regimes(data: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"low": "seagreen", "mid": "goldenrod", "high": "firebrick"}
    for label, color in colors.items():
        mask = data["regime"] == label
        ax.scatter(
            data.loc[mask, "date"],
            data.loc[mask, "cond_vol"],
            s=6,
            color=color,
            label=label,
            alpha=0.6,
        )

    ax.set_title("Conditional Volatility Regimes")
    ax.set_ylabel("Conditional Volatility")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "regimes.png", dpi=150)
    plt.close(fig)


def _plot_vix_vs_realized(
    data: pd.DataFrame, output_dir: Path, realized_window: int
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        data["date"],
        data["vix_close"],
        color="slateblue",
        label="VIX (annualized %)",
        linewidth=0.8,
    )
    ax.plot(
        data["date"],
        data["realized_vol"],
        color="darkorange",
        label=f"{realized_window}D Realized Vol (annualized %)",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.set_title(f"VIX vs {realized_window}D Realized Volatility")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_vs_realized.png", dpi=150)
    plt.close(fig)


def _select_realized_window(
    returns: pd.Series,
    vix: pd.Series,
    windows: list[int],
    data_dir: Path,
    plots_dir: Path,
) -> tuple[int, pd.Series]:
    best_window = windows[0]
    best_corr = float("-inf")
    best_series = pd.Series(dtype=float)
    rows = []

    for window in windows:
        realized = returns.rolling(window).std() * (252**0.5) * 100
        aligned = pd.concat([realized, vix], axis=1).dropna()
        if aligned.empty:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        rmse = ((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2).mean() ** 0.5
        rows.append({"window": window, "corr": corr, "rmse": rmse})

        if corr > best_corr:
            best_corr = corr
            best_window = window
            best_series = realized

    metrics = pd.DataFrame(rows).sort_values("window")
    metrics.to_csv(data_dir / "realized_window_metrics.csv", index=False)
    _plot_window_metrics(metrics, plots_dir)

    return best_window, best_series


def _plot_window_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(metrics["window"].astype(str), metrics["corr"], color="slateblue")
    axes[0].set_title("Correlation vs VIX")
    axes[0].set_xlabel("Window (Days)")
    axes[0].set_ylabel("Correlation")

    axes[1].bar(metrics["window"].astype(str), metrics["rmse"], color="darkorange")
    axes[1].set_title("RMSE vs VIX")
    axes[1].set_xlabel("Window (Days)")
    axes[1].set_ylabel("RMSE")

    fig.tight_layout()
    fig.savefig(output_dir / "realized_window_metrics.png", dpi=150)
    plt.close(fig)


def _write_regime_outcomes(
    data: pd.DataFrame, data_dir: Path, plots_dir: Path
) -> None:
    data = data.copy()
    data["rolling_peak"] = data["spx_adj_close"].cummax()
    data["drawdown"] = data["spx_adj_close"] / data["rolling_peak"] - 1.0

    outcomes = (
        data.groupby("regime", as_index=False)
        .agg(
            avg_log_return=("log_return", "mean"),
            avg_cond_vol=("cond_vol", "mean"),
            avg_realized_vol=("realized_vol", "mean"),
            avg_vix=("vix_close", "mean"),
            min_drawdown=("drawdown", "min"),
            obs=("regime", "size"),
        )
        .sort_values("regime")
    )
    outcomes.to_csv(data_dir / "regime_outcomes.csv", index=False)
    _plot_regime_outcomes(outcomes, plots_dir)


def _plot_regime_outcomes(outcomes: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(outcomes["regime"], outcomes["avg_log_return"], color="slateblue")
    axes[0].set_title("Avg Log Return by Regime")
    axes[0].set_ylabel("Avg Log Return")
    axes[0].set_xlabel("Regime")

    axes[1].bar(outcomes["regime"], outcomes["avg_vix"], color="darkorange")
    axes[1].set_title("Avg VIX by Regime")
    axes[1].set_ylabel("VIX Level")
    axes[1].set_xlabel("Regime")

    axes[2].bar(outcomes["regime"], outcomes["min_drawdown"], color="firebrick")
    axes[2].set_title("Min Drawdown by Regime")
    axes[2].set_ylabel("Drawdown")
    axes[2].set_xlabel("Regime")

    fig.tight_layout()
    fig.savefig(output_dir / "regime_outcomes.png", dpi=150)
    plt.close(fig)
