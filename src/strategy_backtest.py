from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class StrategyResults:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_strategy_backtest(
    data: pd.DataFrame,
    output_dir: Path,
    exposure_map: dict[str, float] | None = None,
    exposure_candidates: list[dict[str, float]] | None = None,
) -> StrategyResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)

    if exposure_map is None:
        exposure_map = {"low": 1.0, "mid": 0.75, "high": 0.25}
    if exposure_candidates is None:
        exposure_candidates = [
            {"low": 1.0, "mid": 0.75, "high": 0.25},
            {"low": 1.25, "mid": 1.0, "high": 0.5},
            {"low": 1.0, "mid": 1.0, "high": 0.5},
            {"low": 1.0, "mid": 0.5, "high": 0.0},
        ]

    frame = data.copy()
    frame = frame.dropna(subset=["log_return", "regime"])
    frame["exposure"] = frame["regime"].map(exposure_map).fillna(0.0)
    frame["strategy_log_return"] = frame["log_return"] * frame["exposure"]

    frame["benchmark_equity"] = np.exp(frame["log_return"].cumsum())
    frame["strategy_equity"] = np.exp(frame["strategy_log_return"].cumsum())

    frame["rolling_peak"] = frame["strategy_equity"].cummax()
    frame["drawdown"] = frame["strategy_equity"] / frame["rolling_peak"] - 1.0

    stats = _compute_stats(frame["strategy_log_return"])
    variants = _evaluate_candidates(frame, exposure_candidates)
    benchmark_stats = _compute_stats(frame["log_return"])
    summary_path = data_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Strategy backtest:",
                f"  annual return: {stats.annual_return:.4f}",
                f"  annual vol: {stats.annual_vol:.4f}",
                f"  sharpe: {stats.sharpe:.4f}",
                f"  max drawdown: {stats.max_drawdown:.4f}",
                "",
                "Benchmark (buy-and-hold):",
                f"  annual return: {benchmark_stats.annual_return:.4f}",
                f"  annual vol: {benchmark_stats.annual_vol:.4f}",
                f"  sharpe: {benchmark_stats.sharpe:.4f}",
                f"  max drawdown: {benchmark_stats.max_drawdown:.4f}",
                "",
                f"  excess return: {(stats.annual_return - benchmark_stats.annual_return):.4f}",
                "",
                "Best candidate by Sharpe:",
                f"  exposure map: {variants.iloc[0]['exposure_map']}",
                f"  sharpe: {variants.iloc[0]['sharpe']:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    frame[
        [
            "date",
            "regime",
            "exposure",
            "strategy_equity",
            "benchmark_equity",
            "drawdown",
        ]
    ].to_csv(data_dir / "strategy_equity.csv", index=False)

    _write_regime_performance(frame, data_dir)
    _write_exposure_stats(frame, data_dir)

    _plot_equity_curve(frame, plots_dir, suffix="")
    _plot_equity_curve(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_exposure_overlay(frame, plots_dir, suffix="")
    _plot_exposure_overlay(_last_year_slice(frame), plots_dir, suffix="_last_year")
    variants.to_csv(data_dir / "strategy_variants.csv", index=False)

    return stats


def _compute_stats(strategy_log_returns: pd.Series) -> StrategyResults:
    ann_return = strategy_log_returns.mean() * 252
    ann_vol = strategy_log_returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity = np.exp(strategy_log_returns.cumsum())
    rolling_peak = pd.Series(equity).cummax().values
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = drawdown.min()

    return StrategyResults(
        annual_return=float(ann_return),
        annual_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
    )


def _plot_equity_curve(frame: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    benchmark = frame["benchmark_equity"]
    strategy = frame["strategy_equity"]
    if len(frame) > 0 and suffix.endswith("last_year"):
        # Rebase to a common start date for a clean last-year comparison.
        benchmark = benchmark / benchmark.iloc[0]
        strategy = strategy / strategy.iloc[0]
    ax.plot(
        frame["date"],
        benchmark,
        color="black",
        linewidth=0.9,
        label="Benchmark (Buy & Hold)",
    )
    ax.plot(
        frame["date"],
        strategy,
        color="slateblue",
        linewidth=0.9,
        label="Regime Strategy",
    )
    ax.set_title("Regime-Aware Equity Curve")
    ax.set_ylabel("Equity (Indexed)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"equity_curve{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_exposure_overlay(
    frame: pd.DataFrame, output_dir: Path, suffix: str
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.step(
        frame["date"],
        frame["exposure"],
        where="post",
        color="slateblue",
        linewidth=1.1,
        label="Exposure",
    )
    ax_top.set_title("Exposure with Regime Strip")
    ax_top.set_ylabel("Exposure")
    ax_top.legend(loc="upper left", frameon=False)

    regime_colors = {"low": "#4CAF50", "mid": "#FFB74D", "high": "#E57373"}
    regimes = frame["regime"].fillna("unknown").tolist()
    dates = pd.to_datetime(frame["date"])
    start_idx = 0
    # Draw a categorical regime strip for readability.
    for idx in range(1, len(regimes) + 1):
        if idx == len(regimes) or regimes[idx] != regimes[start_idx]:
            regime = regimes[start_idx]
            color = regime_colors.get(regime, "#BDBDBD")
            ax_bottom.axvspan(
                dates.iloc[start_idx],
                dates.iloc[idx - 1],
                color=color,
                alpha=0.85,
                linewidth=0,
            )
            start_idx = idx

    ax_bottom.set_yticks([])
    ax_bottom.set_xlabel("Date")
    ax_bottom.set_title("Regime (low / mid / high)")
    fig.tight_layout()
    fig.savefig(output_dir / f"exposure_overlay{suffix}.png", dpi=150)
    plt.close(fig)


def _last_year_slice(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    if dates.empty:
        return frame
    cutoff = dates.max() - pd.DateOffset(years=1)
    return frame.loc[dates >= cutoff].copy()


def _write_regime_performance(frame: pd.DataFrame, data_dir: Path) -> None:
    perf = (
        frame.groupby("regime", as_index=False)
        .agg(
            avg_log_return=("log_return", "mean"),
            avg_strategy_log_return=("strategy_log_return", "mean"),
            avg_exposure=("exposure", "mean"),
            obs=("regime", "size"),
        )
        .sort_values("regime")
    )
    perf.to_csv(data_dir / "regime_performance.csv", index=False)


def _write_exposure_stats(frame: pd.DataFrame, data_dir: Path) -> None:
    stats = frame["exposure"].describe().to_frame(name="value")
    stats.to_csv(data_dir / "exposure_stats.csv")


def _evaluate_candidates(
    frame: pd.DataFrame, candidates: list[dict[str, float]]
) -> pd.DataFrame:
    rows = []
    for mapping in candidates:
        exposure = frame["regime"].map(mapping).fillna(0.0)
        strat_ret = frame["log_return"] * exposure
        stats = _compute_stats(strat_ret)
        rows.append(
            {
                "exposure_map": mapping,
                "annual_return": stats.annual_return,
                "annual_vol": stats.annual_vol,
                "sharpe": stats.sharpe,
                "max_drawdown": stats.max_drawdown,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)
