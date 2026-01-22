from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting import format_date_axis, save_fig
from src.signals.trend import bucket_trend_state, compute_trend_score, confirm_states
from src.signals.volatility import compute_vol_score, recent_regime_change
from src.strategies.sizing import apply_rebalance, continuous_exposure, matrix_exposure

TREND_STATES = ("strong_up", "neutral", "strong_down")

DEFAULT_EXPOSURE_MATRIX: dict[str, dict[str, float]] = {
    "low": {"strong_up": 1.1, "neutral": 0.8, "strong_down": 0.5},
    "mid": {"strong_up": 0.9, "neutral": 0.6, "strong_down": 0.3},
    "high": {"strong_up": 0.7, "neutral": 0.4, "strong_down": 0.1},
}

REGIME_EXPOSURE_MAPS: dict[str, dict[str, float]] = {
    "default": {"low": 1.0, "mid": 0.75, "high": 0.25},
    "sharpe_best": {"low": 1.0, "mid": 1.0, "high": 0.5},
    "aggressive": {"low": 1.25, "mid": 1.0, "high": 0.5},
    "defensive": {"low": 1.0, "mid": 0.5, "high": 0.0},
}


@dataclass
class RegimeTrendResults:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


@dataclass
class AlphaBeta:
    alpha_daily: float
    alpha_annual: float
    beta: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_regime_trend_frame(
    data: pd.DataFrame,
    *,
    trend_score: pd.Series | None = None,
    trend_state: pd.Series | None = None,
    trend_window: int = 126,
    trend_z_window: int = 252,
    trend_threshold: float = 0.5,
    state_confirm: int = 2,
    sizing_mode: str = "matrix",
    exposure_matrix: dict[str, dict[str, float]] | None = None,
    base_exposure: float = 0.6,
    trend_coef: float = 0.25,
    vol_coef: float = 0.25,
    vol_z_window: int = 252,
    min_exposure: float | None = 0.0,
    max_exposure: float | None = None,
    rebalance: str = "monthly",
    transition_window: int = 0,
    transition_multiplier: float = 1.0,
    cost_bps: float = 0.0,
    vol_score: pd.Series | None = None,
) -> pd.DataFrame:
    frame = data.copy()
    frame = frame.dropna(subset=["date", "log_return", "regime", "spx_adj_close"])
    frame = frame.sort_values("date")

    trend_score_series = trend_score
    if trend_score_series is None:
        trend_score_series = compute_trend_score(
            frame["spx_adj_close"], trend_window, trend_z_window
        )
    trend_score_series = trend_score_series.reindex(frame.index).fillna(0.0)
    frame["trend_score"] = trend_score_series

    if trend_state is None:
        trend_state_series = bucket_trend_state(trend_score_series, trend_threshold)
        trend_state_series = confirm_states(trend_state_series, state_confirm)
    else:
        trend_state_series = trend_state.reindex(frame.index)
    frame["trend_state"] = trend_state_series

    if sizing_mode not in {"matrix", "continuous"}:
        raise ValueError("sizing_mode must be 'matrix' or 'continuous'")

    if sizing_mode == "continuous":
        vol_score_series = vol_score
        if vol_score_series is None:
            vol_score_series = compute_vol_score(
                frame["cond_vol"], vol_z_window
            )
        vol_score_series = vol_score_series.reindex(frame.index).fillna(0.0)
        frame["vol_score"] = vol_score_series
        frame["target_exposure"] = continuous_exposure(
            trend_score=trend_score_series,
            vol_score=vol_score_series,
            base_exposure=base_exposure,
            trend_coef=trend_coef,
            vol_coef=vol_coef,
        )
    else:
        matrix = exposure_matrix or DEFAULT_EXPOSURE_MATRIX
        frame["target_exposure"] = matrix_exposure(
            frame["regime"], frame["trend_state"], matrix
        )

    if transition_window > 0:
        recent_change = recent_regime_change(frame["regime"], transition_window)
        frame.loc[recent_change, "target_exposure"] = (
            frame.loc[recent_change, "target_exposure"] * transition_multiplier
        )

    if min_exposure is not None:
        frame["target_exposure"] = frame["target_exposure"].clip(lower=min_exposure)
    if max_exposure is not None:
        frame["target_exposure"] = frame["target_exposure"].clip(upper=max_exposure)

    frame["exposure"] = apply_rebalance(
        frame[["date", "target_exposure"]].copy(),
        exposure_col="target_exposure",
        rebalance=rebalance,
    )

    frame["turnover"] = frame["exposure"].diff().abs().fillna(0.0)
    frame["cost_rate"] = frame["turnover"] * (cost_bps / 10000.0)

    frame["strategy_log_return"] = frame["log_return"] * frame["exposure"]
    frame["strategy_log_return_net"] = (
        frame["strategy_log_return"] - frame["cost_rate"]
    )
    frame["benchmark_return"] = np.exp(frame["log_return"]) - 1.0
    frame["strategy_return"] = np.exp(frame["strategy_log_return"]) - 1.0
    frame["strategy_return_net"] = np.exp(frame["strategy_log_return_net"]) - 1.0
    frame["benchmark_equity"] = np.exp(frame["log_return"].cumsum())
    frame["strategy_equity"] = np.exp(frame["strategy_log_return"].cumsum())
    frame["strategy_equity_net"] = np.exp(
        frame["strategy_log_return_net"].cumsum()
    )

    frame["rolling_peak"] = frame["strategy_equity"].cummax()
    frame["drawdown"] = frame["strategy_equity"] / frame["rolling_peak"] - 1.0
    frame["rolling_peak_net"] = frame["strategy_equity_net"].cummax()
    frame["drawdown_net"] = (
        frame["strategy_equity_net"] / frame["rolling_peak_net"] - 1.0
    )

    return frame


def run_regime_trend_backtest(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    trend_window: int = 126,
    trend_z_window: int = 252,
    trend_threshold: float = 0.5,
    state_confirm: int = 2,
    sizing_mode: str = "matrix",
    exposure_matrix: dict[str, dict[str, float]] | None = None,
    base_exposure: float = 0.6,
    trend_coef: float = 0.25,
    vol_coef: float = 0.25,
    vol_z_window: int = 252,
    min_exposure: float = 0.0,
    max_exposure: float | None = None,
    rebalance: str = "monthly",
    transition_window: int = 0,
    transition_multiplier: float = 1.0,
    cost_bps: float = 0.0,
) -> RegimeTrendResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)

    frame = build_regime_trend_frame(
        data,
        trend_window=trend_window,
        trend_z_window=trend_z_window,
        trend_threshold=trend_threshold,
        state_confirm=state_confirm,
        sizing_mode=sizing_mode,
        exposure_matrix=exposure_matrix,
        base_exposure=base_exposure,
        trend_coef=trend_coef,
        vol_coef=vol_coef,
        vol_z_window=vol_z_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        rebalance=rebalance,
        transition_window=transition_window,
        transition_multiplier=transition_multiplier,
        cost_bps=cost_bps,
    )

    _apply_regime_only(frame)
    trend_only_map = _derive_trend_only_map(exposure_matrix or DEFAULT_EXPOSURE_MATRIX)
    _apply_trend_only(frame, trend_only_map)
    stats = _compute_stats(frame["strategy_log_return"])
    stats_net = _compute_stats(frame["strategy_log_return_net"])
    benchmark_stats = _compute_stats(frame["log_return"])
    alpha_beta = _compute_alpha_beta(
        frame["strategy_return"], frame["benchmark_return"]
    )
    alpha_beta_net = _compute_alpha_beta(
        frame["strategy_return_net"], frame["benchmark_return"]
    )
    alpha_beta_regime = _compute_alpha_beta(
        frame["regime_return"], frame["benchmark_return"]
    )
    stats_trend = _compute_stats(frame["trend_only_log_return"])
    alpha_beta_trend = _compute_alpha_beta(
        frame["trend_only_return"], frame["benchmark_return"]
    )
    avg_turnover = float(frame["turnover"].mean())
    annual_turnover = avg_turnover * 252
    annual_cost_drag = annual_turnover * (cost_bps / 10000.0)

    _write_summary(
        data_dir / "summary.txt",
        stats=stats,
        stats_net=stats_net,
        benchmark_stats=benchmark_stats,
        alpha_beta=alpha_beta,
        alpha_beta_net=alpha_beta_net,
        alpha_beta_regime=alpha_beta_regime,
        stats_trend=stats_trend,
        alpha_beta_trend=alpha_beta_trend,
        trend_only_map=trend_only_map,
        trend_window=trend_window,
        trend_z_window=trend_z_window,
        trend_threshold=trend_threshold,
        state_confirm=state_confirm,
        sizing_mode=sizing_mode,
        exposure_matrix=exposure_matrix or DEFAULT_EXPOSURE_MATRIX,
        base_exposure=base_exposure,
        trend_coef=trend_coef,
        vol_coef=vol_coef,
        vol_z_window=vol_z_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        rebalance=rebalance,
        transition_window=transition_window,
        transition_multiplier=transition_multiplier,
        cost_bps=cost_bps,
        avg_turnover=avg_turnover,
        annual_turnover=annual_turnover,
        annual_cost_drag=annual_cost_drag,
    )

    frame[
        [
            "date",
            "regime",
            "trend_score",
            "trend_state",
            "exposure",
            "turnover",
            "cost_rate",
            "strategy_equity",
            "strategy_equity_net",
            "benchmark_equity",
            "regime_equity",
            "drawdown",
            "drawdown_net",
            "regime_drawdown",
            "strategy_return",
            "strategy_return_net",
            "benchmark_return",
            "regime_return",
            "trend_only_exposure",
            "trend_only_equity",
            "trend_only_drawdown",
            "trend_only_return",
        ]
    ].to_csv(data_dir / "regime_trend_equity.csv", index=False)

    _write_state_performance(frame, data_dir)
    _write_exposure_stats(frame, data_dir)
    _write_turnover_stats(frame, data_dir)
    _write_cost_sensitivity(frame, data_dir)
    _write_regime_map_comparison(frame, data_dir)

    _plot_equity_curve(frame, plots_dir, suffix="")
    _plot_equity_curve(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_equity_curve(
        frame,
        plots_dir,
        suffix="_net",
        strategy_col="strategy_equity_net",
        label="Regime-Trend Strategy (Net)",
    )
    _plot_equity_curve(
        _last_year_slice(frame),
        plots_dir,
        suffix="_net_last_year",
        strategy_col="strategy_equity_net",
        label="Regime-Trend Strategy (Net)",
    )
    _plot_equity_curve_compare(frame, plots_dir, suffix="")
    _plot_equity_curve_compare(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_rolling_cagr(frame, plots_dir)
    _plot_rolling_drawdown(frame, plots_dir)
    _plot_rolling_alpha_beta(frame, plots_dir)
    _plot_exposure_overlay(frame, plots_dir, suffix="")
    _plot_exposure_overlay(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_turnover_hist(frame, plots_dir)
    _plot_cost_sensitivity(data_dir, plots_dir)

    return stats


def _compute_stats(strategy_log_returns: pd.Series) -> RegimeTrendResults:
    ann_return = strategy_log_returns.mean() * 252
    ann_vol = strategy_log_returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity = np.exp(strategy_log_returns.cumsum())
    rolling_peak = pd.Series(equity).cummax().values
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = drawdown.min()

    return RegimeTrendResults(
        annual_return=float(ann_return),
        annual_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
    )


def _compute_alpha_beta(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, rf: float = 0.0
) -> AlphaBeta:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return AlphaBeta(alpha_daily=float("nan"), alpha_annual=float("nan"), beta=float("nan"))
    strat = aligned.iloc[:, 0] - rf
    bench = aligned.iloc[:, 1] - rf
    mean_strat = strat.mean()
    mean_bench = bench.mean()
    var_bench = bench.var()
    if var_bench == 0 or np.isnan(var_bench):
        beta = 0.0
    else:
        cov = ((strat - mean_strat) * (bench - mean_bench)).mean()
        beta = cov / var_bench
    alpha_daily = mean_strat - beta * mean_bench
    alpha_annual = alpha_daily * 252
    return AlphaBeta(
        alpha_daily=float(alpha_daily),
        alpha_annual=float(alpha_annual),
        beta=float(beta),
    )


def _write_summary(
    path: Path,
    *,
    stats: RegimeTrendResults,
    stats_net: RegimeTrendResults,
    benchmark_stats: RegimeTrendResults,
    alpha_beta: AlphaBeta,
    alpha_beta_net: AlphaBeta,
    alpha_beta_regime: AlphaBeta,
    stats_trend: RegimeTrendResults,
    alpha_beta_trend: AlphaBeta,
    trend_only_map: dict[str, float],
    trend_window: int,
    trend_z_window: int,
    trend_threshold: float,
    state_confirm: int,
    sizing_mode: str,
    exposure_matrix: dict[str, dict[str, float]],
    base_exposure: float,
    trend_coef: float,
    vol_coef: float,
    vol_z_window: int,
    min_exposure: float,
    max_exposure: float | None,
    rebalance: str,
    transition_window: int,
    transition_multiplier: float,
    cost_bps: float,
    avg_turnover: float,
    annual_turnover: float,
    annual_cost_drag: float,
) -> None:
    lines = [
        "Regime-Trend strategy backtest:",
        f"  annual return: {stats.annual_return:.4f}",
        f"  annual vol: {stats.annual_vol:.4f}",
        f"  sharpe: {stats.sharpe:.4f}",
        f"  max drawdown: {stats.max_drawdown:.4f}",
        f"  alpha (annual): {alpha_beta.alpha_annual:.4f}",
        f"  beta: {alpha_beta.beta:.4f}",
        "",
        "Regime-Trend strategy (net of costs):",
        f"  annual return: {stats_net.annual_return:.4f}",
        f"  annual vol: {stats_net.annual_vol:.4f}",
        f"  sharpe: {stats_net.sharpe:.4f}",
        f"  max drawdown: {stats_net.max_drawdown:.4f}",
        f"  alpha (annual): {alpha_beta_net.alpha_annual:.4f}",
        f"  beta: {alpha_beta_net.beta:.4f}",
        "",
        "Regime-only strategy:",
        f"  alpha (annual): {alpha_beta_regime.alpha_annual:.4f}",
        f"  beta: {alpha_beta_regime.beta:.4f}",
        "",
        "Trend-only strategy:",
        f"  annual return: {stats_trend.annual_return:.4f}",
        f"  annual vol: {stats_trend.annual_vol:.4f}",
        f"  sharpe: {stats_trend.sharpe:.4f}",
        f"  max drawdown: {stats_trend.max_drawdown:.4f}",
        f"  alpha (annual): {alpha_beta_trend.alpha_annual:.4f}",
        f"  beta: {alpha_beta_trend.beta:.4f}",
        f"  exposure map: {trend_only_map}",
        "",
        "Benchmark (buy-and-hold):",
        f"  annual return: {benchmark_stats.annual_return:.4f}",
        f"  annual vol: {benchmark_stats.annual_vol:.4f}",
        f"  sharpe: {benchmark_stats.sharpe:.4f}",
        f"  max drawdown: {benchmark_stats.max_drawdown:.4f}",
        "",
        f"  excess return: {(stats.annual_return - benchmark_stats.annual_return):.4f}",
        f"  excess return (net): {(stats_net.annual_return - benchmark_stats.annual_return):.4f}",
        "",
        "Turnover and costs:",
        f"  cost_bps_per_turnover: {cost_bps:.2f}",
        f"  avg_turnover: {avg_turnover:.4f}",
        f"  annualized_turnover: {annual_turnover:.2f}",
        f"  annual_cost_drag: {annual_cost_drag:.4f}",
        "",
        "Directional signal:",
        f"  trend_window: {trend_window}",
        f"  trend_z_window: {trend_z_window}",
        f"  trend_threshold: {trend_threshold}",
        f"  state_confirm: {state_confirm}",
        "",
        "Sizing:",
        f"  sizing_mode: {sizing_mode}",
        f"  base_exposure: {base_exposure:.2f}",
        f"  trend_coef: {trend_coef:.2f}",
        f"  vol_coef: {vol_coef:.2f}",
        f"  vol_z_window: {vol_z_window}",
        f"  min_exposure: {min_exposure:.2f}",
        f"  max_exposure: {max_exposure}",
        f"  rebalance: {rebalance}",
        f"  transition_window: {transition_window}",
        f"  transition_multiplier: {transition_multiplier:.2f}",
        "",
        "Exposure matrix:",
        f"  low: {exposure_matrix.get('low')}",
        f"  mid: {exposure_matrix.get('mid')}",
        f"  high: {exposure_matrix.get('high')}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_equity_curve(
    frame: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    strategy_col: str = "strategy_equity",
    label: str = "Regime-Trend Strategy",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    benchmark = frame["benchmark_equity"]
    strategy = frame[strategy_col]
    if len(frame) > 0 and suffix.endswith("last_year"):
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
        label=label,
    )
    ax.set_title("Regime-Trend Strategy Equity Curve")
    ax.set_ylabel("Equity (Indexed)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=False)
    format_date_axis(ax)
    save_fig(fig, output_dir / f"equity_curve{suffix}.png")


def _plot_equity_curve_compare(
    frame: pd.DataFrame, output_dir: Path, suffix: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    benchmark = frame["benchmark_equity"]
    layered = frame["strategy_equity"]
    regime = frame["regime_equity"]
    trend_only = frame["trend_only_equity"]
    if len(frame) > 0 and suffix.endswith("last_year"):
        benchmark = benchmark / benchmark.iloc[0]
        layered = layered / layered.iloc[0]
        regime = regime / regime.iloc[0]
        trend_only = trend_only / trend_only.iloc[0]
    ax.plot(
        frame["date"],
        benchmark,
        color="black",
        linewidth=0.9,
        label="Benchmark (Buy & Hold)",
    )
    ax.plot(
        frame["date"],
        regime,
        color="#7E57C2",
        linewidth=0.9,
        label="Regime Strategy",
    )
    ax.plot(
        frame["date"],
        trend_only,
        color="#26A69A",
        linewidth=0.9,
        label="Trend-Only Strategy",
    )
    ax.plot(
        frame["date"],
        layered,
        color="slateblue",
        linewidth=0.9,
        label="Regime-Trend Strategy",
    )
    ax.set_title("Equity Curve Comparison")
    ax.set_ylabel("Equity (Indexed)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=False)
    format_date_axis(ax)
    save_fig(fig, output_dir / f"equity_curve_compare{suffix}.png")


def _rolling_cagr(log_returns: pd.Series, window: int) -> pd.Series:
    rolling_sum = log_returns.rolling(window).sum()
    annualized_log = rolling_sum * (252 / window)
    return np.exp(annualized_log) - 1.0


def _rolling_max_drawdown(equity: pd.Series, window: int) -> pd.Series:
    def _max_dd(values: np.ndarray) -> float:
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = value / peak - 1.0
            if drawdown < max_dd:
                max_dd = drawdown
        return max_dd

    return equity.rolling(window).apply(_max_dd, raw=True)


def _plot_rolling_cagr(frame: pd.DataFrame, output_dir: Path) -> None:
    windows = [(252, "1Y"), (756, "3Y")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (window, label) in zip(axes, windows):
        layered = _rolling_cagr(frame["strategy_log_return"], window)
        regime = _rolling_cagr(frame["regime_log_return"], window)
        trend_only = _rolling_cagr(frame["trend_only_log_return"], window)
        benchmark = _rolling_cagr(frame["log_return"], window)
        ax.plot(frame["date"], benchmark, color="black", linewidth=0.9, label="Benchmark")
        ax.plot(frame["date"], regime, color="#7E57C2", linewidth=0.9, label="Regime")
        ax.plot(frame["date"], trend_only, color="#26A69A", linewidth=0.9, label="Trend-Only")
        ax.plot(
            frame["date"],
            layered,
            color="slateblue",
            linewidth=0.9,
            label="Regime-Trend",
        )
        ax.set_title(f"Rolling CAGR ({label})")
        ax.set_ylabel("CAGR")
        ax.legend(loc="upper left", frameon=False)
    axes[-1].set_xlabel("Date")
    format_date_axis(axes[-1])
    save_fig(fig, output_dir / "rolling_cagr.png")


def _plot_rolling_drawdown(frame: pd.DataFrame, output_dir: Path) -> None:
    windows = [(252, "1Y"), (756, "3Y")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (window, label) in zip(axes, windows):
        layered = _rolling_max_drawdown(frame["strategy_equity"], window)
        regime = _rolling_max_drawdown(frame["regime_equity"], window)
        trend_only = _rolling_max_drawdown(frame["trend_only_equity"], window)
        benchmark = _rolling_max_drawdown(frame["benchmark_equity"], window)
        ax.plot(frame["date"], benchmark, color="black", linewidth=0.9, label="Benchmark")
        ax.plot(frame["date"], regime, color="#7E57C2", linewidth=0.9, label="Regime")
        ax.plot(frame["date"], trend_only, color="#26A69A", linewidth=0.9, label="Trend-Only")
        ax.plot(
            frame["date"],
            layered,
            color="slateblue",
            linewidth=0.9,
            label="Regime-Trend",
        )
        ax.set_title(f"Rolling Max Drawdown ({label})")
        ax.set_ylabel("Max Drawdown")
        ax.legend(loc="lower left", frameon=False)
    axes[-1].set_xlabel("Date")
    format_date_axis(axes[-1])
    save_fig(fig, output_dir / "rolling_drawdown.png")


def _rolling_alpha_beta(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int
) -> pd.DataFrame:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(index=strategy_returns.index, columns=["alpha_annual", "beta"])
    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    rolling_cov = strat.rolling(window).cov(bench)
    rolling_var = bench.rolling(window).var()
    beta = rolling_cov / rolling_var.replace(0, np.nan)
    mean_strat = strat.rolling(window).mean()
    mean_bench = bench.rolling(window).mean()
    alpha_daily = mean_strat - beta * mean_bench
    alpha_annual = alpha_daily * 252
    return pd.DataFrame({"alpha_annual": alpha_annual, "beta": beta})


def _plot_rolling_alpha_beta(frame: pd.DataFrame, output_dir: Path) -> None:
    window = 252
    layered = _rolling_alpha_beta(
        frame["strategy_return"], frame["benchmark_return"], window
    )
    regime = _rolling_alpha_beta(
        frame["regime_return"], frame["benchmark_return"], window
    )
    trend_only = _rolling_alpha_beta(
        frame["trend_only_return"], frame["benchmark_return"], window
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(
        frame["date"],
        layered["alpha_annual"],
        color="slateblue",
        linewidth=0.9,
        label="Regime-Trend",
    )
    axes[0].plot(
        frame["date"],
        regime["alpha_annual"],
        color="#7E57C2",
        linewidth=0.9,
        label="Regime",
    )
    axes[0].plot(
        frame["date"],
        trend_only["alpha_annual"],
        color="#26A69A",
        linewidth=0.9,
        label="Trend-Only",
    )
    axes[0].axhline(0.0, color="#616161", linewidth=0.8)
    axes[0].set_title("Rolling Alpha (1Y, Annualized)")
    axes[0].set_ylabel("Alpha")
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].plot(
        frame["date"],
        layered["beta"],
        color="slateblue",
        linewidth=0.9,
        label="Regime-Trend",
    )
    axes[1].plot(
        frame["date"],
        regime["beta"],
        color="#7E57C2",
        linewidth=0.9,
        label="Regime",
    )
    axes[1].plot(
        frame["date"],
        trend_only["beta"],
        color="#26A69A",
        linewidth=0.9,
        label="Trend-Only",
    )
    axes[1].axhline(1.0, color="#616161", linewidth=0.8)
    axes[1].set_title("Rolling Beta (1Y)")
    axes[1].set_ylabel("Beta")
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].set_xlabel("Date")

    format_date_axis(axes[-1])
    save_fig(fig, output_dir / "rolling_alpha_beta.png")


def _plot_turnover_hist(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frame["turnover"], bins=40, color="slateblue", alpha=0.8)
    ax.set_title("Turnover Distribution")
    ax.set_xlabel("Daily Turnover (Abs Exposure Change)")
    ax.set_ylabel("Count")
    save_fig(fig, output_dir / "turnover_hist.png")


def _plot_cost_sensitivity(data_dir: Path, output_dir: Path) -> None:
    path = data_dir / "cost_sensitivity.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        data["cost_bps"],
        data["net_annual_return"],
        color="slateblue",
        marker="o",
    )
    axes[0].set_title("Net Annual Return vs Cost")
    axes[0].set_xlabel("Cost (bps per turnover)")
    axes[0].set_ylabel("Net Annual Return")

    axes[1].plot(
        data["cost_bps"],
        data["net_sharpe"],
        color="#7E57C2",
        marker="o",
    )
    axes[1].set_title("Net Sharpe vs Cost")
    axes[1].set_xlabel("Cost (bps per turnover)")
    axes[1].set_ylabel("Net Sharpe")

    save_fig(fig, output_dir / "cost_sensitivity.png")


def _plot_exposure_overlay(
    frame: pd.DataFrame, output_dir: Path, suffix: str
) -> None:
    fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(
        3, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax_top.step(
        frame["date"],
        frame["exposure"],
        where="post",
        color="slateblue",
        linewidth=1.1,
        label="Exposure",
    )
    ax_top.set_title("Exposure with Trend + Regime Strip")
    ax_top.set_ylabel("Exposure")
    ax_top.legend(loc="upper left", frameon=False)

    _draw_strip(
        ax=ax_mid,
        dates=pd.to_datetime(frame["date"]),
        labels=frame["trend_state"].fillna("neutral").tolist(),
        colors={"strong_up": "#81C784", "neutral": "#BDBDBD", "strong_down": "#E57373"},
        title="Trend State (strong up / neutral / strong down)",
    )

    _draw_strip(
        ax=ax_bottom,
        dates=pd.to_datetime(frame["date"]),
        labels=frame["regime"].fillna("unknown").tolist(),
        colors={"low": "#4CAF50", "mid": "#FFB74D", "high": "#E57373"},
        title="Volatility Regime (low / mid / high)",
    )

    ax_bottom.set_xlabel("Date")
    format_date_axis(ax_bottom)
    save_fig(fig, output_dir / f"exposure_overlay{suffix}.png")


def _draw_strip(
    *,
    ax: plt.Axes,
    dates: pd.Series,
    labels: list[str],
    colors: dict[str, str],
    title: str,
) -> None:
    start_idx = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start_idx]:
            label = labels[start_idx]
            color = colors.get(label, "#BDBDBD")
            ax.axvspan(
                dates.iloc[start_idx],
                dates.iloc[idx - 1],
                color=color,
                alpha=0.85,
                linewidth=0,
            )
            start_idx = idx
    ax.set_yticks([])
    ax.set_title(title)


def _last_year_slice(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    if dates.empty:
        return frame
    cutoff = dates.max() - pd.DateOffset(years=1)
    return frame.loc[dates >= cutoff].copy()


def _apply_regime_only(frame: pd.DataFrame) -> None:
    exposure_map = {"low": 1.0, "mid": 0.75, "high": 0.25}
    frame["regime_exposure"] = frame["regime"].map(exposure_map).fillna(0.0)
    frame["regime_log_return"] = frame["log_return"] * frame["regime_exposure"]
    frame["regime_return"] = np.exp(frame["regime_log_return"]) - 1.0
    frame["regime_equity"] = np.exp(frame["regime_log_return"].cumsum())
    frame["regime_peak"] = frame["regime_equity"].cummax()
    frame["regime_drawdown"] = frame["regime_equity"] / frame["regime_peak"] - 1.0


def _derive_trend_only_map(
    exposure_matrix: dict[str, dict[str, float]]
) -> dict[str, float]:
    return {
        state: float(
            np.mean([exposure_matrix[regime][state] for regime in exposure_matrix])
        )
        for state in TREND_STATES
    }


def _apply_trend_only(frame: pd.DataFrame, trend_only_map: dict[str, float]) -> None:
    frame["trend_only_exposure"] = frame["trend_state"].map(trend_only_map).fillna(0.0)
    frame["trend_only_log_return"] = (
        frame["log_return"] * frame["trend_only_exposure"]
    )
    frame["trend_only_return"] = np.exp(frame["trend_only_log_return"]) - 1.0
    frame["trend_only_equity"] = np.exp(frame["trend_only_log_return"].cumsum())
    frame["trend_only_peak"] = frame["trend_only_equity"].cummax()
    frame["trend_only_drawdown"] = (
        frame["trend_only_equity"] / frame["trend_only_peak"] - 1.0
    )


def _write_state_performance(frame: pd.DataFrame, data_dir: Path) -> None:
    perf = (
        frame.groupby(["regime", "trend_state"], as_index=False)
        .agg(
            avg_log_return=("log_return", "mean"),
            avg_strategy_log_return=("strategy_log_return", "mean"),
            avg_exposure=("exposure", "mean"),
            obs=("regime", "size"),
        )
        .sort_values(["regime", "trend_state"])
    )
    perf.to_csv(data_dir / "state_performance.csv", index=False)


def _write_exposure_stats(frame: pd.DataFrame, data_dir: Path) -> None:
    stats = frame["exposure"].describe().to_frame(name="value")
    stats.to_csv(data_dir / "exposure_stats.csv")


def _write_turnover_stats(frame: pd.DataFrame, data_dir: Path) -> None:
    stats = frame["turnover"].describe().to_frame(name="value")
    stats.to_csv(data_dir / "turnover_stats.csv")


def _write_cost_sensitivity(frame: pd.DataFrame, data_dir: Path) -> None:
    cost_bps_list = [0, 1, 3, 5, 10]
    rows = []
    for cost_bps in cost_bps_list:
        cost_rate = frame["turnover"] * (cost_bps / 10000.0)
        net_log_return = frame["strategy_log_return"] - cost_rate
        stats_net = _compute_stats(net_log_return)
        rows.append(
            {
                "cost_bps": cost_bps,
                "net_annual_return": stats_net.annual_return,
                "net_annual_vol": stats_net.annual_vol,
                "net_sharpe": stats_net.sharpe,
                "net_max_drawdown": stats_net.max_drawdown,
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "cost_sensitivity.csv", index=False)


def _write_regime_map_comparison(frame: pd.DataFrame, data_dir: Path) -> None:
    rows = []
    for name, mapping in REGIME_EXPOSURE_MAPS.items():
        exposure = frame["regime"].map(mapping).fillna(0.0)
        strat_ret = frame["log_return"] * exposure
        stats = _compute_stats(strat_ret)
        rows.append(
            {
                "map_name": name,
                "annual_return": stats.annual_return,
                "annual_vol": stats.annual_vol,
                "sharpe": stats.sharpe,
                "max_drawdown": stats.max_drawdown,
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "regime_map_comparison.csv", index=False)
