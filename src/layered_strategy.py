from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TREND_STATES = ("strong_up", "neutral", "strong_down")

DEFAULT_EXPOSURE_MATRIX: dict[str, dict[str, float]] = {
    "low": {"strong_up": 1.1, "neutral": 0.8, "strong_down": 0.5},
    "mid": {"strong_up": 0.9, "neutral": 0.6, "strong_down": 0.3},
    "high": {"strong_up": 0.7, "neutral": 0.4, "strong_down": 0.1},
}


@dataclass
class LayeredStrategyResults:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_layered_frame(
    data: pd.DataFrame,
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
    min_exposure: float | None = 0.0,
    max_exposure: float | None = None,
    rebalance: str = "monthly",
    transition_window: int = 0,
    transition_multiplier: float = 1.0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    frame = data.copy()
    frame = frame.dropna(subset=["date", "log_return", "regime", "spx_adj_close"])
    frame = frame.sort_values("date")

    trend_score = _compute_trend_score(
        frame["spx_adj_close"], trend_window, trend_z_window
    ).fillna(0.0)
    frame["trend_score"] = trend_score
    frame["trend_state"] = _bucket_trend_state(trend_score, trend_threshold)
    frame["trend_state"] = _confirm_states(frame["trend_state"], state_confirm)

    if sizing_mode not in {"matrix", "continuous"}:
        raise ValueError("sizing_mode must be 'matrix' or 'continuous'")

    if sizing_mode == "continuous":
        vol_score = _compute_vol_score(
            frame["cond_vol"], vol_z_window
        ).fillna(0.0)
        frame["vol_score"] = vol_score
        frame["target_exposure"] = _continuous_exposure(
            trend_score=trend_score,
            vol_score=vol_score,
            base_exposure=base_exposure,
            trend_coef=trend_coef,
            vol_coef=vol_coef,
        )
    else:
        matrix = exposure_matrix or DEFAULT_EXPOSURE_MATRIX
        frame["target_exposure"] = _matrix_exposure(
            frame["regime"], frame["trend_state"], matrix
        )

    if transition_window > 0:
        recent_change = _recent_regime_change(frame["regime"], transition_window)
        frame.loc[recent_change, "target_exposure"] = (
            frame.loc[recent_change, "target_exposure"] * transition_multiplier
        )

    if min_exposure is not None:
        frame["target_exposure"] = frame["target_exposure"].clip(lower=min_exposure)
    if max_exposure is not None:
        frame["target_exposure"] = frame["target_exposure"].clip(upper=max_exposure)

    frame["exposure"] = _apply_rebalance(
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


def run_layered_strategy_backtest(
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
) -> LayeredStrategyResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)

    frame = build_layered_frame(
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

    stats = _compute_stats(frame["strategy_log_return"])
    stats_net = _compute_stats(frame["strategy_log_return_net"])
    benchmark_stats = _compute_stats(frame["log_return"])
    _apply_regime_only(frame)
    avg_turnover = float(frame["turnover"].mean())
    annual_turnover = avg_turnover * 252
    annual_cost_drag = annual_turnover * (cost_bps / 10000.0)
    avg_turnover = float(frame["turnover"].mean())
    annual_turnover = avg_turnover * 252
    annual_cost_drag = annual_turnover * (cost_bps / 10000.0)

    _write_summary(
        data_dir / "summary.txt",
        stats=stats,
        stats_net=stats_net,
        benchmark_stats=benchmark_stats,
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
        ]
    ].to_csv(data_dir / "layered_strategy_equity.csv", index=False)

    _write_state_performance(frame, data_dir)
    _write_exposure_stats(frame, data_dir)
    _write_turnover_stats(frame, data_dir)
    _write_cost_sensitivity(frame, data_dir)

    _plot_equity_curve(frame, plots_dir, suffix="")
    _plot_equity_curve(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_equity_curve(
        frame,
        plots_dir,
        suffix="_net",
        strategy_col="strategy_equity_net",
        label="Layered Strategy (Net)",
    )
    _plot_equity_curve(
        _last_year_slice(frame),
        plots_dir,
        suffix="_net_last_year",
        strategy_col="strategy_equity_net",
        label="Layered Strategy (Net)",
    )
    _plot_equity_curve_compare(frame, plots_dir, suffix="")
    _plot_equity_curve_compare(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_rolling_cagr(frame, plots_dir)
    _plot_rolling_drawdown(frame, plots_dir)
    _plot_exposure_overlay(frame, plots_dir, suffix="")
    _plot_exposure_overlay(_last_year_slice(frame), plots_dir, suffix="_last_year")
    _plot_turnover_hist(frame, plots_dir)
    _plot_cost_sensitivity(data_dir, plots_dir)

    return stats


def _compute_trend_score(
    price: pd.Series, trend_window: int, z_window: int
) -> pd.Series:
    log_price = np.log(price)
    trend_raw = (log_price - log_price.shift(trend_window)) / float(trend_window)
    mean = trend_raw.rolling(z_window, min_periods=z_window).mean()
    std = trend_raw.rolling(z_window, min_periods=z_window).std()
    return (trend_raw - mean) / std


def _bucket_trend_state(trend_score: pd.Series, threshold: float) -> pd.Series:
    states = pd.Series("neutral", index=trend_score.index)
    states = states.mask(trend_score >= threshold, "strong_up")
    states = states.mask(trend_score <= -threshold, "strong_down")
    return states


def _confirm_states(states: pd.Series, confirm: int) -> pd.Series:
    if confirm <= 1:
        return states
    confirmed = []
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


def _matrix_exposure(
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


def _compute_vol_score(cond_vol: pd.Series, z_window: int) -> pd.Series:
    mean = cond_vol.rolling(z_window, min_periods=z_window).mean()
    std = cond_vol.rolling(z_window, min_periods=z_window).std()
    return (cond_vol - mean) / std


def _continuous_exposure(
    *,
    trend_score: pd.Series,
    vol_score: pd.Series,
    base_exposure: float,
    trend_coef: float,
    vol_coef: float,
) -> pd.Series:
    return base_exposure + trend_coef * trend_score - vol_coef * vol_score


def _recent_regime_change(regimes: pd.Series, window: int) -> pd.Series:
    changed = regimes != regimes.shift(1)
    return changed.rolling(window, min_periods=1).max().fillna(0).astype(bool)


def _apply_rebalance(
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


def _compute_stats(strategy_log_returns: pd.Series) -> LayeredStrategyResults:
    ann_return = strategy_log_returns.mean() * 252
    ann_vol = strategy_log_returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity = np.exp(strategy_log_returns.cumsum())
    rolling_peak = pd.Series(equity).cummax().values
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = drawdown.min()

    return LayeredStrategyResults(
        annual_return=float(ann_return),
        annual_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
    )


def _write_summary(
    path: Path,
    *,
    stats: LayeredStrategyResults,
    stats_net: LayeredStrategyResults,
    benchmark_stats: LayeredStrategyResults,
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
        "Layered strategy backtest:",
        f"  annual return: {stats.annual_return:.4f}",
        f"  annual vol: {stats.annual_vol:.4f}",
        f"  sharpe: {stats.sharpe:.4f}",
        f"  max drawdown: {stats.max_drawdown:.4f}",
        "",
        "Layered strategy (net of costs):",
        f"  annual return: {stats_net.annual_return:.4f}",
        f"  annual vol: {stats_net.annual_vol:.4f}",
        f"  sharpe: {stats_net.sharpe:.4f}",
        f"  max drawdown: {stats_net.max_drawdown:.4f}",
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
    label: str = "Layered Strategy",
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
    ax.set_title("Layered Strategy Equity Curve")
    ax.set_ylabel("Equity (Indexed)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"equity_curve{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_equity_curve_compare(
    frame: pd.DataFrame, output_dir: Path, suffix: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    benchmark = frame["benchmark_equity"]
    layered = frame["strategy_equity"]
    regime = frame["regime_equity"]
    if len(frame) > 0 and suffix.endswith("last_year"):
        benchmark = benchmark / benchmark.iloc[0]
        layered = layered / layered.iloc[0]
        regime = regime / regime.iloc[0]
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
        layered,
        color="slateblue",
        linewidth=0.9,
        label="Layered Strategy",
    )
    ax.set_title("Equity Curve Comparison")
    ax.set_ylabel("Equity (Indexed)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"equity_curve_compare{suffix}.png", dpi=150)
    plt.close(fig)


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
        benchmark = _rolling_cagr(frame["log_return"], window)
        ax.plot(frame["date"], benchmark, color="black", linewidth=0.9, label="Benchmark")
        ax.plot(frame["date"], regime, color="#7E57C2", linewidth=0.9, label="Regime")
        ax.plot(frame["date"], layered, color="slateblue", linewidth=0.9, label="Layered")
        ax.set_title(f"Rolling CAGR ({label})")
        ax.set_ylabel("CAGR")
        ax.legend(loc="upper left", frameon=False)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_cagr.png", dpi=150)
    plt.close(fig)


def _plot_rolling_drawdown(frame: pd.DataFrame, output_dir: Path) -> None:
    windows = [(252, "1Y"), (756, "3Y")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (window, label) in zip(axes, windows):
        layered = _rolling_max_drawdown(frame["strategy_equity"], window)
        regime = _rolling_max_drawdown(frame["regime_equity"], window)
        benchmark = _rolling_max_drawdown(frame["benchmark_equity"], window)
        ax.plot(frame["date"], benchmark, color="black", linewidth=0.9, label="Benchmark")
        ax.plot(frame["date"], regime, color="#7E57C2", linewidth=0.9, label="Regime")
        ax.plot(frame["date"], layered, color="slateblue", linewidth=0.9, label="Layered")
        ax.set_title(f"Rolling Max Drawdown ({label})")
        ax.set_ylabel("Max Drawdown")
        ax.legend(loc="lower left", frameon=False)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_drawdown.png", dpi=150)
    plt.close(fig)


def _plot_turnover_hist(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frame["turnover"], bins=40, color="slateblue", alpha=0.8)
    ax.set_title("Turnover Distribution")
    ax.set_xlabel("Daily Turnover (Abs Exposure Change)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "turnover_hist.png", dpi=150)
    plt.close(fig)


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

    fig.tight_layout()
    fig.savefig(output_dir / "cost_sensitivity.png", dpi=150)
    plt.close(fig)


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
    fig.tight_layout()
    fig.savefig(output_dir / f"exposure_overlay{suffix}.png", dpi=150)
    plt.close(fig)


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
    frame["regime_equity"] = np.exp(frame["regime_log_return"].cumsum())
    frame["regime_peak"] = frame["regime_equity"].cummax()
    frame["regime_drawdown"] = frame["regime_equity"] / frame["regime_peak"] - 1.0


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
