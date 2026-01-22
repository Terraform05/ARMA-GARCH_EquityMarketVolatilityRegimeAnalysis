from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.regime_trend import (
    DEFAULT_EXPOSURE_MATRIX,
    build_regime_trend_frame,
)


MATRIX_LIBRARY: dict[str, dict[str, dict[str, float]]] = {
    "default": DEFAULT_EXPOSURE_MATRIX,
    "aggressive": {
        "low": {"strong_up": 1.2, "neutral": 0.9, "strong_down": 0.6},
        "mid": {"strong_up": 1.0, "neutral": 0.7, "strong_down": 0.4},
        "high": {"strong_up": 0.8, "neutral": 0.5, "strong_down": 0.2},
    },
    "conservative": {
        "low": {"strong_up": 1.0, "neutral": 0.7, "strong_down": 0.4},
        "mid": {"strong_up": 0.8, "neutral": 0.5, "strong_down": 0.2},
        "high": {"strong_up": 0.5, "neutral": 0.2, "strong_down": 0.0},
    },
}


@dataclass
class SweepConfig:
    regime_csv: Path
    output_dir: Path
    evaluation: str
    train_end: str
    holdout_start: str
    train_years: int
    val_years: int
    test_years: int
    step_years: int
    trend_windows: list[int]
    trend_thresholds: list[float]
    rebalance: list[str]
    state_confirms: list[int]
    matrix_set: list[str]
    sizing_mode: str
    trend_z_window: int
    vol_z_window: int
    base_exposure: float
    trend_coef: float
    vol_coef: float
    min_exposure: float
    max_exposure: float | None
    transition_window: int
    transition_multiplier: float
    cost_bps: float
    drawdown_cap: float
    top_n: int


@dataclass
class StrategyStats:
    annual_return: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _default_config() -> SweepConfig:
    return SweepConfig(
        regime_csv=PROJECT_ROOT
        / "reports"
        / "regime_analysis"
        / "data"
        / "regime_series.csv",
        output_dir=PROJECT_ROOT / "reports" / "strategy_regime_trend_sweep",
        evaluation="walk_forward",
        train_end="2018-12-31",
        holdout_start="2019-01-01",
        train_years=8,
        val_years=2,
        test_years=2,
        step_years=2,
        trend_windows=[21, 42, 63, 84, 126, 252],
        trend_thresholds=[0.15, 0.25, 0.35, 0.5],
        rebalance=["daily", "weekly", "monthly"],
        state_confirms=[1, 2],
        matrix_set=["default", "aggressive"],
        sizing_mode="matrix",
        trend_z_window=126,
        vol_z_window=252,
        base_exposure=0.6,
        trend_coef=0.25,
        vol_coef=0.25,
        min_exposure=0.0,
        max_exposure=None,
        transition_window=0,
        transition_multiplier=1.0,
        cost_bps=5.0,
        drawdown_cap=-0.20,
        top_n=10,
    )


def default_sweep_config() -> SweepConfig:
    return _default_config()


def _compute_stats(strategy_log_returns: pd.Series) -> StrategyStats:
    if strategy_log_returns.empty:
        return StrategyStats(
            annual_return=float("nan"),
            annual_vol=float("nan"),
            sharpe=float("nan"),
            sortino=float("nan"),
            max_drawdown=float("nan"),
        )
    ann_return = strategy_log_returns.mean() * 252
    ann_vol = strategy_log_returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    downside = strategy_log_returns[strategy_log_returns < 0.0]
    downside_vol = downside.std() * (252**0.5)
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    equity = np.exp(strategy_log_returns.cumsum())
    rolling_peak = pd.Series(equity).cummax().values
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = drawdown.min()

    return StrategyStats(
        annual_return=float(ann_return),
        annual_vol=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_drawdown),
    )


def _subset(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is None and end is None:
        return frame
    subset = frame
    if start is not None:
        subset = subset.loc[subset["date"] >= pd.Timestamp(start)]
    if end is not None:
        subset = subset.loc[subset["date"] <= pd.Timestamp(end)]
    return subset


def _generate_walk_forward_windows(
    dates: pd.Series,
    *,
    train_years: int,
    val_years: int,
    test_years: int,
    step_years: int,
) -> list[WalkForwardWindow]:
    start = dates.min()
    end = dates.max()
    windows: list[WalkForwardWindow] = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(years=val_years) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)

        if test_start > end:
            break
        if test_end > end:
            test_end = end

        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start = train_start + pd.DateOffset(years=step_years)

    return windows


def _aggregate_stats(stats: list[StrategyStats]) -> dict[str, float]:
    if not stats:
        return {
            "annual_return_mean": float("nan"),
            "annual_vol_mean": float("nan"),
            "sharpe_mean": float("nan"),
            "sortino_mean": float("nan"),
            "max_drawdown_worst": float("nan"),
        }
    return {
        "annual_return_mean": float(np.nanmean([s.annual_return for s in stats])),
        "annual_vol_mean": float(np.nanmean([s.annual_vol for s in stats])),
        "sharpe_mean": float(np.nanmean([s.sharpe for s in stats])),
        "sortino_mean": float(np.nanmean([s.sortino for s in stats])),
        "max_drawdown_worst": float(np.nanmin([s.max_drawdown for s in stats])),
    }


def _evaluate_config(
    frame: pd.DataFrame,
    train_end: str,
    holdout_start: str,
) -> dict[str, StrategyStats]:
    train = _subset(frame, None, train_end)
    holdout = _subset(frame, holdout_start, None)
    return {
        "train": _compute_stats(train["strategy_log_return_net"]),
        "train_benchmark": _compute_stats(train["log_return"]),
        "holdout": _compute_stats(holdout["strategy_log_return_net"]),
        "holdout_benchmark": _compute_stats(holdout["log_return"]),
    }


def _evaluate_walk_forward(
    frame: pd.DataFrame,
    windows: list[WalkForwardWindow],
) -> dict[str, list[StrategyStats]]:
    train_stats: list[StrategyStats] = []
    train_bench_stats: list[StrategyStats] = []
    val_stats: list[StrategyStats] = []
    val_bench_stats: list[StrategyStats] = []
    test_stats: list[StrategyStats] = []
    test_bench_stats: list[StrategyStats] = []

    for window in windows:
        train = frame.loc[
            (frame["date"] >= window.train_start) & (frame["date"] <= window.train_end)
        ]
        val = frame.loc[
            (frame["date"] >= window.val_start) & (frame["date"] <= window.val_end)
        ]
        test = frame.loc[
            (frame["date"] >= window.test_start) & (frame["date"] <= window.test_end)
        ]

        train_stats.append(_compute_stats(train["strategy_log_return_net"]))
        train_bench_stats.append(_compute_stats(train["log_return"]))
        val_stats.append(_compute_stats(val["strategy_log_return_net"]))
        val_bench_stats.append(_compute_stats(val["log_return"]))
        test_stats.append(_compute_stats(test["strategy_log_return_net"]))
        test_bench_stats.append(_compute_stats(test["log_return"]))

    return {
        "train": train_stats,
        "train_benchmark": train_bench_stats,
        "val": val_stats,
        "val_benchmark": val_bench_stats,
        "test": test_stats,
        "test_benchmark": test_bench_stats,
    }


def run_regime_trend_sweep(config: SweepConfig) -> None:
    data = pd.read_csv(config.regime_csv, parse_dates=["date"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = config.output_dir / "data"
    plots_dir = config.output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if config.evaluation not in {"walk_forward", "single_split"}:
        raise ValueError("evaluation must be 'walk_forward' or 'single_split'")

    windows = None
    if config.evaluation == "walk_forward":
        windows = _generate_walk_forward_windows(
            data["date"],
            train_years=config.train_years,
            val_years=config.val_years,
            test_years=config.test_years,
            step_years=config.step_years,
        )
        if not windows:
            raise ValueError("No walk-forward windows available for the dataset.")

    rows = []
    for (
        trend_window,
        trend_threshold,
        rebalance_mode,
        state_confirm,
        matrix_name,
    ) in product(
        config.trend_windows,
        config.trend_thresholds,
        config.rebalance,
        config.state_confirms,
        config.matrix_set,
    ):
        exposure_matrix = MATRIX_LIBRARY[matrix_name]
        frame = build_regime_trend_frame(
            data,
            trend_window=trend_window,
            trend_z_window=config.trend_z_window,
            trend_threshold=trend_threshold,
            state_confirm=state_confirm,
            sizing_mode=config.sizing_mode,
            exposure_matrix=exposure_matrix,
            base_exposure=config.base_exposure,
            trend_coef=config.trend_coef,
            vol_coef=config.vol_coef,
            vol_z_window=config.vol_z_window,
            min_exposure=config.min_exposure,
            max_exposure=config.max_exposure,
            rebalance=rebalance_mode,
            transition_window=config.transition_window,
            transition_multiplier=config.transition_multiplier,
            cost_bps=config.cost_bps,
        )

        if config.evaluation == "single_split":
            metrics = _evaluate_config(
                frame, train_end=config.train_end, holdout_start=config.holdout_start
            )
            train = metrics["train"]
            train_bench = metrics["train_benchmark"]
            holdout = metrics["holdout"]
            holdout_bench = metrics["holdout_benchmark"]

            train_pass_dd = (
                train.max_drawdown >= config.drawdown_cap
                if np.isfinite(train.max_drawdown)
                else False
            )
            holdout_pass_dd = (
                holdout.max_drawdown >= config.drawdown_cap
                if np.isfinite(holdout.max_drawdown)
                else False
            )
            row = {
                "trend_window": trend_window,
                "trend_threshold": trend_threshold,
                "rebalance": rebalance_mode,
                "state_confirm": state_confirm,
                "matrix_name": matrix_name,
                "sizing_mode": config.sizing_mode,
                "cost_bps": config.cost_bps,
                "train_excess_return": train.annual_return
                - train_bench.annual_return,
                "train_max_drawdown": train.max_drawdown,
                "train_pass_dd": train_pass_dd,
                "train_sharpe": train.sharpe,
                "train_sortino": train.sortino,
                "holdout_excess_return": holdout.annual_return
                - holdout_bench.annual_return,
                "holdout_max_drawdown": holdout.max_drawdown,
                "holdout_pass_dd": holdout_pass_dd,
                "holdout_sharpe": holdout.sharpe,
                "holdout_sortino": holdout.sortino,
            }
        else:
            if windows is None:
                raise ValueError("Walk-forward windows not initialized.")
            metrics_walk = _evaluate_walk_forward(frame, windows)
            train_stats = metrics_walk["train"]
            train_bench_stats = metrics_walk["train_benchmark"]
            val_stats = metrics_walk["val"]
            val_bench_stats = metrics_walk["val_benchmark"]
            test_stats = metrics_walk["test"]
            test_bench_stats = metrics_walk["test_benchmark"]

            train_agg = _aggregate_stats(train_stats)
            train_bench_agg = _aggregate_stats(train_bench_stats)
            val_agg = _aggregate_stats(val_stats)
            val_bench_agg = _aggregate_stats(val_bench_stats)
            test_agg = _aggregate_stats(test_stats)
            test_bench_agg = _aggregate_stats(test_bench_stats)

            val_pass_dd = (
                val_agg["max_drawdown_worst"] >= config.drawdown_cap
                if np.isfinite(val_agg["max_drawdown_worst"])
                else False
            )

            row = {
                "trend_window": trend_window,
                "trend_threshold": trend_threshold,
                "rebalance": rebalance_mode,
                "state_confirm": state_confirm,
                "matrix_name": matrix_name,
                "sizing_mode": config.sizing_mode,
                "cost_bps": config.cost_bps,
                "fold_count": len(train_stats),
                "train_return_mean": train_agg["annual_return_mean"],
                "train_sharpe_mean": train_agg["sharpe_mean"],
                "train_sortino_mean": train_agg["sortino_mean"],
                "train_max_drawdown_worst": train_agg["max_drawdown_worst"],
                "train_benchmark_return_mean": train_bench_agg["annual_return_mean"],
                "val_return_mean": val_agg["annual_return_mean"],
                "val_sharpe_mean": val_agg["sharpe_mean"],
                "val_sortino_mean": val_agg["sortino_mean"],
                "val_max_drawdown_worst": val_agg["max_drawdown_worst"],
                "val_benchmark_return_mean": val_bench_agg["annual_return_mean"],
                "val_benchmark_max_drawdown_worst": val_bench_agg["max_drawdown_worst"],
                "val_pass_dd": val_pass_dd,
                "test_return_mean": test_agg["annual_return_mean"],
                "test_sharpe_mean": test_agg["sharpe_mean"],
                "test_sortino_mean": test_agg["sortino_mean"],
                "test_max_drawdown_worst": test_agg["max_drawdown_worst"],
                "test_benchmark_return_mean": test_bench_agg["annual_return_mean"],
                "test_benchmark_max_drawdown_worst": test_bench_agg["max_drawdown_worst"],
            }

        rows.append(row)

    results = pd.DataFrame(rows)
    if config.evaluation == "single_split":
        results = results.sort_values(
            ["holdout_pass_dd", "holdout_sortino"],
            ascending=[False, False],
        )
    else:
        results = results.sort_values(
            ["val_pass_dd", "val_sortino_mean"],
            ascending=[False, False],
        )
    results.to_csv(data_dir / "sweep_results.csv", index=False)

    eligible = (
        results.loc[results["holdout_pass_dd"]].copy()
        if config.evaluation == "single_split"
        else results.loc[results["val_pass_dd"]].copy()
    )
    top_candidates = eligible.head(config.top_n)
    top_candidates.to_csv(data_dir / "top_candidates.csv", index=False)

    summary = {
        "evaluation": config.evaluation,
        "train_end": config.train_end,
        "holdout_start": config.holdout_start,
        "train_years": config.train_years,
        "val_years": config.val_years,
        "test_years": config.test_years,
        "step_years": config.step_years,
        "trend_windows": config.trend_windows,
        "trend_thresholds": config.trend_thresholds,
        "rebalance": config.rebalance,
        "state_confirms": config.state_confirms,
        "matrix_set": config.matrix_set,
        "sizing_mode": config.sizing_mode,
        "trend_z_window": config.trend_z_window,
        "vol_z_window": config.vol_z_window,
        "base_exposure": config.base_exposure,
        "trend_coef": config.trend_coef,
        "vol_coef": config.vol_coef,
        "min_exposure": config.min_exposure,
        "max_exposure": config.max_exposure,
        "transition_window": config.transition_window,
        "transition_multiplier": config.transition_multiplier,
        "cost_bps": config.cost_bps,
        "drawdown_cap": config.drawdown_cap,
        "top_n": config.top_n,
        "eligible_count": int(eligible.shape[0]),
        "total_count": int(results.shape[0]),
    }
    (data_dir / "sweep_summary.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in summary.items()]),
        encoding="utf-8",
    )

    print(f"Wrote sweep outputs to {config.output_dir}")


def run_regime_trend_sweep_job(config: SweepConfig | None = None) -> None:
    cfg = config or _default_config()
    run_regime_trend_sweep(cfg)


def _parse_args() -> SweepConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep regime-trend strategy configs and rank by excess return."
    )
    parser.add_argument(
        "--regime-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "regime_analysis" / "data" / "regime_series.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "strategy_regime_trend_sweep",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default="walk_forward",
        choices=["walk_forward", "single_split"],
    )
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--holdout-start", type=str, default=None)
    parser.add_argument("--train-years", type=int, default=8)
    parser.add_argument("--val-years", type=int, default=2)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--step-years", type=int, default=2)
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[63, 126, 252])
    parser.add_argument("--trend-thresholds", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--rebalance", type=str, nargs="+", default=["monthly", "weekly"])
    parser.add_argument("--state-confirms", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--matrix-set",
        type=str,
        nargs="+",
        default=["default"],
        choices=sorted(MATRIX_LIBRARY.keys()),
    )
    parser.add_argument("--sizing-mode", type=str, default="matrix")
    parser.add_argument("--trend-z-window", type=int, default=252)
    parser.add_argument("--vol-z-window", type=int, default=252)
    parser.add_argument("--base-exposure", type=float, default=0.6)
    parser.add_argument("--trend-coef", type=float, default=0.25)
    parser.add_argument("--vol-coef", type=float, default=0.25)
    parser.add_argument("--min-exposure", type=float, default=0.0)
    parser.add_argument("--max-exposure", type=float, default=None)
    parser.add_argument("--transition-window", type=int, default=0)
    parser.add_argument("--transition-multiplier", type=float, default=1.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--drawdown-cap", type=float, default=-0.20)
    parser.add_argument("--top-n", type=int, default=10)

    args = parser.parse_args()
    holdout_start = args.holdout_start
    if holdout_start is None:
        holdout_start = (
            pd.Timestamp(args.train_end) + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    return SweepConfig(
        regime_csv=args.regime_csv,
        output_dir=args.output_dir,
        evaluation=args.evaluation,
        train_end=args.train_end,
        holdout_start=holdout_start,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        step_years=args.step_years,
        trend_windows=args.trend_windows,
        trend_thresholds=args.trend_thresholds,
        rebalance=args.rebalance,
        state_confirms=args.state_confirms,
        matrix_set=args.matrix_set,
        sizing_mode=args.sizing_mode,
        trend_z_window=args.trend_z_window,
        vol_z_window=args.vol_z_window,
        base_exposure=args.base_exposure,
        trend_coef=args.trend_coef,
        vol_coef=args.vol_coef,
        min_exposure=args.min_exposure,
        max_exposure=args.max_exposure,
        transition_window=args.transition_window,
        transition_multiplier=args.transition_multiplier,
        cost_bps=args.cost_bps,
        drawdown_cap=args.drawdown_cap,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_regime_trend_sweep(_parse_args())
    else:
        run_regime_trend_sweep_job()
