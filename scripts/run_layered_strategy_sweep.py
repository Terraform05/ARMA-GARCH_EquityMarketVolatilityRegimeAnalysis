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

from src.layered_strategy import DEFAULT_EXPOSURE_MATRIX, build_layered_frame


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
    train_end: str
    holdout_start: str
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
    top_n: int


@dataclass
class StrategyStats:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


def _default_config() -> SweepConfig:
    return SweepConfig(
        regime_csv=PROJECT_ROOT
        / "reports"
        / "regime_analysis"
        / "data"
        / "regime_series.csv",
        output_dir=PROJECT_ROOT / "reports" / "strategy_layered_sweep",
        train_end="2018-12-31",
        holdout_start="2019-01-01",
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
        top_n=10,
    )


def _compute_stats(strategy_log_returns: pd.Series) -> StrategyStats:
    if strategy_log_returns.empty:
        return StrategyStats(
            annual_return=float("nan"),
            annual_vol=float("nan"),
            sharpe=float("nan"),
            max_drawdown=float("nan"),
        )
    ann_return = strategy_log_returns.mean() * 252
    ann_vol = strategy_log_returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity = np.exp(strategy_log_returns.cumsum())
    rolling_peak = pd.Series(equity).cummax().values
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = drawdown.min()

    return StrategyStats(
        annual_return=float(ann_return),
        annual_vol=float(ann_vol),
        sharpe=float(sharpe),
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


def _evaluate_config(
    frame: pd.DataFrame,
    train_end: str,
    holdout_start: str,
) -> dict[str, StrategyStats]:
    train = _subset(frame, None, train_end)
    holdout = _subset(frame, holdout_start, None)
    return {
        "train": _compute_stats(train["strategy_log_return"]),
        "train_benchmark": _compute_stats(train["log_return"]),
        "holdout": _compute_stats(holdout["strategy_log_return"]),
        "holdout_benchmark": _compute_stats(holdout["log_return"]),
    }


def run_layered_strategy_sweep(config: SweepConfig) -> None:
    data = pd.read_csv(config.regime_csv, parse_dates=["date"])
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        frame = build_layered_frame(
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
        )

        metrics = _evaluate_config(
            frame, train_end=config.train_end, holdout_start=config.holdout_start
        )
        train = metrics["train"]
        train_bench = metrics["train_benchmark"]
        holdout = metrics["holdout"]
        holdout_bench = metrics["holdout_benchmark"]

        train_pass_dd = (
            train.max_drawdown >= train_bench.max_drawdown
            if np.isfinite(train.max_drawdown)
            and np.isfinite(train_bench.max_drawdown)
            else False
        )
        holdout_pass_dd = (
            holdout.max_drawdown >= holdout_bench.max_drawdown
            if np.isfinite(holdout.max_drawdown)
            and np.isfinite(holdout_bench.max_drawdown)
            else False
        )

        rows.append(
            {
                "trend_window": trend_window,
                "trend_threshold": trend_threshold,
                "rebalance": rebalance_mode,
                "state_confirm": state_confirm,
                "matrix_name": matrix_name,
                "sizing_mode": config.sizing_mode,
                "train_excess_return": train.annual_return
                - train_bench.annual_return,
                "train_max_drawdown": train.max_drawdown,
                "train_benchmark_drawdown": train_bench.max_drawdown,
                "train_pass_dd": train_pass_dd,
                "train_sharpe": train.sharpe,
                "holdout_excess_return": holdout.annual_return
                - holdout_bench.annual_return,
                "holdout_max_drawdown": holdout.max_drawdown,
                "holdout_benchmark_drawdown": holdout_bench.max_drawdown,
                "holdout_pass_dd": holdout_pass_dd,
                "holdout_sharpe": holdout.sharpe,
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["train_pass_dd", "train_excess_return"],
        ascending=[False, False],
    )
    results.to_csv(config.output_dir / "sweep_results.csv", index=False)

    eligible = results.loc[results["train_pass_dd"]].copy()
    top_candidates = eligible.head(config.top_n)
    top_candidates.to_csv(config.output_dir / "top_candidates.csv", index=False)

    summary = {
        "train_end": config.train_end,
        "holdout_start": config.holdout_start,
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
        "top_n": config.top_n,
        "eligible_count": int(eligible.shape[0]),
        "total_count": int(results.shape[0]),
    }
    (config.output_dir / "sweep_summary.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in summary.items()]),
        encoding="utf-8",
    )

    print(f"Wrote sweep outputs to {config.output_dir}")


def run_layered_strategy_sweep_job(config: SweepConfig | None = None) -> None:
    cfg = config or _default_config()
    run_layered_strategy_sweep(cfg)


def _parse_args() -> SweepConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep layered strategy configs and rank by excess return."
    )
    parser.add_argument(
        "--regime-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "regime_analysis" / "data" / "regime_series.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "strategy_layered_sweep",
    )
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--holdout-start", type=str, default=None)
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
        train_end=args.train_end,
        holdout_start=holdout_start,
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
        top_n=args.top_n,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_layered_strategy_sweep(_parse_args())
    else:
        run_layered_strategy_sweep_job()
