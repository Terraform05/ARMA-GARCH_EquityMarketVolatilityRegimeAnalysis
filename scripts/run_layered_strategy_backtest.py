from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layered_strategy import DEFAULT_EXPOSURE_MATRIX, run_layered_strategy_backtest

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "strategy_layered_sweep"
    / "analysis"
    / "results_with_gaps.csv"
)
FALLBACK_RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "strategy_layered_sweep" / "sweep_results.csv"
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
class SweepPick:
    trend_window: int
    trend_threshold: float
    rebalance: str
    state_confirm: int
    matrix_name: str
    sizing_mode: str


def _coerce_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _build_label(row: pd.Series) -> str:
    return (
        f"tw{row['trend_window']}_"
        f"th{row['trend_threshold']}_"
        f"reb{row['rebalance']}_"
        f"sc{row['state_confirm']}_"
        f"mx{row['matrix_name']}"
    )


def _pick_best_from_results(results_path: Path, label: str | None) -> SweepPick:
    results = pd.read_csv(results_path)
    if "label" not in results.columns:
        results["label"] = results.apply(_build_label, axis=1)

    if "train_pass_dd" in results.columns:
        results["train_pass_dd"] = results["train_pass_dd"].apply(_coerce_bool)
    if "holdout_pass_dd" in results.columns:
        results["holdout_pass_dd"] = results["holdout_pass_dd"].apply(_coerce_bool)

    if label:
        match = results.loc[results["label"] == label]
        if match.empty:
            raise ValueError(f"Label not found in results: {label}")
        row = match.iloc[0]
    else:
        eligible = results
        if "train_pass_dd" in results.columns:
            eligible = eligible[eligible["train_pass_dd"]]
        if "holdout_pass_dd" in results.columns:
            eligible = eligible[eligible["holdout_pass_dd"]]
        if eligible.empty:
            eligible = results
        row = eligible.sort_values(
            "holdout_excess_return", ascending=False
        ).iloc[0]

    return SweepPick(
        trend_window=int(row["trend_window"]),
        trend_threshold=float(row["trend_threshold"]),
        rebalance=str(row["rebalance"]),
        state_confirm=int(row["state_confirm"]),
        matrix_name=str(row["matrix_name"]),
        sizing_mode=str(row["sizing_mode"]),
    )


def run_layered_strategy_backtest_job(
    regime_csv: str | Path = PROJECT_ROOT
    / "reports"
    / "regime_analysis"
    / "data"
    / "regime_series.csv",
    output_dir: str | Path = PROJECT_ROOT / "reports" / "strategy_layered",
    cost_bps: float = 0.0,
) -> None:
    data = pd.read_csv(regime_csv, parse_dates=["date"])
    run_layered_strategy_backtest(data, Path(output_dir), cost_bps=cost_bps)
    print(f"Wrote layered strategy outputs to {output_dir}")


def run_layered_strategy_from_sweep(
    regime_csv: Path,
    output_dir: Path,
    results_path: Path,
    label: str | None = None,
    cost_bps: float = 0.0,
) -> None:
    data = pd.read_csv(regime_csv, parse_dates=["date"])
    pick = _pick_best_from_results(results_path, label)
    exposure_matrix = MATRIX_LIBRARY.get(pick.matrix_name, DEFAULT_EXPOSURE_MATRIX)

    run_layered_strategy_backtest(
        data,
        output_dir,
        trend_window=pick.trend_window,
        trend_threshold=pick.trend_threshold,
        rebalance=pick.rebalance,
        state_confirm=pick.state_confirm,
        sizing_mode=pick.sizing_mode,
        exposure_matrix=exposure_matrix,
        trend_z_window=126,
        vol_z_window=252,
        base_exposure=0.6,
        trend_coef=0.25,
        vol_coef=0.25,
        min_exposure=0.0,
        max_exposure=None,
        transition_window=0,
        transition_multiplier=1.0,
        cost_bps=cost_bps,
    )
    label_suffix = f" ({label})" if label else ""
    print(f"Wrote layered strategy outputs to {output_dir}{label_suffix}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the layered strategy backtest.")
    parser.add_argument(
        "--from-sweep",
        action="store_true",
        help="Use the best sweep candidate from the results file.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Path to sweep results CSV (defaults to analysis results if present).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label from sweep results to run.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Transaction cost in bps per unit of turnover.",
    )
    parser.add_argument(
        "--regime-csv",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "regime_analysis"
        / "data"
        / "regime_series.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "strategy_layered",
    )
    args = parser.parse_args()

    if args.from_sweep:
        results_path = args.results_path or DEFAULT_RESULTS_PATH
        if not results_path.exists():
            results_path = FALLBACK_RESULTS_PATH
        run_layered_strategy_from_sweep(
            regime_csv=args.regime_csv,
            output_dir=args.output_dir,
            results_path=results_path,
            label=args.label,
            cost_bps=args.cost_bps,
        )
    else:
        run_layered_strategy_backtest_job(cost_bps=args.cost_bps)
