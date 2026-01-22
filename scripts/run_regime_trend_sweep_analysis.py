from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotting import save_fig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "reports" / "strategy_regime_trend_sweep"


@dataclass
class SweepAnalysisConfig:
    input_dir: Path
    top_n: int


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "train_excess_return",
        "train_max_drawdown",
        "train_benchmark_drawdown",
        "train_sharpe",
        "train_sortino",
        "holdout_excess_return",
        "holdout_max_drawdown",
        "holdout_benchmark_drawdown",
        "holdout_sharpe",
        "holdout_sortino",
        "train_return_mean",
        "train_sharpe_mean",
        "train_sortino_mean",
        "train_max_drawdown_worst",
        "val_return_mean",
        "val_sharpe_mean",
        "val_sortino_mean",
        "val_max_drawdown_worst",
        "val_benchmark_max_drawdown_worst",
        "test_return_mean",
        "test_sharpe_mean",
        "test_sortino_mean",
        "test_max_drawdown_worst",
        "test_benchmark_max_drawdown_worst",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "train_pass_dd" in df.columns:
        df["train_pass_dd"] = _coerce_bool(df["train_pass_dd"])
    if "holdout_pass_dd" in df.columns:
        df["holdout_pass_dd"] = _coerce_bool(df["holdout_pass_dd"])
    if "val_pass_dd" in df.columns:
        df["val_pass_dd"] = _coerce_bool(df["val_pass_dd"])

    if "train_max_drawdown" in df.columns and "train_benchmark_drawdown" in df.columns:
        df["train_drawdown_gap"] = (
            df["train_max_drawdown"] - df["train_benchmark_drawdown"]
        )
    if (
        "holdout_max_drawdown" in df.columns
        and "holdout_benchmark_drawdown" in df.columns
    ):
        df["holdout_drawdown_gap"] = (
            df["holdout_max_drawdown"] - df["holdout_benchmark_drawdown"]
        )
    if (
        "val_max_drawdown_worst" in df.columns
        and "val_benchmark_max_drawdown_worst" in df.columns
    ):
        df["val_drawdown_gap"] = (
            df["val_max_drawdown_worst"]
            - df["val_benchmark_max_drawdown_worst"]
        )
    df["label"] = (
        "tw"
        + df["trend_window"].astype(str)
        + "_th"
        + df["trend_threshold"].astype(str)
        + "_reb"
        + df["rebalance"].astype(str)
        + "_sc"
        + df["state_confirm"].astype(str)
        + "_mx"
        + df["matrix_name"].astype(str)
    )
    return df


def _plot_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"monthly": "o", "weekly": "s"}
    for rebalance, marker in markers.items():
        subset = df[df["rebalance"] == rebalance]
        if "val_pass_dd" in subset.columns:
            colors = np.where(subset["val_pass_dd"], "#2E7D32", "#C62828")
            x = subset["val_sortino_mean"]
            y = subset["val_drawdown_gap"]
            xlabel = "Validation Sortino (net)"
            ylabel = "Validation Drawdown Gap (strategy - benchmark)"
        else:
            colors = np.where(subset["holdout_pass_dd"], "#2E7D32", "#C62828")
            x = subset["holdout_excess_return"]
            y = subset["holdout_drawdown_gap"]
            xlabel = "Holdout Excess Return (strategy - benchmark)"
            ylabel = "Holdout Drawdown Gap (strategy - benchmark)"

        ax.scatter(
            x,
            y,
            s=35,
            alpha=0.75,
            marker=marker,
            c=colors,
            label=f"{rebalance} rebalance",
            edgecolors="none",
        )

    ax.axvline(0.0, color="#616161", linewidth=0.8)
    ax.axhline(0.0, color="#616161", linewidth=0.8)
    ax.set_title("Sweep Results: Return vs Drawdown Gap")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", frameon=False)
    save_fig(fig, output_path)


def _plot_top_bars(df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if "val_sortino_mean" in df.columns:
        top = df.sort_values("val_sortino_mean", ascending=False).head(top_n)
        values = top["val_sortino_mean"]
        title = f"Top {top_n} Validation Sortino (net)"
        xlabel = "Validation Sortino"
    else:
        top = df.sort_values("holdout_excess_return", ascending=False).head(top_n)
        values = top["holdout_excess_return"]
        title = f"Top {top_n} Holdout Excess Return"
        xlabel = "Holdout Excess Return"
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top_n + 1)))
    ax.barh(top["label"], values, color="#5C6BC0")
    ax.axvline(0.0, color="#616161", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    save_fig(fig, output_path)


def _plot_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    if "val_pass_dd" in df.columns:
        subset = df[df["val_pass_dd"]].copy()
        values = "val_sortino_mean"
        title = "Mean Validation Sortino (Pass DD)"
    else:
        subset = df[df["train_pass_dd"]].copy()
        values = "holdout_excess_return"
        title = "Mean Holdout Excess Return (Train Pass DD)"
    pivot = (
        subset.pivot_table(
            index="trend_window",
            columns="trend_threshold",
            values=values,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    ax.set_xlabel("Trend Threshold")
    ax.set_ylabel("Trend Window")
    fig.colorbar(im, ax=ax, shrink=0.85)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    save_fig(fig, output_path)


def _write_summary(df: pd.DataFrame, output_path: Path) -> None:
    if "val_pass_dd" in df.columns:
        eligible = df[df["val_pass_dd"]]
        best = eligible.sort_values("val_sortino_mean", ascending=False).head(1)
    else:
        eligible = df[df["train_pass_dd"]]
        best = eligible.sort_values("holdout_excess_return", ascending=False).head(1)
    lines = [
        f"total_candidates: {len(df)}",
        f"pass_dd_candidates: {len(eligible)}",
    ]
    if not best.empty:
        row = best.iloc[0]
        if "val_sortino_mean" in row:
            lines.extend(
                [
                    "best_validation_candidate:",
                    f"  label: {row['label']}",
                    f"  val_sortino_mean: {row['val_sortino_mean']:.6f}",
                    f"  val_drawdown_gap: {row['val_drawdown_gap']:.6f}",
                    f"  val_sharpe_mean: {row['val_sharpe_mean']:.4f}",
                ]
            )
        else:
            lines.extend(
                [
                    "best_holdout_candidate:",
                    f"  label: {row['label']}",
                    f"  holdout_excess_return: {row['holdout_excess_return']:.6f}",
                    f"  holdout_drawdown_gap: {row['holdout_drawdown_gap']:.6f}",
                    f"  holdout_sharpe: {row['holdout_sharpe']:.4f}",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_sweep_analysis(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    top_n: int = 10,
) -> None:
    data_dir = input_dir / "data"
    plots_dir = input_dir / "plots"
    results_path = data_dir / "sweep_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing sweep results: {results_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(results_path)
    results = _prepare_results(results)
    if "val_sortino_mean" in results.columns:
        results = results.dropna(subset=["val_sortino_mean", "val_drawdown_gap"])
    else:
        results = results.dropna(subset=["holdout_excess_return", "holdout_drawdown_gap"])

    results.to_csv(data_dir / "results_with_gaps.csv", index=False)
    _write_summary(results, data_dir / "summary.txt")
    _plot_scatter(results, plots_dir / "sweep_scatter.png")
    _plot_top_bars(results, plots_dir / "top_candidates.png", top_n=top_n)
    _plot_heatmap(results, plots_dir / "heatmap_metric.png")

    print(f"Wrote sweep analysis outputs to {input_dir}")


def _parse_args() -> SweepAnalysisConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze regime-trend strategy sweep results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    return SweepAnalysisConfig(
        input_dir=args.input_dir, top_n=args.top_n
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg = _parse_args()
        run_sweep_analysis(input_dir=cfg.input_dir, top_n=cfg.top_n)
    else:
        run_sweep_analysis()
