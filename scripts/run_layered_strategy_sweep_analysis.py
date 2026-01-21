from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "reports" / "strategy_layered_sweep"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "analysis"


@dataclass
class SweepAnalysisConfig:
    input_dir: Path
    output_dir: Path
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
        "holdout_excess_return",
        "holdout_max_drawdown",
        "holdout_benchmark_drawdown",
        "holdout_sharpe",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["train_pass_dd"] = _coerce_bool(df["train_pass_dd"])
    df["holdout_pass_dd"] = _coerce_bool(df["holdout_pass_dd"])
    df["train_drawdown_gap"] = (
        df["train_max_drawdown"] - df["train_benchmark_drawdown"]
    )
    df["holdout_drawdown_gap"] = (
        df["holdout_max_drawdown"] - df["holdout_benchmark_drawdown"]
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
        colors = np.where(subset["holdout_pass_dd"], "#2E7D32", "#C62828")
        ax.scatter(
            subset["holdout_excess_return"],
            subset["holdout_drawdown_gap"],
            s=35,
            alpha=0.75,
            marker=marker,
            c=colors,
            label=f"{rebalance} rebalance",
            edgecolors="none",
        )

    ax.axvline(0.0, color="#616161", linewidth=0.8)
    ax.axhline(0.0, color="#616161", linewidth=0.8)
    ax.set_title("Holdout Excess Return vs Drawdown Gap")
    ax.set_xlabel("Holdout Excess Return (strategy - benchmark)")
    ax.set_ylabel("Holdout Drawdown Gap (strategy - benchmark)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_top_bars(df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = df.sort_values("holdout_excess_return", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top_n + 1)))
    ax.barh(top["label"], top["holdout_excess_return"], color="#5C6BC0")
    ax.axvline(0.0, color="#616161", linewidth=0.8)
    ax.set_title(f"Top {top_n} Holdout Excess Return")
    ax.set_xlabel("Holdout Excess Return")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    subset = df[df["train_pass_dd"]].copy()
    pivot = (
        subset.pivot_table(
            index="trend_window",
            columns="trend_threshold",
            values="holdout_excess_return",
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
    ax.set_title("Mean Holdout Excess Return (Train Pass DD)")
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, output_path: Path) -> None:
    eligible = df[df["train_pass_dd"]]
    best_holdout = eligible.sort_values(
        "holdout_excess_return", ascending=False
    ).head(1)
    lines = [
        f"total_candidates: {len(df)}",
        f"train_pass_dd_candidates: {len(eligible)}",
    ]
    if not best_holdout.empty:
        row = best_holdout.iloc[0]
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
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 10,
) -> None:
    results_path = input_dir / "sweep_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing sweep results: {results_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(results_path)
    results = _prepare_results(results)
    results = results.dropna(subset=["holdout_excess_return", "holdout_drawdown_gap"])

    results.to_csv(output_dir / "results_with_gaps.csv", index=False)
    _write_summary(results, output_dir / "summary.txt")
    _plot_scatter(results, output_dir / "holdout_scatter.png")
    _plot_top_bars(results, output_dir / "top_holdout_excess_return.png", top_n=top_n)
    _plot_heatmap(results, output_dir / "heatmap_holdout_excess_return.png")

    print(f"Wrote sweep analysis outputs to {output_dir}")


def _parse_args() -> SweepAnalysisConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze layered strategy sweep results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    return SweepAnalysisConfig(
        input_dir=args.input_dir, output_dir=args.output_dir, top_n=args.top_n
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg = _parse_args()
        run_sweep_analysis(
            input_dir=cfg.input_dir, output_dir=cfg.output_dir, top_n=cfg.top_n
        )
    else:
        run_sweep_analysis()
