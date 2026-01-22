from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.regime_trend import build_regime_trend_frame



@dataclass
class AlphaBeta:
    alpha_annual: float
    beta: float


def _compute_alpha_beta(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> AlphaBeta:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return AlphaBeta(alpha_annual=float("nan"), beta=float("nan"))
    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
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
    return AlphaBeta(alpha_annual=float(alpha_annual), beta=float(beta))


def _compute_metrics(
    strategy_log_returns: pd.Series,
    benchmark_log_returns: pd.Series,
) -> dict[str, float]:
    ann_return = strategy_log_returns.mean() * 252
    ann_vol = strategy_log_returns.std() * math.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    equity = np.exp(strategy_log_returns.cumsum())
    rolling_peak = pd.Series(equity).cummax().values
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())
    terminal_equity = float(equity.iloc[-1]) if len(equity) else float("nan")

    strat_ret = np.exp(strategy_log_returns) - 1.0
    bench_ret = np.exp(benchmark_log_returns) - 1.0
    alpha_beta = _compute_alpha_beta(strat_ret, bench_ret)

    return {
        "annual_return": float(ann_return),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "terminal_equity": terminal_equity,
        "alpha_annual": alpha_beta.alpha_annual,
        "beta": alpha_beta.beta,
    }


def _assign_regimes(realized_vol: pd.Series, low_thresh: float, high_thresh: float) -> pd.Series:
    def _label(value: float) -> str:
        if value <= low_thresh:
            return "low"
        if value >= high_thresh:
            return "high"
        return "mid"

    return realized_vol.apply(_label)


def _simulate_paths(
    *,
    mean_daily: float,
    std_daily: float,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mean_daily + std_daily * rng.standard_normal(size=(n_paths, horizon_days))


def _summarize_metrics(rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = df.agg(
        ["mean", "median", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]
    ).rename(index={"<lambda_0>": "p05", "<lambda_1>": "p95"})
    return summary


def _plot_distribution(
    data: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color in [
        ("benchmark", "black"),
        ("regime", "#7E57C2"),
        ("regime_trend", "slateblue"),
    ]:
        subset = data.loc[data["strategy"] == label, metric]
        ax.hist(subset, bins=40, alpha=0.5, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_outperformance(
    data: pd.DataFrame, output_path: Path, metric: str
) -> None:
    pivot = data.pivot(index="path", columns="strategy", values=metric)
    layered_out = (pivot["regime_trend"] > pivot["benchmark"]).mean()
    regime_out = (pivot["regime"] > pivot["benchmark"]).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["regime_trend", "regime"],
        [layered_out, regime_out],
        color=["slateblue", "#7E57C2"],
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Outperformance Probability ({metric.replace('_', ' ')})")
    ax.set_ylabel("Share of Paths")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_gbm_forward_test(
    *,
    regime_csv: Path = PROJECT_ROOT
    / "reports"
    / "regime_analysis"
    / "data"
    / "regime_series.csv",
    output_dir: Path = PROJECT_ROOT / "reports" / "strategy_regime_trend_gbm",
    horizons_years: list[int] | None = None,
    n_paths: int = 500,
    seed: int = 42,
    train_start: str | None = None,
    train_end: str | None = None,
    trend_window: int = 21,
    trend_z_window: int = 126,
    trend_threshold: float = 0.15,
    state_confirm: int = 1,
    rebalance: str = "daily",
    exposure_matrix: dict[str, dict[str, float]] | None = None,
    realized_window: int | None = None,
) -> None:
    if horizons_years is None:
        horizons_years = [1, 3, 5]
    exposure_matrix = exposure_matrix or {
        "low": {"strong_up": 1.2, "neutral": 0.9, "strong_down": 0.6},
        "mid": {"strong_up": 1.0, "neutral": 0.7, "strong_down": 0.4},
        "high": {"strong_up": 0.8, "neutral": 0.5, "strong_down": 0.2},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(regime_csv, parse_dates=["date"])
    data = data.sort_values("date")
    if train_start:
        data = data.loc[data["date"] >= pd.Timestamp(train_start)]
    if train_end:
        data = data.loc[data["date"] <= pd.Timestamp(train_end)]

    mean_daily = data["log_return"].mean()
    std_daily = data["log_return"].std()
    last_price = float(data["spx_adj_close"].iloc[-1])

    if realized_window is None:
        realized_window = int(data["realized_window"].dropna().iloc[-1])
    realized_vol_train = (
        data["log_return"].rolling(realized_window).std() * math.sqrt(252) * 100
    )
    low_thresh = realized_vol_train.quantile(0.33)
    high_thresh = realized_vol_train.quantile(0.66)

    summary_rows = []

    for horizon in horizons_years:
        horizon_days = int(horizon * 252)
        paths = _simulate_paths(
            mean_daily=mean_daily,
            std_daily=std_daily,
            horizon_days=horizon_days,
            n_paths=n_paths,
            seed=seed + horizon,
        )
        path_metrics = []

        for path_idx, log_returns in enumerate(paths):
            price = last_price * np.exp(np.cumsum(log_returns))
            realized_vol = (
                pd.Series(log_returns)
                .rolling(realized_window, min_periods=realized_window)
                .std()
                * math.sqrt(252)
                * 100
            )
            realized_vol = realized_vol.bfill()
            regimes = _assign_regimes(realized_vol, low_thresh, high_thresh)

            dates = pd.date_range(
                start=data["date"].iloc[-1] + pd.Timedelta(days=1),
                periods=horizon_days,
                freq="B",
            )
            sim = pd.DataFrame(
                {
                    "date": dates,
                    "log_return": log_returns,
                    "spx_adj_close": price,
                    "regime": regimes.values,
                }
            )

            layered_frame = build_regime_trend_frame(
                sim,
                trend_window=trend_window,
                trend_z_window=trend_z_window,
                trend_threshold=trend_threshold,
                state_confirm=state_confirm,
                sizing_mode="matrix",
                exposure_matrix=exposure_matrix,
                rebalance=rebalance,
            )

            regime_exposure = regimes.map(
                {"low": 1.0, "mid": 0.75, "high": 0.25}
            ).fillna(0.0)
            regime_log_return = pd.Series(log_returns) * regime_exposure.values

            layered_metrics = _compute_metrics(
                layered_frame["strategy_log_return"], sim["log_return"]
            )
            regime_metrics = _compute_metrics(regime_log_return, sim["log_return"])
            bench_metrics = _compute_metrics(sim["log_return"], sim["log_return"])

            for label, metrics in [
                ("regime_trend", layered_metrics),
                ("regime", regime_metrics),
                ("benchmark", bench_metrics),
            ]:
                metrics_row = {
                    "horizon_years": horizon,
                    "path": path_idx,
                    "strategy": label,
                }
                metrics_row.update(metrics)
                path_metrics.append(metrics_row)

        metrics_df = pd.DataFrame(path_metrics)
        metrics_df.to_csv(
            data_dir / f"gbm_path_metrics_{horizon}y.csv", index=False
        )

        summary = (
            metrics_df.groupby("strategy")[metrics_df.columns.difference(["path", "strategy", "horizon_years"])]
            .apply(_summarize_metrics)
        )
        summary.to_csv(data_dir / f"gbm_summary_{horizon}y.csv")

        _plot_distribution(
            metrics_df,
            plots_dir / f"gbm_return_dist_{horizon}y.png",
            metric="annual_return",
            title=f"GBM Annual Return Distribution ({horizon}Y)",
        )
        _plot_distribution(
            metrics_df,
            plots_dir / f"gbm_drawdown_dist_{horizon}y.png",
            metric="max_drawdown",
            title=f"GBM Max Drawdown Distribution ({horizon}Y)",
        )
        _plot_distribution(
            metrics_df,
            plots_dir / f"gbm_alpha_dist_{horizon}y.png",
            metric="alpha_annual",
            title=f"GBM Alpha Distribution ({horizon}Y)",
        )
        _plot_outperformance(
            metrics_df,
            plots_dir / f"gbm_outperformance_{horizon}y.png",
            metric="annual_return",
        )

        summary_rows.append(
            {
                "horizon_years": horizon,
                "n_paths": n_paths,
                "mean_daily": mean_daily,
                "std_daily": std_daily,
                "low_thresh": low_thresh,
                "high_thresh": high_thresh,
            }
        )

    pd.DataFrame(summary_rows).to_csv(data_dir / "gbm_run_summary.csv", index=False)

    README_PATH = output_dir / "README.md"
    if not README_PATH.exists():
        README_PATH.write_text(
            "\n".join(
                [
                    "# GBM Forward Test (Regime-Trend Strategy)",
                    "",
                    "Run `python scripts/run_gbm_forward_test.py` to populate this report.",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    run_gbm_forward_test()
