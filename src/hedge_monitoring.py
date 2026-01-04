from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class HedgeMonitoringResults:
    ratio_low: float
    ratio_high: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_hedge_monitoring(
    data: pd.DataFrame,
    output_dir: Path,
    ratio_quantiles: tuple[float, float] = (0.2, 0.8),
) -> HedgeMonitoringResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)

    frame = data.copy()
    frame = frame.dropna(subset=["vix_close", "realized_vol", "regime"])
    frame["hedge_gap"] = frame["vix_close"] - frame["realized_vol"]
    frame["hedge_ratio"] = frame["vix_close"] / frame["realized_vol"]

    ratio_low = frame["hedge_ratio"].quantile(ratio_quantiles[0])
    ratio_high = frame["hedge_ratio"].quantile(ratio_quantiles[1])

    frame["hedge_signal"] = "neutral"
    frame.loc[frame["hedge_ratio"] <= ratio_low, "hedge_signal"] = "cheap"
    frame.loc[frame["hedge_ratio"] >= ratio_high, "hedge_signal"] = "expensive"

    _plot_hedge_ratio(frame, plots_dir, ratio_low, ratio_high)
    _plot_vix_vs_realized(frame, plots_dir)

    summary_path = data_dir / "summary.txt"
    signal_counts = frame["hedge_signal"].value_counts(normalize=True)
    by_regime = (
        frame.groupby("regime")["hedge_signal"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )
    segment_id = frame["hedge_signal"].ne(frame["hedge_signal"].shift()).cumsum()
    segment_lengths = (
        frame.groupby(segment_id)["hedge_signal"]
        .first()
        .to_frame(name="signal")
        .assign(length=frame.groupby(segment_id).size().values)
    )
    persistence = (
        segment_lengths.groupby("signal")["length"].mean().rename("avg_segment_len")
    )
    summary_path.write_text(
        "\n".join(
            [
                "Hedge-cost monitoring:",
                f"  ratio low (quantile {ratio_quantiles[0]}): {ratio_low:.3f}",
                f"  ratio high (quantile {ratio_quantiles[1]}): {ratio_high:.3f}",
                "",
                "Signal mix:",
                f"  cheap: {(frame['hedge_signal'] == 'cheap').mean():.2%}",
                f"  neutral: {(frame['hedge_signal'] == 'neutral').mean():.2%}",
                f"  expensive: {(frame['hedge_signal'] == 'expensive').mean():.2%}",
                "",
                "Signal persistence (avg segment length, days):",
                f"  cheap: {persistence.get('cheap', 0):.1f}",
                f"  neutral: {persistence.get('neutral', 0):.1f}",
                f"  expensive: {persistence.get('expensive', 0):.1f}",
            ]
        ),
        encoding="utf-8",
    )

    frame.to_csv(data_dir / "hedge_monitoring.csv", index=False)
    signal_counts.to_csv(data_dir / "signal_mix.csv")
    by_regime.to_csv(data_dir / "signal_by_regime.csv", index=False)
    persistence.to_csv(data_dir / "signal_persistence.csv")

    return HedgeMonitoringResults(ratio_low=float(ratio_low), ratio_high=float(ratio_high))


def _plot_hedge_ratio(
    frame: pd.DataFrame, output_dir: Path, ratio_low: float, ratio_high: float
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["date"], frame["hedge_ratio"], color="slateblue", linewidth=0.8)
    ax.axhline(ratio_low, color="seagreen", linestyle="--", linewidth=0.8, label="Cheap")
    ax.axhline(
        ratio_high, color="firebrick", linestyle="--", linewidth=0.8, label="Expensive"
    )
    ax.set_title("Hedge Cost Ratio (VIX / Realized Vol)")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "hedge_ratio.png", dpi=150)
    plt.close(fig)


def _plot_vix_vs_realized(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["date"], frame["vix_close"], color="black", linewidth=0.8, label="VIX")
    ax.plot(
        frame["date"],
        frame["realized_vol"],
        color="darkorange",
        linewidth=0.8,
        label="Realized Vol",
    )
    ax.set_title("VIX vs Realized Volatility (Annualized %)")
    ax.set_ylabel("Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_vs_realized.png", dpi=150)
    plt.close(fig)
