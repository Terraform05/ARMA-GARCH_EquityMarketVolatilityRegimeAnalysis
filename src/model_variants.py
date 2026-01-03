from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.garch_utils import fit_garch_variant
from src.modeling import select_arma_order


@dataclass
class VariantResult:
    variant: str
    aic: float
    bic: float
    loglikelihood: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fit_arma(returns: pd.Series, max_p: int, max_q: int) -> tuple[ARIMA, tuple[int, int]]:
    p, q, _ = select_arma_order(returns, max_p, max_q)
    model = ARIMA(returns, order=(p, 0, q))
    return model.fit(), (p, q)


def run_model_variants(
    data: pd.DataFrame,
    output_dir: Path,
    arma_max_p: int = 2,
    arma_max_q: int = 2,
) -> None:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)
    returns = data["log_return"].dropna()

    arma_result, arma_order = _fit_arma(returns, arma_max_p, arma_max_q)
    resid = arma_result.resid.dropna()

    variants = ["GARCH", "GARCH_t", "GJR", "GJR_t", "EGARCH", "EGARCH_t"]
    metrics: list[VariantResult] = []
    params_rows: list[dict[str, float | str]] = []
    vol_series: dict[str, pd.Series] = {}

    for variant in variants:
        result = fit_garch_variant(resid, variant)
        metrics.append(
            VariantResult(
                variant=variant,
                aic=float(result.aic),
                bic=float(result.bic),
                loglikelihood=float(result.loglikelihood),
            )
        )
        dates = data.loc[resid.index, "date"]
        vol_series[variant] = pd.Series(
            result.conditional_volatility.values, index=dates
        )
        for name, value in result.params.items():
            params_rows.append({"variant": variant, "parameter": name, "value": value})

    metrics_df = pd.DataFrame([m.__dict__ for m in metrics]).sort_values("bic")
    metrics_df.to_csv(data_dir / "variant_metrics.csv", index=False)

    params_df = pd.DataFrame(params_rows)
    params_df.to_csv(data_dir / "variant_params.csv", index=False)

    best_variant = metrics_df.iloc[0]["variant"]
    (data_dir / "best_variant.txt").write_text(
        f"ARMA order: {arma_order}\nBest variant by BIC: {best_variant}\n",
        encoding="utf-8",
    )

    vol_df = pd.concat(vol_series, axis=1)
    vol_df.index.name = "date"
    vol_df.reset_index().to_csv(data_dir / "variant_volatility.csv", index=False)

    annualized = vol_df * (252**0.5) * 100
    _plot_best_variant(annualized, best_variant, plots_dir)
    _plot_variant_comparison(
        annualized, metrics_df.head(3)["variant"].tolist(), plots_dir
    )
    _plot_variants_vs_realized(annualized, data, metrics_df, plots_dir)
    _plot_variant_metrics(metrics_df, plots_dir)
    _write_variant_realized_metrics(annualized, data, data_dir)


def _plot_best_variant(vol_df: pd.DataFrame, best_variant: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(vol_df.index, vol_df[best_variant], color="slateblue", linewidth=0.8)
    ax.set_title(f"Conditional Volatility (Annualized %): {best_variant}")
    ax.set_ylabel("Volatility (%)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / "best_variant_volatility.png", dpi=150)
    plt.close(fig)


def _plot_variant_comparison(
    vol_df: pd.DataFrame, variants: list[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for variant in variants:
        ax.plot(vol_df.index, vol_df[variant], linewidth=0.8, label=variant)
    ax.set_title("Conditional Volatility Comparison (Annualized %)")
    ax.set_ylabel("Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "variant_comparison.png", dpi=150)
    plt.close(fig)


def _plot_variants_vs_realized(
    vol_df: pd.DataFrame, data: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path
) -> None:
    realized = (
        data.set_index("date")["log_return"].rolling(21).std() * (252**0.5) * 100
    )
    aligned = pd.concat([vol_df, realized.rename("realized")], axis=1).dropna()
    top_variants = metrics_df.head(3)["variant"].tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        aligned.index,
        aligned["realized"],
        color="black",
        linewidth=0.9,
        label="Realized Vol (21D, annualized)",
    )
    for variant in top_variants:
        ax.plot(aligned.index, aligned[variant], linewidth=0.7, label=variant)
    ax.set_title("Top Variants vs Realized Volatility (Annualized %)")
    ax.set_ylabel("Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "variant_vs_realized.png", dpi=150)
    plt.close(fig)


def _plot_variant_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(metrics_df["variant"], metrics_df["bic"], color="slateblue")
    axes[0].set_title("BIC by Variant")
    axes[0].set_ylabel("BIC")
    axes[0].set_xlabel("Variant")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(metrics_df["variant"], metrics_df["aic"], color="darkorange")
    axes[1].set_title("AIC by Variant")
    axes[1].set_ylabel("AIC")
    axes[1].set_xlabel("Variant")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_dir / "variant_metrics.png", dpi=150)
    plt.close(fig)


def _write_variant_realized_metrics(
    vol_df: pd.DataFrame, data: pd.DataFrame, data_dir: Path
) -> None:
    realized = (
        data.set_index("date")["log_return"].rolling(21).std() * (252**0.5) * 100
    )
    aligned = pd.concat([vol_df, realized.rename("realized")], axis=1).dropna()
    rows = []
    for variant in vol_df.columns:
        corr = aligned[variant].corr(aligned["realized"])
        rmse = ((aligned[variant] - aligned["realized"]) ** 2).mean() ** 0.5
        variance = aligned[variant] ** 2
        realized_var = aligned["realized"] ** 2
        qlike = (realized_var / variance + variance.apply(np.log)).mean()
        rows.append({"variant": variant, "corr": corr, "rmse": rmse, "qlike": qlike})
    pd.DataFrame(rows).sort_values("rmse").to_csv(
        data_dir / "variant_realized_metrics.csv", index=False
    )
