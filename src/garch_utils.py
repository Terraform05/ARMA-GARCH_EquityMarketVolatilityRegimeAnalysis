from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from arch import arch_model


DistName = Literal["normal", "gaussian", "t", "studentst", "skewstudent", "skewt", "ged", "generalized error"]


def fit_garch_variant(resid: pd.Series, variant: str):
    dist: DistName = "t" if variant.endswith("_t") else "normal"
    base = variant.replace("_t", "")

    if base == "GARCH":
        return arch_model(
            resid, mean="Zero", vol="GARCH", p=1, q=1, dist=dist, rescale=False
        ).fit(disp="off")
    if base == "GJR":
        return arch_model(
            resid, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=False
        ).fit(disp="off")
    if base == "EGARCH":
        return arch_model(
            resid, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist=dist, rescale=False
        ).fit(disp="off")

    raise ValueError(f"Unknown variant: {variant}")


def get_best_variant(
    best_variant_path: Path,
    metrics_path: Path | None = None,
    bic_metrics_path: Path | None = None,
    default: str = "GARCH",
    mode: str = "bic",
    tracking_metric: str = "rmse",
    rmse_close_pct: float = 0.02,
    bic_improvement: float = 10.0,
) -> str:
    if mode == "tracking" and metrics_path and metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if tracking_metric in metrics.columns:
            best_row = metrics.sort_values(tracking_metric, ascending=True).iloc[0]
            return str(best_row["variant"])
    if mode == "hybrid" and metrics_path and metrics_path.exists():
        tracking = pd.read_csv(metrics_path)
        if tracking_metric not in tracking.columns:
            return default
        best_track = tracking.sort_values(tracking_metric, ascending=True).iloc[0]
        best_rmse = float(best_track[tracking_metric])
        rmse_threshold = best_rmse * (1.0 + rmse_close_pct)

        if bic_metrics_path and bic_metrics_path.exists():
            bic_df = pd.read_csv(bic_metrics_path)
            merged = tracking.merge(
                bic_df[["variant", "bic"]], on="variant", how="inner"
            )
            close = merged.loc[merged[tracking_metric] <= rmse_threshold]
            if not close.empty:
                best_bic_row = close.sort_values("bic", ascending=True).iloc[0]
                rmse_best_bic = float(best_bic_row[tracking_metric])
                bic_best_bic = float(best_bic_row["bic"])

                best_track_bic = merged.loc[
                    merged["variant"] == best_track["variant"], "bic"
                ]
                if not best_track_bic.empty:
                    bic_track = float(best_track_bic.iloc[0])
                    if bic_track - bic_best_bic >= bic_improvement:
                        return str(best_bic_row["variant"])

                if rmse_best_bic <= rmse_threshold:
                    return str(best_bic_row["variant"])

        return str(best_track["variant"])
    if not best_variant_path.exists():
        return default
    content = best_variant_path.read_text(encoding="utf-8").strip().splitlines()
    for line in content:
        if line.startswith("Best variant (active):"):
            return line.split(":", 1)[-1].strip()
    for line in content:
        if line.startswith("Best variant by BIC:"):
            return line.split(":", 1)[-1].strip()
    for line in content:
        if "Best variant" in line:
            return line.split(":", 1)[-1].strip()
    return default
