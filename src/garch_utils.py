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
    default: str = "GARCH",
    mode: str = "bic",
    tracking_metric: str = "rmse",
) -> str:
    if mode == "tracking" and metrics_path and metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if tracking_metric in metrics.columns:
            best_row = metrics.sort_values(tracking_metric, ascending=True).iloc[0]
            return str(best_row["variant"])
    if not best_variant_path.exists():
        return default
    content = best_variant_path.read_text(encoding="utf-8").strip().splitlines()
    for line in content:
        if "Best variant" in line:
            return line.split(":")[-1].strip()
    return default
