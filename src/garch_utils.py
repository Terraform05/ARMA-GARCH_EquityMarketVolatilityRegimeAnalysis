from __future__ import annotations

from pathlib import Path

import pandas as pd
from arch import arch_model


def fit_garch_variant(resid: pd.Series, variant: str):
    dist = "t" if variant.endswith("_t") else "normal"
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


def get_best_variant(best_variant_path: Path, default: str = "GARCH") -> str:
    if not best_variant_path.exists():
        return default
    content = best_variant_path.read_text(encoding="utf-8").strip().splitlines()
    for line in content:
        if "Best variant" in line:
            return line.split(":")[-1].strip()
    return default
