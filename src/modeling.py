from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.garch_utils import fit_garch_variant

@dataclass
class ModelingResults:
    arma_order: tuple[int, int]
    arma_bic: float
    garch_params: dict[str, float]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _candidate_orders(max_p: int, max_q: int) -> Iterable[tuple[int, int]]:
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            yield p, q


def select_arma_order(returns: pd.Series, max_p: int = 2, max_q: int = 2) -> tuple[int, int, float]:
    best_order: tuple[int, int] | None = None
    best_bic = float("inf")

    for p, q in _candidate_orders(max_p, max_q):
        try:
            model = ARIMA(returns, order=(p, 0, q))
            result = model.fit()
        except Exception:
            continue

        if result.bic < best_bic:
            best_bic = result.bic
            best_order = (p, q)

    if best_order is None:
        raise RuntimeError(
            "Failed to fit any ARMA model for the provided returns.")

    return best_order[0], best_order[1], best_bic


def run_modeling(
    data: pd.DataFrame,
    output_dir: Path,
    arma_max_p: int = 2,
    arma_max_q: int = 2,
    variant: str = "GARCH",
) -> ModelingResults:
    _ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    _ensure_dir(plots_dir)
    _ensure_dir(data_dir)
    returns = data["log_return"].dropna()

    p, q, bic = select_arma_order(returns, arma_max_p, arma_max_q)
    arma_result = ARIMA(returns, order=(p, 0, q)).fit()

    resid = arma_result.resid.dropna()
    garch_result = fit_garch_variant(resid, variant)

    params_path = data_dir / "parameters.csv"
    params = (
        pd.concat(
            [
                arma_result.params.rename("value").to_frame().assign(model="arma"),
                garch_result.params.rename("value").to_frame().assign(model="garch"),
            ]
        )
        .reset_index()
        .rename(columns={"index": "parameter"})
    )
    params.to_csv(params_path, index=False)

    conditional_vol = garch_result.conditional_volatility
    vol_series = pd.DataFrame(
        {"date": data.loc[resid.index, "date"], "cond_vol": conditional_vol}
    )
    vol_series.to_csv(data_dir / "conditional_volatility.csv", index=False)

    persistence = garch_result.params.get("alpha[1]", 0.0) + garch_result.params.get(
        "beta[1]", 0.0
    )
    summary_path = data_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "ARMA model selection:",
                f"  selected order: ({p}, {q})",
                f"  BIC: {bic:.6f}",
                "",
                f"{variant} parameters:",
                f"  omega: {garch_result.params.get('omega', float('nan')):.6f}",
                f"  alpha[1]: {garch_result.params.get('alpha[1]', float('nan')):.6f}",
                f"  beta[1]: {garch_result.params.get('beta[1]', float('nan')):.6f}",
                f"  alpha + beta: {persistence:.6f}",
            ]
        ),
        encoding="utf-8",
    )

    return ModelingResults(
        arma_order=(p, q),
        arma_bic=bic,
        garch_params=garch_result.params.to_dict(),
    )
