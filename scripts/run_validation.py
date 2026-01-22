from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_VARIANT_FILE,
    OUTPUT_CSV,
    VARIANT_BIC_METRICS_FILE,
    VARIANT_METRICS_FILE,
    VARIANT_SELECTION,
    RMSE_CLOSE_PCT,
    BIC_IMPROVEMENT,
)
from src.garch_utils import fit_garch_variant, get_best_variant
from src.modeling import select_arma_order
from src.validation import run_validation


def run_validation_job(
    input_csv: str | Path = OUTPUT_CSV,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "validation",
    arma_max_p: int = 2,
    arma_max_q: int = 2,
    lags: int = 20,
    variant: str | None = None,
) -> None:
    data = pd.read_csv(input_csv, parse_dates=["date"])
    returns = data["log_return"].dropna()

    p, q, _ = select_arma_order(returns, arma_max_p, arma_max_q)
    arma_result = ARIMA(returns, order=(p, 0, q)).fit()

    resid = arma_result.resid.dropna()
    chosen_variant = variant or get_best_variant(
        BEST_VARIANT_FILE,
        VARIANT_METRICS_FILE,
        bic_metrics_path=VARIANT_BIC_METRICS_FILE,
        mode=VARIANT_SELECTION,
        rmse_close_pct=RMSE_CLOSE_PCT,
        bic_improvement=BIC_IMPROVEMENT,
    )
    garch_result = fit_garch_variant(resid, chosen_variant)

    standardized_resid = garch_result.std_resid
    run_validation(standardized_resid, Path(output_dir), lags=lags)
    print(f"Wrote validation outputs to {output_dir}")


if __name__ == "__main__":
    run_validation_job()
