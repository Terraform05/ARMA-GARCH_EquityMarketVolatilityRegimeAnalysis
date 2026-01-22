from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import run_data_prep
from scripts.run_model_variants import run_model_variants_job
from scripts.run_modeling import run_modeling_job
from scripts.run_regime_analysis import run_regime_job
from src.config import BEST_VARIANT_FILE, OUTPUT_CSV, VARIANT_METRICS_FILE


CONDITIONAL_VOL_CSV = (
    PROJECT_ROOT / "reports" / "modeling" / "data" / "conditional_volatility.csv"
)
REGIME_SERIES_CSV = (
    PROJECT_ROOT / "reports" / "regime_analysis" / "data" / "regime_series.csv"
)


def run_regime_series_job() -> None:
    if not OUTPUT_CSV.exists():
        run_data_prep()
    if not (BEST_VARIANT_FILE.exists() and VARIANT_METRICS_FILE.exists()):
        run_model_variants_job()
    if not CONDITIONAL_VOL_CSV.exists():
        run_modeling_job()
    run_regime_job()
    print(f"Wrote regime series to {REGIME_SERIES_CSV}")


if __name__ == "__main__":
    run_regime_series_job()
