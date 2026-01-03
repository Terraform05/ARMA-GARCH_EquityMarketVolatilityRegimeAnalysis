from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_VARIANT_FILE, OUTPUT_CSV
from src.garch_utils import get_best_variant
from src.modeling import run_modeling

def run_modeling_job(
    input_csv: str | Path = OUTPUT_CSV,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "modeling",
    arma_max_p: int = 2,
    arma_max_q: int = 2,
    variant: str | None = None,
) -> None:
    data = pd.read_csv(input_csv, parse_dates=["date"])
    chosen_variant = variant or get_best_variant(BEST_VARIANT_FILE)
    run_modeling(
        data,
        Path(output_dir),
        arma_max_p=arma_max_p,
        arma_max_q=arma_max_q,
        variant=chosen_variant,
    )
    print(f"Wrote modeling outputs to {output_dir}")


if __name__ == "__main__":
    run_modeling_job()
