from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_VARIANT_FILE, OUTPUT_CSV
from src.garch_utils import get_best_variant
from src.oos_check import run_oos_check


def run_oos_job(
    input_csv: str | Path = OUTPUT_CSV,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "oos_check",
    split_date: str = "2024-01-01",
    arma_order: tuple[int, int] = (2, 0),
    variant: str | None = None,
) -> None:
    data = pd.read_csv(input_csv, parse_dates=["date"])
    chosen_variant = variant or get_best_variant(BEST_VARIANT_FILE)
    run_oos_check(
        data,
        Path(output_dir),
        split_date=split_date,
        arma_order=arma_order,
        variant=chosen_variant,
    )
    print(f"Wrote OOS outputs to {output_dir}")


if __name__ == "__main__":
    run_oos_job()
