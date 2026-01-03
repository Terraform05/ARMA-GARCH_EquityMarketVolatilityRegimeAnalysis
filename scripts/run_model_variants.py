from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_CSV
from src.model_variants import run_model_variants


def run_model_variants_job(
    input_csv: str | Path = OUTPUT_CSV,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "modeling_variants",
    arma_max_p: int = 2,
    arma_max_q: int = 2,
) -> None:
    data = pd.read_csv(input_csv, parse_dates=["date"])
    run_model_variants(data, Path(output_dir), arma_max_p=arma_max_p, arma_max_q=arma_max_q)
    print(f"Wrote model variant outputs to {output_dir}")


if __name__ == "__main__":
    run_model_variants_job()
