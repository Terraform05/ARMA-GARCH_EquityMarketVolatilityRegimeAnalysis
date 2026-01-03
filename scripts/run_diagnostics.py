from __future__ import annotations
from src.diagnostics import run_diagnostics
from src.config import OUTPUT_CSV

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_diagnostics_job(
    input_csv: str | Path = OUTPUT_CSV,
    output_dir: str | Path = PROJECT_ROOT / "reports" / "diagnostics",
    lags: int = 20,
) -> None:
    data = pd.read_csv(input_csv, parse_dates=["date"])
    run_diagnostics(data, Path(output_dir), lags=lags)
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    run_diagnostics_job()
