from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_CSV
from src.regime_analysis import run_regime_analysis


def run_regime_job(
    input_csv: str | Path = OUTPUT_CSV,
    conditional_vol_csv: str | Path = PROJECT_ROOT
    / "reports"
    / "modeling"
    / "data"
    / "conditional_volatility.csv",
    output_dir: str | Path = PROJECT_ROOT / "reports" / "regime_analysis",
) -> None:
    data = pd.read_csv(input_csv, parse_dates=["date"])
    conditional_vol = pd.read_csv(conditional_vol_csv, parse_dates=["date"])
    run_regime_analysis(data, conditional_vol, Path(output_dir))
    print(f"Wrote regime analysis outputs to {output_dir}")


if __name__ == "__main__":
    run_regime_job()
