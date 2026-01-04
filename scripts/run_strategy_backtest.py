from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy_backtest import run_strategy_backtest


def run_strategy_backtest_job(
    regime_csv: str | Path = PROJECT_ROOT
    / "reports"
    / "regime_analysis"
    / "data"
    / "regime_series.csv",
    output_dir: str | Path = PROJECT_ROOT / "reports" / "strategy_backtest",
) -> None:
    data = pd.read_csv(regime_csv, parse_dates=["date"])
    run_strategy_backtest(data, Path(output_dir))
    print(f"Wrote strategy backtest outputs to {output_dir}")


if __name__ == "__main__":
    run_strategy_backtest_job()
