from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import run_data_prep
from scripts.run_diagnostics import run_diagnostics_job
from scripts.run_model_variants import run_model_variants_job
from scripts.run_modeling import run_modeling_job
from scripts.run_validation import run_validation_job
from scripts.run_regime_analysis import run_regime_job
from scripts.run_hedge_monitoring import run_hedge_monitoring_job
from scripts.run_strategy_backtest import run_strategy_backtest_job
from scripts.run_layered_strategy_backtest import run_layered_strategy_backtest_job
from scripts.run_oos_check import run_oos_job


def run_all() -> None:
    run_data_prep()
    run_diagnostics_job()
    run_model_variants_job()
    run_modeling_job()
    run_validation_job()
    run_regime_job()
    run_oos_job()
    run_hedge_monitoring_job()
    run_strategy_backtest_job()
    run_layered_strategy_backtest_job()


if __name__ == "__main__":
    run_all()
