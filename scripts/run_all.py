from __future__ import annotations

from pathlib import Path
import sys
import json
import time

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
from scripts.run_regime_trend_backtest import run_regime_trend_backtest_job
from scripts.run_oos_check import run_oos_job
from src.run_registry import RunRecord, create_run_id, snapshot_config, write_run_registry


def run_all() -> None:
    started = time.time()
    run_id = create_run_id()
    reports_dir = PROJECT_ROOT / "reports"
    registry_path = PROJECT_ROOT / "runs" / "index.csv"
    active_config_dir = PROJECT_ROOT / "configs" / "active"
    regime_trend_config_path = PROJECT_ROOT / "configs" / "regime_trend_best.json"
    objective = "sortino_net"
    evaluation = "walk_forward"
    drawdown_cap = ""
    cost_bps = ""
    if regime_trend_config_path.exists():
        try:
            cfg = json.loads(regime_trend_config_path.read_text(encoding="utf-8"))
            drawdown_cap = str(cfg.get("drawdown_cap", ""))
            cost_bps = str(cfg.get("cost_bps", ""))
            objective = str(cfg.get("objective", objective))
            evaluation = str(cfg.get("evaluation", evaluation))
        except json.JSONDecodeError:
            pass
    data_end_date = ""
    data_path = PROJECT_ROOT / "data" / "processed" / "spx_vix_aligned.csv"
    if data_path.exists():
        try:
            import csv

            with data_path.open() as f:
                reader = csv.reader(f)
                next(reader, None)
                last = None
                for row in reader:
                    last = row
            if last:
                data_end_date = last[0]
        except Exception:
            data_end_date = ""

    active_snapshot = snapshot_config(
        regime_trend_config_path, active_config_dir, run_id
    )

    run_data_prep()
    run_diagnostics_job()
    run_model_variants_job()
    run_modeling_job()
    run_validation_job()
    run_regime_job()
    run_oos_job()
    run_hedge_monitoring_job()
    run_strategy_backtest_job()
    run_regime_trend_backtest_job()

    ended = time.time()
    record = RunRecord(
        run_id=run_id,
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        ended_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
        status="success",
        data_end_date=data_end_date,
        objective=objective,
        evaluation=evaluation,
        drawdown_cap=drawdown_cap,
        cost_bps=cost_bps,
        frozen_config_path=str(active_snapshot) if active_snapshot else "",
        reports_dir=str(reports_dir),
        notes="",
    )
    write_run_registry(registry_path, record)


if __name__ == "__main__":
    run_all()
