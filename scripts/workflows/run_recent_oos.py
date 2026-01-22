from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from scripts.prepare_data import run_data_prep
from src.config import (
    BEST_VARIANT_FILE,
    VARIANT_BIC_METRICS_FILE,
    VARIANT_METRICS_FILE,
    VARIANT_SELECTION,
    RMSE_CLOSE_PCT,
    BIC_IMPROVEMENT,
)
from src.diagnostics import run_diagnostics
from src.garch_utils import fit_garch_variant, get_best_variant
from src.hedge_monitoring import run_hedge_monitoring
from src.model_variants import run_model_variants
from src.modeling import run_modeling, select_arma_order
from src.oos_check import run_oos_check
from src.regime_analysis import run_regime_analysis
from src.strategy_backtest import run_strategy_backtest
from src.validation import run_validation


# -----------------------------
# User-configurable settings
# -----------------------------
PROJECT_NAME = "aram_last_year"
# Separate root folder for this workflow's outputs (keeps reports/ untouched).
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "oos"

# Evaluation window: last 1 year ending today by default.
EVAL_END = None  # "YYYY-MM-DD" or None for today
EVAL_YEARS = 1

# Training window length before evaluation.
TRAIN_YEARS = 5

# ARMA search space for training window selection.
ARMA_MAX_P = 3
ARMA_MAX_Q = 3

# Set to True to avoid overwriting if you rerun the same window.
USE_RUN_TIMESTAMP = True

# Optional OOS graph/report steps.
RUN_ALL_PLOTS = True
RUN_DIAGNOSTICS = True
RUN_MODEL_VARIANTS = True
RUN_MODELING = True
RUN_VALIDATION = True
RUN_OOS_CHECK = True
RUN_REGIME_ANALYSIS = True
RUN_STRATEGY_BACKTEST = True
RUN_HEDGE_MONITORING = True


def _parse_date(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(value).normalize()


def _compute_dates(
    eval_end: str | None, eval_years: int, train_years: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    eval_end_ts = _parse_date(eval_end)
    eval_start_ts = eval_end_ts - pd.DateOffset(years=eval_years)
    train_start_ts = eval_start_ts - pd.DateOffset(years=train_years)
    return train_start_ts, eval_start_ts, eval_end_ts


def _run_dir(
    output_root: Path,
    project_name: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_years: int,
) -> Path:
    window = f"{eval_start.date().isoformat()}_to_{eval_end.date().isoformat()}"
    train_tag = f"train_{train_years}y"
    if USE_RUN_TIMESTAMP:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_root / project_name / window / train_tag / stamp
    return output_root / project_name / window / train_tag


def run_recent_oos_workflow() -> None:
    run_diagnostics_flag = RUN_ALL_PLOTS or RUN_DIAGNOSTICS
    run_model_variants_flag = RUN_ALL_PLOTS or RUN_MODEL_VARIANTS
    run_modeling_flag = RUN_ALL_PLOTS or RUN_MODELING
    run_validation_flag = RUN_ALL_PLOTS or RUN_VALIDATION
    run_oos_check_flag = RUN_ALL_PLOTS or RUN_OOS_CHECK
    run_regime_flag = RUN_ALL_PLOTS or RUN_REGIME_ANALYSIS
    run_strategy_flag = RUN_ALL_PLOTS or RUN_STRATEGY_BACKTEST
    run_hedge_flag = RUN_ALL_PLOTS or RUN_HEDGE_MONITORING

    train_start, eval_start, eval_end_ts = _compute_dates(
        EVAL_END, EVAL_YEARS, TRAIN_YEARS
    )
    output_root = Path(OUTPUT_ROOT)
    run_dir = _run_dir(output_root, PROJECT_NAME, eval_start, eval_end_ts, TRAIN_YEARS)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    data_path = inputs_dir / "spx_vix_aligned.csv"

    run_data_prep(
        start=train_start.date().isoformat(),
        end=eval_end_ts.date().isoformat(),
        output=data_path,
    )

    data = pd.read_csv(data_path, parse_dates=["date"])
    data["date"] = pd.to_datetime(data["date"])
    train = data[(data["date"] >= train_start) &
                 (data["date"] < eval_start)].copy()
    if train.empty:
        raise ValueError("Training window produced no rows.")

    returns = train["log_return"].dropna()
    if returns.empty:
        raise ValueError("Training window produced no returns.")

    p, q, _ = select_arma_order(returns, max_p=ARMA_MAX_P, max_q=ARMA_MAX_Q)
    if run_diagnostics_flag:
        run_diagnostics(train, run_dir / "diagnostics")

    if run_model_variants_flag:
        run_model_variants(
            train, run_dir / "modeling_variants", ARMA_MAX_P, ARMA_MAX_Q)

    best_variant_path = (
        run_dir / "modeling_variants" / "data" / "best_variant.txt"
        if run_model_variants_flag
        else BEST_VARIANT_FILE
    )
    metrics_path = (
        run_dir / "modeling_variants" / "data" / "variant_realized_metrics.csv"
        if run_model_variants_flag
        else VARIANT_METRICS_FILE
    )
    bic_metrics_path = (
        run_dir / "modeling_variants" / "data" / "variant_metrics.csv"
        if run_model_variants_flag
        else VARIANT_BIC_METRICS_FILE
    )
    variant = get_best_variant(
        best_variant_path,
        metrics_path,
        bic_metrics_path=bic_metrics_path,
        default="GARCH",
        mode=VARIANT_SELECTION,
        rmse_close_pct=RMSE_CLOSE_PCT,
        bic_improvement=BIC_IMPROVEMENT,
    )

    if run_modeling_flag:
        run_modeling(train, run_dir / "modeling",
                     ARMA_MAX_P, ARMA_MAX_Q, variant=variant)

    if run_validation_flag:
        arma_result = ARIMA(returns, order=(p, 0, q)).fit()
        resid = arma_result.resid.dropna()
        garch_result = fit_garch_variant(resid, variant)
        run_validation(garch_result.std_resid, run_dir / "validation")

    if run_oos_check_flag:
        oos_dir = run_dir / "oos_check"
        run_oos_check(
            data,
            oos_dir,
            split_date=eval_start.date().isoformat(),
            arma_order=(p, q),
            variant=variant,
        )

    if (run_regime_flag or run_strategy_flag or run_hedge_flag) and not run_oos_check_flag:
        raise ValueError(
            "OOS check must run to generate regime/strategy/hedge outputs.")

    if (run_strategy_flag or run_hedge_flag) and not run_regime_flag:
        raise ValueError(
            "Regime analysis must run for strategy/hedge outputs.")

    if run_regime_flag or run_strategy_flag or run_hedge_flag:
        oos_path = run_dir / "oos_check" / "data" / "forecast_vs_realized.csv"
        oos = pd.read_csv(oos_path, parse_dates=["date"])
        oos = oos.dropna(subset=["forecast_vol_rolling"])
        if oos.empty:
            raise ValueError(
                "No rolling forecasts available for regime analysis.")

        vol_scale = (252**0.5) * 100.0
        conditional_vol = oos[["date", "forecast_vol_rolling"]].rename(
            columns={"forecast_vol_rolling": "cond_vol"}
        )
        conditional_vol["cond_vol"] = conditional_vol["cond_vol"] / vol_scale

        eval_data = data[
            (data["date"] >= eval_start) & (data["date"] <= eval_end_ts)
        ].copy()
        regime_dir = run_dir / "regime_analysis"
        if run_regime_flag:
            run_regime_analysis(eval_data, conditional_vol, regime_dir)

        regime_series = pd.read_csv(
            regime_dir / "data" / "regime_series.csv", parse_dates=["date"]
        )

        if run_strategy_flag:
            strategy_dir = run_dir / "strategy_backtest"
            run_strategy_backtest(regime_series, strategy_dir)

        if run_hedge_flag:
            hedge_dir = run_dir / "hedge_monitoring"
            run_hedge_monitoring(regime_series, hedge_dir)

    print(
        "OOS run complete:",
        f"train={train_start.date().isoformat()} to {eval_start.date().isoformat()}",
        f"eval={eval_start.date().isoformat()} to {eval_end_ts.date().isoformat()}",
        f"arma=({p}, {q})",
        f"variant={variant}",
        f"outputs={run_dir}",
    )


if __name__ == "__main__":
    run_recent_oos_workflow()
