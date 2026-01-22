from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_regime_trend_sweep import (
    MATRIX_LIBRARY,
    default_sweep_config,
    run_regime_trend_sweep,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "regime_trend_best.json"


def _coerce_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _select_best(results: pd.DataFrame, evaluation: str) -> pd.Series:
    if evaluation == "single_split":
        pass_col = "holdout_pass_dd"
        sort_col = "holdout_sortino"
    else:
        pass_col = "val_pass_dd"
        sort_col = "val_sortino_mean"

    if pass_col in results.columns:
        results[pass_col] = results[pass_col].apply(_coerce_bool)

    eligible = results
    if pass_col in results.columns:
        eligible = results.loc[results[pass_col]]
    if eligible.empty:
        eligible = results

    if sort_col not in eligible.columns:
        raise ValueError(f"Missing sort column '{sort_col}' in sweep results.")

    return eligible.sort_values(sort_col, ascending=False).iloc[0]


def _build_frozen_config(row: pd.Series, sweep_cfg, results_path: Path) -> dict:
    matrix_name = str(row["matrix_name"])
    exposure_matrix = MATRIX_LIBRARY.get(matrix_name)
    if exposure_matrix is None:
        raise ValueError(f"Unknown matrix_name: {matrix_name}")

    return {
        "selected_at": date.today().isoformat(),
        "strategy": "regime_trend",
        "objective": "sortino_net",
        "evaluation": sweep_cfg.evaluation,
        "drawdown_cap": sweep_cfg.drawdown_cap,
        "cost_bps": sweep_cfg.cost_bps,
        "walk_forward": {
            "train_years": sweep_cfg.train_years,
            "val_years": sweep_cfg.val_years,
            "test_years": sweep_cfg.test_years,
            "step_years": sweep_cfg.step_years,
        },
        "hyperparams": {
            "trend_window": int(row["trend_window"]),
            "trend_threshold": float(row["trend_threshold"]),
            "rebalance": str(row["rebalance"]),
            "state_confirm": int(row["state_confirm"]),
            "matrix_name": matrix_name,
            "sizing_mode": str(row["sizing_mode"]),
            "trend_z_window": sweep_cfg.trend_z_window,
            "vol_z_window": sweep_cfg.vol_z_window,
            "base_exposure": sweep_cfg.base_exposure,
            "trend_coef": sweep_cfg.trend_coef,
            "vol_coef": sweep_cfg.vol_coef,
            "min_exposure": sweep_cfg.min_exposure,
            "max_exposure": sweep_cfg.max_exposure,
            "transition_window": sweep_cfg.transition_window,
            "transition_multiplier": sweep_cfg.transition_multiplier,
            "exposure_matrix": exposure_matrix,
        },
        "selection_metrics": row.to_dict(),
        "sweep_results_path": str(results_path),
    }


def run_optimization(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    skip_sweep: bool = False,
) -> Path:
    sweep_cfg = default_sweep_config()
    results_path = sweep_cfg.output_dir / "data" / "sweep_results.csv"

    if not skip_sweep:
        run_regime_trend_sweep(sweep_cfg)

    if not results_path.exists():
        raise FileNotFoundError(f"Sweep results not found at {results_path}")

    results = pd.read_csv(results_path)
    best_row = _select_best(results, sweep_cfg.evaluation)
    frozen = _build_frozen_config(best_row, sweep_cfg, results_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Wrote frozen config to {config_path}")
    return config_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize regime-trend strategy and write a frozen config."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Where to write the frozen config JSON.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Use existing sweep results instead of running the sweep.",
    )

    args = parser.parse_args()
    run_optimization(config_path=args.config_path, skip_sweep=args.skip_sweep)
