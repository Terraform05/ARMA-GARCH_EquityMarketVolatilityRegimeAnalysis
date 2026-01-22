# Reports Catalog

This directory contains curated outputs intended for reading and comparison.

## Core Reports

- `reports/diagnostics/`: stationarity + ACF/PACF diagnostics
- `reports/modeling/`: ARMA + GARCH fit outputs
- `reports/modeling_variants/`: variant comparison and tracking metrics
- `reports/validation/`: residual diagnostics and model validation
- `reports/regime_analysis/`: regime labels and realized vs implied comparisons
- `reports/oos_check/`: out-of-sample forecast checks
- `reports/hedge_monitoring/`: hedge cost and regime signals

## Strategies

- `reports/strategy_backtest/`: regime-only strategy backtest
- `reports/strategy_regime_trend/`: regime+trend backtest
- `reports/strategy_regime_trend_sweep/`: optimization sweep outputs (data/plots)
- `reports/strategy_regime_trend_gbm/`: GBM forward test for regime-trend strategy
- `reports/hedge_strategy/`: hedge strategy notes

## Notes

- Data tables live in `reports/**/data/`
- Plots live in `reports/**/plots/`
- For timestamped runs and historical comparisons, see `runs/`
