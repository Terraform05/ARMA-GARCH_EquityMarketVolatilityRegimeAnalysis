# Reports Index

All contents under `reports/` are regenerated on each run.

## Core Pipeline Outputs

- `reports/diagnostics/`
  - `data/summary.txt`
  - `plots/returns_series.png`, `plots/acf_pacf.png`
- `reports/modeling_variants/`
  - `data/best_variant.txt`
  - `data/variant_metrics.csv`, `data/variant_realized_metrics.csv`
  - `plots/variant_comparison.png`, `plots/variant_vs_realized.png`, `plots/bic_vs_tracking.png`
- `reports/modeling/`
  - `data/summary.txt`, `data/parameters.csv`, `data/conditional_volatility.csv`
- `reports/validation/`
  - `data/summary.txt`
  - `plots/residuals_series.png`, `plots/residuals_acf.png`, `plots/residuals_qq.png`
- `reports/regime_analysis/`
  - `data/regime_series.csv`
  - `data/regime_outcomes.csv`, `data/realized_window_metrics.csv`, `data/summary.txt`
  - `plots/regimes.png`, `plots/vix_vs_realized.png`, `plots/realized_window_metrics.png`, `plots/regime_outcomes.png`
- `reports/oos_check/`
  - `data/oos_metrics.csv`, `data/forecast_vs_realized.csv`, `data/summary.txt`
  - `plots/forecast_vs_realized.png`, `plots/forecast_vs_realized_rolling.png`

## Strategy Outputs

- `reports/strategy_backtest/`
  - `data/strategy_equity.csv`, `data/strategy_variants.csv`, `data/summary.txt`
  - `plots/equity_curve.png`, `plots/equity_curve_last_year.png`
  - `plots/exposure_overlay.png`, `plots/exposure_overlay_last_year.png`
  - `plots/rolling_alpha_beta.png`
- `reports/strategy_regime_trend/`
  - `data/regime_trend_equity.csv`, `data/summary.txt`
  - `data/turnover_stats.csv`, `data/cost_sensitivity.csv`
  - `plots/equity_curve.png`, `plots/equity_curve_last_year.png`
  - `plots/equity_curve_compare.png`, `plots/equity_curve_compare_last_year.png`
  - `plots/exposure_overlay.png`, `plots/exposure_overlay_last_year.png`
  - `plots/rolling_cagr.png`, `plots/rolling_drawdown.png`, `plots/rolling_alpha_beta.png`
  - `plots/turnover_hist.png`, `plots/cost_sensitivity.png`
- `reports/strategy_regime_trend_sweep/`
  - `data/sweep_results.csv`, `data/top_candidates.csv`, `data/sweep_summary.txt`

## Hedge Monitoring

- `reports/hedge_monitoring/`
  - `data/hedge_monitoring.csv`, `data/summary.txt`
  - `plots/hedge_ratio.png`, `plots/vix_vs_realized.png`
