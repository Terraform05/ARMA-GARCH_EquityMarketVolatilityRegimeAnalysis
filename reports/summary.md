# Executive Summary

## Overview

This project interprets volatility regimes in equity markets by modeling SPX return dynamics with ARMA and conditional volatility with GARCH, then comparing model-based volatility to VIX and realized volatility proxies.

## Data & Scope

- Source: Yahoo Finance (`^GSPC`, `^VIX`)
- Frequency: daily EOD
- Range: 2010-01-01 to 2026-01-01
- Output: aligned CSV with `log_return` and `sq_return`

## Key Findings

- **Stationarity:** ADF on log returns strongly rejects a unit root (p < 0.001).
- **Volatility clustering:** ARCH test is highly significant (p < 0.001).
- **Mean process:** ARMA order selected by BIC is (2, 0).
- **Volatility model:** GARCH is selected by realized‑vol tracking (alpha + beta = 0.9800), indicating persistent volatility with strong clustering.
- **Regimes:** Conditional volatility thresholds (33%/66% quantiles) split into low and high regimes; high regimes align with known stress windows. The 10-day realized vol window provides the strongest alignment with VIX in-sample.
- **Hedge monitoring:** Hedge ratio thresholds are 1.056 (cheap) and 1.925 (expensive), with 20.00% cheap / 59.99% neutral / 20.00% expensive and average signal lengths of 4.6/7.8/5.9 days.
- **Strategy backtest:** Regime strategy returns 0.0618 with vol 0.0872, Sharpe 0.7085, and max drawdown -0.1234 vs buy‑and‑hold return 0.1128 with vol 0.1738 and max drawdown -0.3392.

## Validation Summary

- Ljung-Box on standardized residuals remains significant (p ≈ 0.000011).
- Ljung-Box on squared residuals remains significant (p ≈ 0.008656).
- ARCH test on standardized residuals remains significant (p ≈ 0.025483).
- Interpretation: variance dynamics remain partially unmodeled; use regimes as risk indicators rather than precision forecasts.

## Out-of-Sample Check

- Holdout window: 2024-01-01 to 2026-01-01 (502 rows).
- Static and rolling forecasts are both available for GARCH.
- Rolling 1-step forecasts provide the realistic comparison against realized volatility.

## Step Outputs

- Data prep: `data/processed/spx_vix_aligned.csv`
- Diagnostics: `reports/diagnostics/README.md`, `reports/diagnostics/plots/returns_series.png`, `reports/diagnostics/plots/acf_pacf.png`
- Modeling: `reports/modeling/README.md`, `reports/modeling/data/conditional_volatility.csv`, `reports/modeling/data/summary.txt`
- Modeling variants: `reports/modeling_variants/data/variant_metrics.csv`, `reports/modeling_variants/data/best_variant.txt`, `reports/modeling_variants/plots/variant_comparison.png`, `reports/modeling_variants/plots/variant_vs_realized.png`, `reports/modeling_variants/plots/variant_metrics.png`
- Validation: `reports/validation/README.md`, `reports/validation/plots/residuals_acf.png`, `reports/validation/plots/residuals_qq.png`
- Regime analysis: `reports/regime_analysis/README.md`, `reports/regime_analysis/plots/vix_vs_realized.png`, `reports/regime_analysis/plots/realized_window_metrics.png`, `reports/regime_analysis/plots/regime_outcomes.png`
- OOS check: `reports/oos_check/README.md`, `reports/oos_check/plots/forecast_vs_realized_rolling.png`, `reports/oos_check/data/oos_metrics.csv`
- Insights: `reports/insights.md`
- Hedge monitoring: `reports/hedge_monitoring/README.md`
- Strategy backtest: `reports/strategy_backtest/README.md`
 - Hedge + strategy overview: `reports/hedge_strategy/README.md`

## How To Reproduce

1. `python scripts/prepare_data.py`
2. `python scripts/run_diagnostics.py`
3. `python scripts/run_model_variants.py`
4. `python scripts/run_modeling.py`
5. `python scripts/run_validation.py`
6. `python scripts/run_regime_analysis.py`
7. `python scripts/run_oos_check.py`
8. `python scripts/run_hedge_monitoring.py`
9. `python scripts/run_strategy_backtest.py`

Or run everything in one go:

- `python scripts/run_all.py`
