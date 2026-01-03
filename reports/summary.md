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
- **Volatility model:** EGARCH_t is the best variant by BIC, indicating asymmetric volatility and fat tails improve fit.
- **Regimes:** Conditional volatility thresholds (33%/66% quantiles) split into low and high regimes; high regimes align with known stress windows. The 10-day realized vol window provides the strongest alignment with VIX in-sample.

## Validation Summary

- Ljung-Box on standardized residuals remains significant (p ≈ 0.00011).
- Ljung-Box on squared residuals is not significant (p ≈ 0.79).
- ARCH test on standardized residuals is not significant (p ≈ 0.81).
- Interpretation: variance dynamics are well captured; remaining dependence is mainly in the mean.

## Out-of-Sample Check

- Holdout window: 2024-01-01 to 2026-01-01 (502 rows).
- Static multi-step forecast is unavailable for EGARCH variants.
- Rolling 1-step forecasts provide the realistic comparison against realized volatility.

## Step Outputs

- Data prep: `data/processed/spx_vix_aligned.csv`
- Diagnostics: `reports/diagnostics/README.md`, `reports/diagnostics/plots/returns_series.png`, `reports/diagnostics/plots/acf_pacf.png`
- Modeling: `reports/modeling/README.md`, `reports/modeling/data/conditional_volatility.csv`, `reports/modeling/data/summary.txt`
- Modeling variants: `reports/modeling_variants/data/variant_metrics.csv`, `reports/modeling_variants/data/best_variant.txt`, `reports/modeling_variants/plots/variant_comparison.png`, `reports/modeling_variants/plots/variant_vs_realized.png`, `reports/modeling_variants/plots/variant_metrics.png`
- Validation: `reports/validation/README.md`, `reports/validation/plots/residuals_acf.png`, `reports/validation/plots/residuals_qq.png`
- Regime analysis: `reports/regime_analysis/README.md`, `reports/regime_analysis/plots/vix_vs_realized.png`, `reports/regime_analysis/plots/realized_window_metrics.png`
- OOS check: `reports/oos_check/README.md`, `reports/oos_check/plots/forecast_vs_realized_rolling.png`, `reports/oos_check/data/oos_metrics.csv`
- Insights: `reports/insights.md`

## How To Reproduce

1. `python scripts/prepare_data.py`
2. `python scripts/run_diagnostics.py`
3. `python scripts/run_modeling.py`
4. `python scripts/run_validation.py`
5. `python scripts/run_regime_analysis.py`
6. `python scripts/run_oos_check.py`

Or run everything in one go:

- `python scripts/run_all.py`

## Resume-Ready Bullets

- Modeled equity market volatility with ARMA–GARCH, identifying persistent volatility shocks and regime shifts across 2010–2026.
- Compared implied (VIX) and realized volatility to interpret risk dynamics and stress sensitivity in major market regimes.
