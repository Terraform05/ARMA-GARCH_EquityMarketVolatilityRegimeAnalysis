# Plots

This document describes the plots and where they live. All plots are generated under `reports/` and are overwritten on each run. See `reports/README.md` for the current index.

## Diagnostics

![Returns and squared returns](../../reports/diagnostics/plots/returns_series.png)
![ACF and PACF](../../reports/diagnostics/plots/acf_pacf.png)

Plot notes:
- Returns and squared returns: squared returns cluster into bursts, showing volatility clustering and motivating GARCH.
- ACF and PACF: return autocorrelation decays quickly, but squared returns persist, indicating volatility dynamics.

## Model Variants and Selection

![Model variant comparison](../../reports/modeling_variants/plots/variant_comparison.png)
![Top variants vs realized](../../reports/modeling_variants/plots/variant_vs_realized.png)
![Variant metrics](../../reports/modeling_variants/plots/variant_metrics.png)
![Best variant volatility](../../reports/modeling_variants/plots/best_variant_volatility.png)
![BIC vs tracking](../../reports/modeling_variants/plots/bic_vs_tracking.png)

Plot notes:
- Variant comparison: top models co-move in calm periods; stress windows reveal divergence.
- Variants vs realized: closest tracking line indicates best realized-vol alignment.
- Variant metrics: lower (more negative) AIC/BIC is better, but only relative differences matter.
- Best variant volatility: the conditional vol path should spike during stress and mean-revert after.
- BIC vs tracking: the tradeoff shows whether you are optimizing fit or realized-vol tracking.

## Modeling

- `reports/modeling/data/conditional_volatility.csv` (series used for regimes)

## Validation

![Residual series](../../reports/validation/plots/residuals_series.png)
![Residual ACF](../../reports/validation/plots/residuals_acf.png)
![Residual Q-Q](../../reports/validation/plots/residuals_qq.png)

Plot notes:
- Residual series: clustered spikes indicate remaining structure after the fit.
- Residual ACF: residual and squared-residual autocorrelation shows incomplete mean and variance capture.
- Residual Q-Q: tail deviations from the diagonal suggest heavy tails and justify t-errors.

## Regime Analysis

![Regime scatter](../../reports/regime_analysis/plots/regimes.png)
![VIX vs realized](../../reports/regime_analysis/plots/vix_vs_realized.png)
![Window metrics](../../reports/regime_analysis/plots/realized_window_metrics.png)
![Regime outcomes](../../reports/regime_analysis/plots/regime_outcomes.png)

Plot notes:
- Regime scatter: long blocks show persistent volatility states; rapid flips indicate noisy transitions.
- VIX vs realized: alignment in peaks and troughs validates the realized window choice.
- Window metrics: the chosen window should sit near a stable correlation/RMSE tradeoff.
- Regime outcomes: high-vol regimes should show worse drawdowns and weaker returns.

## Out-of-Sample Check

![Out-of-sample forecast vs realized](../../reports/oos_check/plots/forecast_vs_realized.png)
![Out-of-sample rolling forecast vs realized](../../reports/oos_check/plots/forecast_vs_realized_rolling.png)

Plot notes:
- The forecast should move in the same direction as realized volatility even if levels differ.
- Persistent gaps indicate scaling errors or a mismatch between model and realized window length.

## Hedge Monitoring

![Hedge ratio](../../reports/hedge_monitoring/plots/hedge_ratio.png)
![VIX vs realized (hedge monitoring)](../../reports/hedge_monitoring/plots/vix_vs_realized.png)

Plot notes:
- Hedge ratio: highlights windows where hedging is cheap vs expensive.
- VIX vs realized: use this as context for hedge-cost regimes.

## Regime Strategy Backtest

![Regime strategy equity curve](../../reports/strategy_backtest/plots/equity_curve.png)
![Regime strategy equity curve (last year)](../../reports/strategy_backtest/plots/equity_curve_last_year.png)
![Exposure overlay](../../reports/strategy_backtest/plots/exposure_overlay.png)
![Exposure overlay (last year)](../../reports/strategy_backtest/plots/exposure_overlay_last_year.png)
![Rolling alpha/beta](../../reports/strategy_backtest/plots/rolling_alpha_beta.png)

Plot notes:
- Equity curve (last year): the regime line should dip less during drawdowns if the overlay is working.
- Exposure overlay: exposure steps down in high-volatility regimes and rises in low-volatility regimes.
- Rolling alpha/beta: sanity-check that alpha is not isolated to a single window.

## Regime-Trend Strategy Backtest

![Regime-trend equity curve](../../reports/strategy_regime_trend/plots/equity_curve.png)
![Regime-trend equity curve (last year)](../../reports/strategy_regime_trend/plots/equity_curve_last_year.png)
![Regime-trend vs baselines](../../reports/strategy_regime_trend/plots/equity_curve_compare.png)
![Regime-trend vs baselines (last year)](../../reports/strategy_regime_trend/plots/equity_curve_compare_last_year.png)
![Regime-trend exposure overlay](../../reports/strategy_regime_trend/plots/exposure_overlay.png)
![Regime-trend exposure overlay (last year)](../../reports/strategy_regime_trend/plots/exposure_overlay_last_year.png)
![Rolling CAGR](../../reports/strategy_regime_trend/plots/rolling_cagr.png)
![Rolling drawdown](../../reports/strategy_regime_trend/plots/rolling_drawdown.png)
![Rolling alpha/beta](../../reports/strategy_regime_trend/plots/rolling_alpha_beta.png)
![Turnover distribution](../../reports/strategy_regime_trend/plots/turnover_hist.png)
![Cost sensitivity](../../reports/strategy_regime_trend/plots/cost_sensitivity.png)
![Equity curve net](../../reports/strategy_regime_trend/plots/equity_curve_net.png)
![Equity curve net (last year)](../../reports/strategy_regime_trend/plots/equity_curve_net_last_year.png)

Plot notes:
- Equity curve comparison: trend-only usually tops returns; regime-trend trades some return for smoother risk.
- Rolling drawdown: regime-trend drawdowns are consistently shallower than benchmark and trend-only.
- Cost sensitivity: net return and Sharpe decline roughly linearly with higher cost bps.

## Regime-Trend Sweep

- `reports/strategy_regime_trend_sweep/data/sweep_results.csv`
- `reports/strategy_regime_trend_sweep/data/top_candidates.csv`

Use these to review the hyperparameter rankings and see which configs were selected.
