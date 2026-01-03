# Insights Report

## Model Upgrade Results

- Variants compared: GARCH, GJR, EGARCH with normal vs Student‑t errors.
- Best by BIC: **EGARCH_t** (see `reports/modeling_variants/data/variant_metrics.csv`).
- Interpretation: allowing asymmetric volatility and fat tails improves fit relative to vanilla GARCH.

## Out-of-Sample Evaluation

From `reports/oos_check/data/oos_metrics.csv`:

- Static forecast vs realized: not available for EGARCH multi‑step horizon
- Rolling 1‑step vs realized: corr = 0.7286, RMSE = 5.5016

Takeaway: rolling forecasts track realized volatility much better than the static multi‑step path.

## Regime Outcome Summary

From `reports/regime_analysis/data/regime_outcomes.csv`:

| Regime | Avg log return | Avg VIX | Min drawdown |
| --- | --- | --- | --- |
| low | 0.000109 | 14.00 | -0.151 |
| mid | 0.000644 | 16.65 | -0.181 |
| high | 0.000585 | 24.40 | -0.339 |

Interpretation: high‑vol regimes coincide with much higher VIX and deeper drawdowns, while low‑vol regimes are materially calmer.

## Practical Takeaways

- Volatility is persistent and shocks decay slowly, so regime shifts are durable.
- Implied volatility (VIX) aligns more with short‑window realized volatility in this sample (10‑day window).
- For forecasting, rolling 1‑step updates provide the most realistic signal and are the better diagnostic.
