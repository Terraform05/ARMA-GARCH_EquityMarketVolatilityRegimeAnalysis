# Modeling Variants Summary

## Key Results

- Best variant by BIC: EGARCH_t
- Variants compared: GARCH, GJR, EGARCH with normal vs Student‑t errors
- Metrics: see `data/variant_metrics.csv` and `data/variant_realized_metrics.csv`

## Interpretation

- EGARCH_t suggests asymmetric volatility and heavy tails improve fit.
- Variant comparisons help validate model selection beyond the baseline GARCH.
- QLIKE (quasi-likelihood) evaluates volatility forecasts using realized variance; lower values are better and it is robust to measurement noise.
- Set `VARIANT_SELECTION = "tracking"` in `src/config.py` to select the variant with the best realized‑vol tracking metrics.
- AIC/BIC measure fit to returns and residual structure, not alignment to a realized‑vol proxy; tracking metrics answer a different question (how well the model matches observed volatility levels).
- Decision rule: prefer BIC for in‑sample explanatory fit, prefer tracking metrics when the goal is risk monitoring or hedge‑cost alignment.

## Variant Metrics (vs Realized Volatility)

| Variant | Corr | RMSE | QLIKE |
| --- | --- | --- | --- |
| GARCH | 0.9587 | 3.0347 | 6.1994 |
| GARCH_t | 0.9297 | 3.5652 | 6.2014 |
| GJR | 0.8901 | 4.3646 | 6.2286 |
| GJR_t | 0.8901 | 4.3646 | 6.2286 |
| EGARCH_t | 0.8173 | 5.4161 | 6.2747 |
| EGARCH | 0.8081 | 5.5010 | 6.2790 |

Note: BIC favors EGARCH_t for in-sample fit, while RMSE/QLIKE favor GARCH for realized-vol tracking. Use this to decide between fit quality and tracking accuracy.

## Figures

![Variant comparison](plots/variant_comparison.png)

Plot notes:
- The top variants move together most of the time, which means model choice is not a huge swing factor in calm periods.
- Divergences around stress windows highlight where asymmetric or heavy-tail models react differently.
- If a model consistently sits above/below peers, it may be systematically over- or under-estimating volatility.

![Variants vs realized](plots/variant_vs_realized.png)

Plot notes:
- The closest line to realized volatility (orange) indicates the best tracking model.
- Look for phase alignment: peaks and troughs should occur at the same times, not just similar levels.
- Persistent gaps show scale bias; a model that tracks direction but misses level may need rescaling.

![Variant metrics](plots/variant_metrics.png)

Plot notes:
- Lower (more negative) AIC/BIC is better; focus on relative gaps, not absolute values.
- If two variants are close, prefer the one that also tracks realized volatility better.
- Large gaps in BIC justify a more complex model; tiny gaps usually do not.

![Best variant volatility](plots/best_variant_volatility.png)

Plot notes:
- The conditional volatility path should spike during known stress windows and mean-revert afterward.
- A smooth, lagged response suggests a model that is too slow for risk monitoring.
- Use this line as the baseline input to regime classification.

![BIC vs tracking](plots/bic_vs_tracking.png)

Plot notes:
- Points with lower BIC and lower RMSE are ideal but often do not exist simultaneously.
- A model with strong BIC but weak RMSE is good for in-sample fit, not for tracking.
- This tradeoff plot is the fastest way to decide whether to optimize for fit or for tracking.
