# Validation Summary

## Key Results

- Ljung-Box on standardized residuals: statistic = 58.770944, p-value = 0.000011
- Ljung-Box on squared standardized residuals: statistic = 38.080963, p-value = 0.008656
- ARCH test on standardized residuals: statistic = 34.095982, p-value = 0.025483
- Lags used: 20

## Interpretation

- Residual autocorrelation is still present at lag 20 (p < 0.001), indicating the mean dynamics are not fully captured.
- Squared residuals and the ARCH test remain significant (p ≈ 0.0087 and p ≈ 0.0255), indicating residual volatility structure.
- This supports using the model for regime labeling and risk monitoring, not precision forecasting.

## Figures

![Residual series](plots/residuals_series.png)

Plot notes:
- Residual spikes cluster around stress windows, indicating remaining structure after the fit.
- The variance is not constant over time, which matches the significant ARCH test.
- Use this to confirm the model is adequate for regimes, not for precise return prediction.

![Residual ACF](plots/residuals_acf.png)

Plot notes:
- Autocorrelation in residuals indicates the mean process is not fully captured.
- Autocorrelation in squared residuals confirms lingering volatility structure.
- If these tails were flat and near zero, the variance model would be closer to fully specified.

![Residual Q-Q](plots/residuals_qq.png)

Plot notes:
- Tail deviations from the diagonal indicate heavy tails relative to the assumed distribution.
- A strong S-shape suggests skew or kurtosis that a normal error model misses.
- This supports using a Student-t error distribution in variant testing.
