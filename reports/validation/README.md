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

The residual series shows remaining structure over time after the model fit.

![Residual series](plots/residuals_series.png)

ACF plots reveal any remaining autocorrelation in residuals and squared residuals.

![Residual ACF](plots/residuals_acf.png)

The Q‑Q plot checks whether residuals follow the assumed distribution.

![Residual Q-Q](plots/residuals_qq.png)
