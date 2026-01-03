# Validation Summary

## Key Results

- Ljung-Box on standardized residuals: statistic = 52.098777, p-value = 0.000110
- Ljung-Box on squared standardized residuals: statistic = 14.688354, p-value = 0.793954
- ARCH test on standardized residuals: statistic = 14.387933, p-value = 0.810295
- Lags used: 20

## Interpretation

- Residual autocorrelation is still present at lag 20 (p < 0.001), indicating the mean dynamics are not fully captured.
- Squared residuals no longer show significant dependence (p ≈ 0.79), indicating improved volatility fit.
- The ARCH test is no longer significant (p ≈ 0.81), suggesting the variance model captures clustering well.
- Remaining improvement opportunity is mainly in the mean equation.

## Figures

The residual series shows remaining structure over time after the model fit.

![Residual series](plots/residuals_series.png)

ACF plots reveal any remaining autocorrelation in residuals and squared residuals.

![Residual ACF](plots/residuals_acf.png)

The Q‑Q plot checks whether residuals follow the assumed distribution.

![Residual Q-Q](plots/residuals_qq.png)
