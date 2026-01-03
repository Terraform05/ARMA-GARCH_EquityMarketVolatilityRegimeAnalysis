# Diagnostics Summary

## Key Results

- ADF test on `log_return`: statistic = -13.733800, p-value < 0.001
- ARCH test on `log_return`: statistic = 1270.026392, p-value < 0.001
- Lags used: 20

## Interpretation

- The ADF result strongly rejects a unit root, so SPX log returns are stationary and appropriate for ARMA modeling.
- The ARCH test is highly significant, indicating volatility clustering and time‑varying variance that a constant‑variance model would miss.
- Together, these diagnostics justify modeling the mean with ARMA and the variance with GARCH.

## Figures

The return and squared-return plot shows volatility clustering, which motivates time‑varying variance models.

![Returns and squared returns](plots/returns_series.png)

ACF/PACF indicate short‑lag structure in returns and guide ARMA order selection.

![ACF and PACF](plots/acf_pacf.png)
