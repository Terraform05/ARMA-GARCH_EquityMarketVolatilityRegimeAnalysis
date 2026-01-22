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

![Returns and squared returns](plots/returns_series.png)

Plot notes:
- The squared-return series clusters into bursts, showing volatility clustering that a constant-variance model would miss.
- Large spikes align with stress windows; the persistence after spikes supports a GARCH-style variance process.
- The raw return series centers near zero with occasional shocks, consistent with a weak mean signal.

![ACF and PACF](plots/acf_pacf.png)

Plot notes:
- Return ACF/PACF are small beyond short lags, indicating limited linear predictability in the mean.
- Squared-return ACF decays more slowly, which signals persistent volatility dynamics.
- The quick decay in returns but persistence in squared returns is the classic ARMA+GARCH use case.
