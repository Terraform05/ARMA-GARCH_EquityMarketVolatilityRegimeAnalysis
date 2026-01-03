# Preliminary Diagnostics

## Objective

Justify ARMA for the mean and GARCH for the variance using stationarity and autocorrelation diagnostics.

## Inputs

- Use the aligned CSV produced in `docs/01_data_prep.md`.
- Date range: 2010-01-01 to 2026-01-01.
- Required columns: `date`, `spx_adj_close`, `vix_close`, `log_return`, `sq_return`.

## Stationarity

- Run Augmented Dickey-Fuller (ADF) test on SPX log returns.
- Record the test statistic, p-value, and conclusion.

## Autocorrelation

- Plot ACF/PACF for returns.
- Identify candidate ARMA(p, q) orders for the mean process.

## ARCH Effects

- Run an ARCH test on returns or squared returns.
- Record the test statistic, p-value, and conclusion.

## Volatility Clustering

- Plot returns and squared returns.
- Highlight periods of clustering to motivate conditional heteroskedasticity.

## Output

- Diagnostic plots (ACF/PACF, return series, squared returns).
- Short narrative linking diagnostics to model choice.
- A brief note summarizing ADF and ARCH test results.

## Implementation

- Run `python scripts/run_diagnostics.py` after the CSV exists.
