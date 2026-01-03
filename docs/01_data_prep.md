# Data Preparation

## Objective

Deliver a clean, aligned time series of SPX prices and VIX levels with derived return and realized volatility proxies.

## Data Sources

- SPX prices (daily close): Yahoo Finance or equivalent
- VIX index (daily close): Yahoo Finance or FRED

## Data Alignment

1. Load SPX and VIX daily closes.
2. Align on shared trading dates.
3. Handle missing values:
   - Drop non-overlapping days.
   - Document any forward-fill logic if used.

## Derived Series

- Log returns (SPX):
  - Formula: `r_t = log(P_t / P_{t-1})`
- Realized volatility proxy:
  - Squared returns: `r_t^2`

## Visual Checks

- Price level series (SPX)
- Return series
- VIX levels

## Output

- A single aligned dataset (CSV/parquet) with columns such as:
  - `date`, `spx_close`, `vix_close`, `log_return`, `sq_return`

## Notes & Caveats

- Confirm the time horizon includes multiple volatility regimes.
- Document any data quirks (e.g., missing VIX dates, stale closes).
