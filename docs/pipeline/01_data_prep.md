# Data Preparation

## Objective

Deliver a clean, aligned time series of SPX prices and VIX levels with derived return and realized volatility proxies.

## Data Sources

- SPX prices (daily EOD): Yahoo Finance (`^GSPC`)
- VIX index (daily EOD): Yahoo Finance (`^VIX`)

## Date Range

- Start: 2010-01-01
- End: 2026-01-01

## Data Alignment

1. Load SPX and VIX daily closes.
2. Use SPX trading days as the NYSE calendar.
3. Align VIX to the SPX calendar and drop any dates with missing values.
4. Handle missing values:
   - Drop non-overlapping days.
   - Document any forward-fill logic if used.

## Derived Series

- Log returns (SPX):
  - Formula: `r_t = log(P_t / P_{t-1})` using SPX adjusted close
- Realized volatility proxy:
  - Squared returns: `r_t^2`

## Visual Checks

- Price level series (SPX)
- Return series
- VIX levels

## Output

- A single aligned dataset (CSV) with columns such as:
  - `date`, `spx_adj_close`, `vix_close`, `log_return`, `sq_return`

## Implementation

- Run `python scripts/prepare_data.py` to download and build the aligned CSV.
- Or import and call the helper:

```python
from scripts.prepare_data import run_data_prep

run_data_prep()
```

## Notes & Caveats

- Confirm the time horizon includes multiple volatility regimes.
- Document any data quirks (e.g., missing VIX dates, stale closes).
