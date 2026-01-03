# Out-of-Sample Check

## Objective

Perform a lightweight sanity check on volatility forecasts.

## Inputs

- Use the aligned CSV from `docs/01_data_prep.md`.
- Use the model specification selected in `docs/03_modeling.md`.
- Date range: 2010-01-01 to 2026-01-01.

## Train/Test Split

- Hold out the last 2 years of data (2024-01-01 to 2026-01-01).
- Fit ARMA–GARCH on the training set.

## Forecast Evaluation

- Generate conditional volatility forecasts over the holdout period.
- Compare forecasts to 21-day realized volatility (annualized) so units match.

## Notes

- This is a validation step, not optimization.
- Document limitations and potential improvements.

## Output

- Forecast vs realized plot or table.
- Brief summary of forecast adequacy.

## Implementation

- Run `python scripts/run_oos_check.py` after regime analysis.
- If model variants have been evaluated, the OOS step uses the best variant by BIC.
