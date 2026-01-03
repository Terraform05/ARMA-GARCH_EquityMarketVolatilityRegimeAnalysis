# Out-of-Sample Check

## Objective

Perform a lightweight sanity check on volatility forecasts.

## Train/Test Split

- Hold out the last 1–2 years of data.
- Fit ARMA–GARCH on the training set.

## Forecast Evaluation

- Generate conditional volatility forecasts over the holdout period.
- Compare forecasts to realized volatility proxy.

## Notes

- This is a validation step, not optimization.
- Document limitations and potential improvements.

## Output

- Forecast vs realized plot or table.
- Brief summary of forecast adequacy.
