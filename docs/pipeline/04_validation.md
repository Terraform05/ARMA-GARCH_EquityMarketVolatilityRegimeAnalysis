# Model Validation

## Objective

Confirm adequacy of the mean and variance specifications using residual diagnostics.

## Inputs

- Use the fitted ARMA + GARCH model from `docs/pipeline/03_modeling.md`.
- Date range: 2010-01-01 to 2026-01-01.
- Required series: standardized residuals and standardized squared residuals.

## Residual Diagnostics

- Ljung–Box test on standardized residuals.
- Ljung–Box test on squared standardized residuals.
- Check remaining ARCH effects.

## Expected Outcomes

- No significant autocorrelation in standardized residuals.
- Minimal remaining ARCH effects after GARCH fit.

## Output

- Test statistics and p-values.
- Summary narrative on model adequacy.
- Clear pass/fail note for each diagnostic.
- Residual diagnostics plots (series, ACF, Q-Q).

## Implementation

- Run `python scripts/run_validation.py` after modeling.
- If model variants have been evaluated, validation uses the best variant by BIC.
