# ARMA + GARCH Modeling

## Objective

Model return dynamics with ARMA and conditional volatility with GARCH(1,1).

## ARMA Mean Process

1. Fit candidate ARMA(p, q) models on returns.
2. Select p, q using AIC/BIC.
3. Check residuals for autocorrelation.

## GARCH Variance Process

1. Fit GARCH(1,1) on ARMA residuals.
2. Report parameters:
   - `ω` (long-run variance)
   - `α` (shock reaction)
   - `β` (persistence)
   - `α + β` (overall persistence)

## Interpretation Notes

- High `α + β` indicates persistent volatility shocks.
- Compare conditional volatility with realized volatility proxy.

## Output

- Model summaries.
- Parameter tables (mean and variance).
- Conditional volatility series.
