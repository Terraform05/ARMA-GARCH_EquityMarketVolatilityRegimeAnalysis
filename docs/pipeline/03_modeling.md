# ARMA + GARCH Modeling

## Objective

Model return dynamics with ARMA and conditional volatility with GARCH(1,1).

## Inputs

- Use the aligned CSV produced in `docs/pipeline/01_data_prep.md`.
- Date range: 2010-01-01 to 2026-01-01.
- Required columns: `date`, `log_return`, `sq_return`.

## ARMA Mean Process

1. Fit candidate ARMA(p, q) models on returns.
2. Select p, q using AIC/BIC.
3. Check residuals for autocorrelation.

## GARCH Variance Process

1. Fit GARCH(1,1) on ARMA residuals.
2. Validate that residuals show reduced ARCH effects.
3. Report parameters:
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
- Short note comparing conditional volatility to `sq_return`.

## Implementation

- Run `python scripts/run_modeling.py` after diagnostics.
- Optional: run `python scripts/run_model_variants.py` to compare GARCH variants (GARCH, GJR, EGARCH) with normal vs Student-t errors.
- If model variants have been evaluated, `run_modeling.py` will use the best variant by BIC.
