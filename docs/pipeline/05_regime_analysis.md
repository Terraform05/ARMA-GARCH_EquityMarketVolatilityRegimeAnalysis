# Regime Interpretation

## Objective

Interpret volatility regimes and compare implied vs realized volatility.

## Inputs

- Use conditional volatility series from `docs/pipeline/03_modeling.md`.
- Use aligned CSV from `docs/pipeline/01_data_prep.md`.
- Date range: 2010-01-01 to 2026-01-01.

## Conditional Volatility Regimes

- Plot conditional volatility over time.
- Identify low-, high-, and transitional volatility regimes.
- Align regimes with known market stress events.

## Implied vs Realized Volatility

- Compare VIX to annualized realized volatility using a window selected for sensitivity (e.g., 10/21/63 days).
- Highlight periods where VIX over- or understates realized risk.

## Interpretation Prompts

- What do persistent high-vol regimes imply for risk perception?
- When does implied volatility diverge from realized risk?

## Output

- Regime plots.
- VIX vs realized volatility plot.
- Short narrative interpretation.
- Note any regimes that align with 2010-2012 Eurozone stress, 2015-2016 selloff, 2018 vol spike, 2020 COVID crash, 2022 inflation tightening, and 2024-2025 volatility shifts.

## Implementation

- Run `python scripts/run_regime_analysis.py` after modeling.
