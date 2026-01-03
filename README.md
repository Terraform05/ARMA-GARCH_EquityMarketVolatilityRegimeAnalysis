# ARMA-GARCH_EquityMarketVolatilityRegimeAnalysis

Model and interpret volatility regimes in equity markets by separating return dynamics from volatility dynamics, and assess how risk evolves across market conditions.

## Project Goal

This project focuses on regime interpretation, not trading or alpha generation. The core questions are:

- When is the market calm vs stressed?
- How persistent are volatility shocks?
- How does implied volatility (VIX) compare to realized volatility?
- What does this mean for risk and valuation confidence?

## Scope

- Underlying: S&P 500 Index returns (SPX)
- Volatility benchmark: VIX Index (implied volatility)
- Horizon: 10–20+ years with low-vol, crisis, and transitional regimes

## Documentation

- [Data preparation](docs/01_data_prep.md)
- [Preliminary diagnostics](docs/02_diagnostics.md)
- [ARMA + GARCH modeling](docs/03_modeling.md)
- [Model validation](docs/04_validation.md)
- [Regime interpretation](docs/05_regime_analysis.md)
- [Out-of-sample check](docs/06_oos_check.md)
- [Executive summary](reports/summary.md)

## Expected Outputs

- Plots of returns, conditional volatility, and VIX vs realized volatility
- 1–2 page interpretation summary
- Reproducible, documented workflow
