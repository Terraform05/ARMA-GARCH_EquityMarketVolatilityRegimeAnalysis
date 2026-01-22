# Concepts

This document explains what the project is doing, why the modeling choices were made, and how the outputs should be interpreted.

## Project Objective

The core objective is to understand equity-market volatility regimes and how those regimes change risk, then evaluate strategy overlays that respond to those regimes. This is not just about trading performance; it is about understanding risk states and how exposure should adapt when volatility regimes shift.

Key questions:
- When is the market calm vs stressed?
- How persistent are volatility shocks?
- How does implied volatility (VIX) compare to realized volatility?
- Do regime-aware strategies reduce drawdown and improve risk-adjusted returns?

## Conceptual Framework: What Is a Market Regime?

A market regime is a statistically persistent state of market behavior characterized by distinct volatility dynamics. Regimes are descriptive, not predictive: they summarize how markets behave over time rather than forecast price direction.

Key distinctions:
- Regimes describe risk environments, not price direction.
- Regimes persist but can transition abruptly.
- Volatility is regime-dependent, not constant.
- Volatility clustering means large moves tend to be followed by large moves (and small by small), which is why regimes persist.

## Why ARMA + GARCH?

ARMA and GARCH target two different properties of financial returns and keep signal separate from risk:

ARMA (mean): removes short-run predictable structure in returns. This yields shocks that are closer to “unexpected” moves.

GARCH (variance): models time-varying volatility and persistence using those shocks. Volatility responds to unexpected news rather than predictable patterns.

If ARMA is skipped, predictable structure can leak into the volatility model and inflate risk estimates. Together, ARMA + GARCH isolate regime-driven variance behavior so regime analysis focuses on volatility rather than price direction.

## Regime Exposure: The Core Insight

Regime labels alone are insufficient. The key variable is regime exposure — how concentrated the market is in one regime versus dispersed across multiple regimes.

High regime exposure:
- Strong market consensus
- Capital anchored to a dominant volatility regime
- Stable risk pricing
- Typically lower realized volatility

Low regime exposure:
- Weak regime conviction
- Uncertainty about the prevailing risk environment
- Continuous repricing of risk
- Typically higher realized volatility, especially during transitions

Why transitions matter more than steady states:
- Transitions are where exposure is weakest and volatility spikes.
- Stable regimes show compressed volatility and slower repricing.

## Relationship Between Model Volatility and VIX

Model-implied volatility and VIX play complementary roles:
- GARCH volatility reflects realized and conditional variance.
- VIX reflects market-implied expectations of future volatility.

Divergences often occur during regime transitions, when expectations adjust faster than realized outcomes (or vice versa). These gaps are informative about risk pricing and hedge costs.

## What Gets Produced

The pipeline produces:
- Aligned SPX/VIX dataset (`data/processed/spx_vix_aligned.csv`)
- Model variant diagnostics and selection outputs (`reports/modeling_variants/`)
- Conditional volatility from the selected model (`reports/modeling/`)
- Validation diagnostics (`reports/validation/`)
- Regime labeling and regime summary tables (`reports/regime_analysis/`)
- Out-of-sample diagnostics (`reports/oos_check/`)
- Hedge cost monitoring (`reports/hedge_monitoring/`)
- Strategy backtests (`reports/strategy_backtest/` and `reports/strategy_regime_trend/`)
- Hyperparameter sweeps for regime-trend (`reports/strategy_regime_trend_sweep/`)

## Model Variant Selection (BIC, Tracking, Hybrid)

Variant selection is controlled in `src/config.py`:

- `VARIANT_SELECTION = "bic"`: best in-sample fit by BIC
- `VARIANT_SELECTION = "tracking"`: best realized-vol tracking (RMSE)
- `VARIANT_SELECTION = "hybrid"`: tracking-first, prefer lower BIC if RMSE is close

The model-variants step writes `reports/modeling_variants/data/best_variant.txt` with all three modes and the active selection.

## Strategy Layering: Regime + Trend

Two strategies are implemented:

1) Regime-only strategy
   - Exposure is reduced in high-volatility regimes and increased in low-volatility regimes.
   - Primary objective: drawdown control.

2) Regime-trend strategy
   - Adds a directional trend layer on top of volatility regimes.
   - Exposures are set by a regime x trend matrix or a continuous sizing rule.
   - Objective: improve risk-adjusted returns while controlling drawdown.

The regime-trend sweep optimizes hyperparameters with walk-forward evaluation and writes a frozen best config to `configs/regime_trend_best.json`.

## Evaluation Focus

- Total return vs benchmark
- Max drawdown and recovery time
- Risk-adjusted metrics: Sharpe, Sortino, Calmar
- Turnover and transaction cost sensitivity
- Stability across walk-forward windows

## Practical Applications

- Risk management and stress testing
- Volatility targeting and leverage control
- Identifying volatility selling vs buying environments
- Detecting early signs of regime instability

## Limitations

- Regimes are backward-looking.
- GARCH captures conditional variance, not tail risk.
- Structural breaks can disrupt regime persistence.
- Regime identification depends on model assumptions.

## Where to Read Next

- Plots and interpretations: `docs/readmes/plots.md`
- Math and formulas: `docs/readmes/math.md`
- Reports index: `reports/README.md`
