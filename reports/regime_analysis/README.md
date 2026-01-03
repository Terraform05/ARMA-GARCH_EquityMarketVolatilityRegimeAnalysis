# Regime Analysis Summary

## Key Results

- Low regime threshold (<=): 0.007292
- High regime threshold (>=): 0.009889
- Realized volatility window: 10-day (annualized), selected by highest correlation to VIX

## Interpretation

- Periods below the low threshold represent relatively calm volatility conditions; mid‑regime points are transitional.
- Periods above the high threshold indicate elevated volatility regimes with sustained risk.
- Compare these regime stretches with known stress episodes (2010-2012 Eurozone stress, 2015-2016 selloff, 2018 vol spike, 2020 COVID crash, 2022 tightening, 2024-2025 shifts).
- The 10-day realized window makes the implied vs realized comparison more sensitive to short‑run volatility shifts.
- Use `data/realized_window_metrics.csv` to compare correlation and RMSE across candidate windows.
- `plots/realized_window_metrics.png` visualizes those metrics for quick selection.
- `data/regime_outcomes.csv` summarizes average returns, VIX, and drawdowns by regime.
- `regime_outcomes.png` visualizes those outcomes.

## Figures

The regime scatter shows low/mid/high volatility states across time.

![Regime scatter](plots/regimes.png)

This plot compares implied volatility (VIX) to realized volatility on the same scale.

![VIX vs realized](plots/vix_vs_realized.png)

Window metrics show how the realized volatility window choice affects alignment with VIX.

![Window metrics](plots/realized_window_metrics.png)

Regime outcomes summarize return, VIX, and drawdown differences by regime.

![Regime outcomes](plots/regime_outcomes.png)
