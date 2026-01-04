# Regime Analysis Summary

## Key Results

- Low regime threshold (<=): 0.006721
- High regime threshold (>=): 0.010107
- Realized volatility window: 10-day (annualized), selected by highest correlation to VIX

## Interpretation

- Periods below the low threshold represent relatively calm volatility conditions; mid‑regime points are transitional.
- Periods above the high threshold indicate elevated volatility regimes with sustained risk.
- Compare these regime stretches with known stress episodes (2010-2012 Eurozone stress, 2015-2016 selloff, 2018 vol spike, 2020 COVID crash, 2022 tightening, 2024-2025 shifts).
- The 10-day realized window makes the implied vs realized comparison more sensitive to short‑run volatility shifts.
- Operationally, high‑regime windows align with drawdown risk and higher hedge costs; low‑regime windows align with more stable risk conditions and lower hedging pressure.
- A shorter realized window reacts faster but is noisier; a longer window is smoother but can lag turning points. The 10‑day choice balances responsiveness with stability for this sample.
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
