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
- Operationally, high‑regime windows align with drawdown risk and higher hedge costs; low‑regime windows align with more stable risk conditions and lower hedging pressure.
- A shorter realized window reacts faster but is noisier; a longer window is smoother but can lag turning points. The 10‑day choice balances responsiveness with stability for this sample.
- Use `data/realized_window_metrics.csv` to compare correlation and RMSE across candidate windows.
- `plots/realized_window_metrics.png` visualizes those metrics for quick selection.
- `data/regime_outcomes.csv` summarizes average returns, VIX, and drawdowns by regime.
- `regime_outcomes.png` visualizes those outcomes.

## Figures

![Regime scatter](plots/regimes.png)

Plot notes:
- The scatter bands show discrete regime blocks; long blocks indicate persistent volatility states.
- Rapid color flipping signals short-lived volatility spikes that can whipsaw exposure.
- Use this plot to sanity check whether regime changes align with known stress episodes.

![VIX vs realized](plots/vix_vs_realized.png)

Plot notes:
- VIX should generally lead or move in the same direction as realized volatility.
- Persistent gaps imply the realized window is too slow (lags VIX) or too fast (overreacts).
- Alignment supports using the chosen realized window for regime labeling.

![Window metrics](plots/realized_window_metrics.png)

Plot notes:
- Correlation tends to improve with shorter windows but RMSE can worsen from noise.
- The chosen window should sit near a stable tradeoff, not at an unstable extreme.
- Use this to justify the selected window length in the pipeline.

![Regime outcomes](plots/regime_outcomes.png)

Plot notes:
- High-vol regimes should show larger drawdowns and lower returns if the labeling is sensible.
- Low-vol regimes should show higher average returns and lower drawdowns.
- If outcomes look similar across regimes, the regime split is not informative.
