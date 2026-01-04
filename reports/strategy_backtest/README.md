# Regime Strategy Backtest

## Purpose

Evaluate a simple regime-aware equity exposure rule: higher exposure in low volatility regimes and lower exposure in high volatility regimes.

## Key Figures (Last 12 Months)

These two plots are the most important visuals for understanding the regime overlay in practice. They zoom into the most recent 12 months and rebase both equity lines to the same start point, so you can see relative performance and risk‑scaling behavior clearly.

The strategy uses the regime labels generated from the selected ARMA‑GARCH variant (see `reports/modeling/README.md` and `reports/modeling_variants/README.md`). Exposure levels are the mapping defined by the backtest (default low=1.0, mid=0.75, high=0.25), which represent the fraction of equity exposure in each regime.

**Use cases:**
- Compare short‑horizon risk control vs buy‑and‑hold under current market conditions.
- Validate that exposure steps down during high‑volatility regimes and back up in low‑volatility regimes.
- Communicate how the regime signal translates into concrete position sizing.

**Color and scale guide:**
- Black line: buy‑and‑hold equity curve (rebased for last‑year view).
- Slate‑blue line: regime strategy equity curve (rebased for last‑year view).
- Regime strip colors: green = low, amber = mid, red = high volatility.
- Exposure values (e.g., 1.0, 0.75, 0.25) are the fraction of full equity exposure.

![Equity curve (last year)](plots/equity_curve_last_year.png)

![Exposure overlay (last year)](plots/exposure_overlay_last_year.png)

## Outputs

- `data/summary.txt` for return/vol/sharpe/drawdown
- `data/strategy_equity.csv` for equity curve and drawdowns
- `data/regime_performance.csv` for regime-level performance
- `data/exposure_stats.csv` for exposure distribution
- `data/strategy_variants.csv` for Sharpe-optimized exposure comparisons
- `plots/equity_curve.png` for the strategy vs benchmark chart
- `plots/exposure_overlay.png` for exposure levels with regime shading
- `plots/equity_curve_last_year.png` for the last-year equity curve zoom
- `plots/exposure_overlay_last_year.png` for the last-year exposure/regime zoom

## Interpretation

- The equity curve compares the regime strategy to buy‑and‑hold; focus on drawdown depth and recovery speed, not just end value.
- This strategy underperforms buy‑and‑hold on raw return but improves drawdown and volatility, so it is a risk‑control tool.
- Use `data/strategy_variants.csv` to see whether alternative exposure maps improve Sharpe without materially hurting returns.
- If the objective is return maximization, buy‑and‑hold is the correct benchmark; if the objective is capital preservation and smoother risk, the regime overlay is the better fit.

## Summary Table

Populated after running the pipeline (from `data/summary.txt`):

| Metric | Value |
| --- | --- |
| Strategy annual return | 0.0618 |
| Strategy annual vol | 0.0872 |
| Strategy Sharpe | 0.7085 |
| Strategy max drawdown | -0.1234 |
| Buy-and-hold annual return | 0.1128 |
| Buy-and-hold annual vol | 0.1738 |
| Buy-and-hold Sharpe | 0.6486 |
| Buy-and-hold max drawdown | -0.3392 |
| Excess return vs buy-and-hold | -0.0510 |
| Best Sharpe exposure map | low=1.0, mid=1.0, high=0.5 |
| Best Sharpe | 0.7397 |

## Figures (Full Sample)

The full‑sample equity curve shows whether the strategy reduces drawdowns and volatility relative to buy‑and‑hold over the entire history.

![Equity curve](plots/equity_curve.png)

The full‑sample exposure overlay highlights how the regime labels drive risk scaling over time: exposure rises in low‑volatility regimes and falls in high‑volatility regimes.

![Exposure overlay](plots/exposure_overlay.png)
