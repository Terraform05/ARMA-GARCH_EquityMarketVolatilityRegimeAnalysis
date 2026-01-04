# Hedge + Strategy Overview

## Purpose

Provide a single place to review hedge‑cost monitoring signals and the regime‑aware equity strategy backtest.

## Hedge Monitoring Summary

See `reports/hedge_monitoring/README.md` for thresholds, signals, and plots.

## Strategy Backtest Summary

See `reports/strategy_backtest/README.md` for equity curve and performance stats.

## Tables

These tables are populated after running the pipeline:

**Hedge monitoring summary (from `reports/hedge_monitoring/data/summary.txt`)**

| Metric | Value |
| --- | --- |
| Ratio low threshold | 1.056 |
| Ratio high threshold | 1.925 |
| Cheap % | 20.00% |
| Neutral % | 59.99% |
| Expensive % | 20.00% |
| Avg cheap duration (days) | 4.6 |
| Avg neutral duration (days) | 7.8 |
| Avg expensive duration (days) | 5.9 |

**Strategy performance (from `reports/strategy_backtest/data/summary.txt`)**

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

## Takeaways

- The hedge‑cost signal is most useful as a budgeting/position‑sizing guide, not a standalone timing tool.
- The regime strategy reduces drawdowns and volatility, but it gives up return vs buy‑and‑hold.
- If the goal is risk control, the strategy has value; if the goal is maximum return, buy‑and‑hold is stronger in this sample.
