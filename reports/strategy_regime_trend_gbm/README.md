# GBM Forward Test (Regime-Trend Strategy)

## Purpose

Simulate future paths using geometric Brownian motion (GBM) to stress-test the
regime-trend strategy, the regime-only baseline, and buy-and-hold over 1/3/5-year
horizons.

This is a rough sanity check, not a realistic market model. GBM does not model
volatility clustering, fat tails, or regime switching.

## Method

1) Fit GBM parameters on historical log returns:
   - Use daily mean and standard deviation from the training window.

2) Simulate future log returns:
   - `log_return = mean_daily + std_daily * Z`
   - `Z ~ N(0, 1)`

3) Create synthetic prices:
   - `price_t = price_0 * exp(cumsum(log_return))`

4) Regime labels for the simulation:
   - Compute rolling realized volatility (window from training data).
   - Use low/high thresholds from the training realized-vol quantiles (33/66).
   - Regime-only baseline uses exposure map low=1.0, mid=0.75, high=0.25.

5) Regime-Trend strategy:
   - Same trend window/threshold and exposure matrix as the sweep winner.
   - No retraining inside the simulated horizon (forward-only).

## Retraining Policy

No retraining during the simulated horizon. Parameters are fixed based on the
training window. This avoids any look-ahead into simulated data.

## Baselines Included

- Buy-and-hold
- Regime-only strategy
- Regime-Trend strategy

## Outputs

Data:
- `data/gbm_run_summary.csv` run configuration summary
- `data/gbm_path_metrics_1y.csv` per-path metrics (1Y)
- `data/gbm_path_metrics_3y.csv` per-path metrics (3Y)
- `data/gbm_path_metrics_5y.csv` per-path metrics (5Y)
- `data/gbm_summary_1y.csv` aggregated stats (1Y)
- `data/gbm_summary_3y.csv` aggregated stats (3Y)
- `data/gbm_summary_5y.csv` aggregated stats (5Y)

Plots:
- `plots/gbm_return_dist_1y.png`
- `plots/gbm_return_dist_3y.png`
- `plots/gbm_return_dist_5y.png`
- `plots/gbm_drawdown_dist_1y.png`
- `plots/gbm_drawdown_dist_3y.png`
- `plots/gbm_drawdown_dist_5y.png`
- `plots/gbm_alpha_dist_1y.png`
- `plots/gbm_alpha_dist_3y.png`
- `plots/gbm_alpha_dist_5y.png`
- `plots/gbm_outperformance_1y.png`
- `plots/gbm_outperformance_3y.png`
- `plots/gbm_outperformance_5y.png`

## Plot-by-Plot Interpretation (Detailed)

### `plots/gbm_return_dist_1y.png`
![GBM annual return distribution (1Y)](plots/gbm_return_dist_1y.png)

Plot notes:
- Shows the 1‑year annual return distribution across simulated paths.
- A rightward shift of the regime-trend distribution vs benchmark indicates a higher
  likelihood of outperformance in GBM paths.
- Regime-only clustering below benchmark suggests conservative exposure under
  a constant‑volatility model.

### `plots/gbm_return_dist_3y.png`
![GBM annual return distribution (3Y)](plots/gbm_return_dist_3y.png)

Plot notes:
- Longer horizon tightens return dispersion compared with 1Y.
- Regime-Trend staying right‑shifted suggests its edge persists beyond short windows.
- If regime-trend and benchmark overlap heavily here, the edge may be short‑term only.

### `plots/gbm_return_dist_5y.png`
![GBM annual return distribution (5Y)](plots/gbm_return_dist_5y.png)

Plot notes:
- The 5Y horizon emphasizes compounding differences between strategies.
- A persistent right shift for regime-trend indicates robust multi‑year advantage.
- A left shift would imply the trend layer fails under GBM assumptions.

### `plots/gbm_drawdown_dist_1y.png`
![GBM max drawdown distribution (1Y)](plots/gbm_drawdown_dist_1y.png)

Plot notes:
- Compares 1Y max drawdown distributions across strategies.
- Regime-Trend skewed toward smaller drawdowns implies better risk control.
- Regime-only should sit between regime-trend and benchmark if it is effective.

### `plots/gbm_drawdown_dist_3y.png`
![GBM max drawdown distribution (3Y)](plots/gbm_drawdown_dist_3y.png)

Plot notes:
- Multi‑year drawdowns grow larger in magnitude for all strategies.
- Regime-Trend maintaining a tighter drawdown distribution is evidence of regime
  throttling even under GBM.
- If benchmark and regime-trend overlap, risk control may be path‑dependent.

### `plots/gbm_drawdown_dist_5y.png`
![GBM max drawdown distribution (5Y)](plots/gbm_drawdown_dist_5y.png)

Plot notes:
- Drawdown dispersion typically widens with horizon length.
- Regime-Trend’s tail should be less extreme than benchmark if the exposure map is
  effective in simulated stress.
- Regime-only helps isolate the benefit of volatility regimes alone.

### `plots/gbm_alpha_dist_1y.png`
![GBM alpha distribution (1Y)](plots/gbm_alpha_dist_1y.png)

Plot notes:
- 1Y alpha distribution shows the probability of positive annualized alpha.
- A higher mass above zero for regime-trend means frequent outperformance vs beta.
- Regime-only centered near zero indicates limited alpha generation.

### `plots/gbm_alpha_dist_3y.png`
![GBM alpha distribution (3Y)](plots/gbm_alpha_dist_3y.png)

Plot notes:
- Alpha distribution should narrow with longer windows.
- A positive shift for regime-trend suggests durable alpha under GBM.
- If the distribution centers at zero, alpha is not robust to GBM assumptions.

### `plots/gbm_alpha_dist_5y.png`
![GBM alpha distribution (5Y)](plots/gbm_alpha_dist_5y.png)

Plot notes:
- Long‑horizon alpha highlights whether the strategy adds value after compounding.
- A positive median implies structural edge; a symmetric distribution implies
  no meaningful alpha under GBM.

### `plots/gbm_outperformance_1y.png`
![GBM outperformance (1Y)](plots/gbm_outperformance_1y.png)

Plot notes:
- Shows the share of paths where strategy annual return beats benchmark.
- Regime-Trend above 0.5 indicates more than half the paths outperform.
- Regime-only below 0.5 indicates its defensive bias under GBM.

### `plots/gbm_outperformance_3y.png`
![GBM outperformance (3Y)](plots/gbm_outperformance_3y.png)

Plot notes:
- Outperformance probability should stabilize with longer horizon.
- A high regime-trend bar indicates robust performance persistence.
- If it drops below 0.5, short‑term gains may be mean‑reverting.

### `plots/gbm_outperformance_5y.png`
![GBM outperformance (5Y)](plots/gbm_outperformance_5y.png)

Plot notes:
- Long‑horizon outperformance is the most stringent sanity check.
- Regime-Trend staying above 0.5 suggests the edge is not path‑specific.
- Benchmark dominance here would imply GBM paths favor pure beta.

## How to Run

```
python scripts/run_gbm_forward_test.py
```

## Interpretation Notes

- If regime-trend dominates benchmark across all horizons in GBM, it suggests the
  edge is not dependent on a single realized path.
- If regime-only performs similarly to regime-trend in GBM, it suggests trend is not
  adding much in this simplified environment.
- If outperformance collapses at longer horizons, the edge might be short-term
  and heavily dependent on trend responsiveness.

## Limitations

- GBM assumes constant volatility and no regime shifts.
- This conflicts with the regime-driven design of the strategy.
- Use GBM as a sanity check only; rely on walk-forward OOS for realism.
