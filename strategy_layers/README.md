# Strategy Layers (Concept Map)

Goal: beat S&P 500 buy-and-hold on total return, not just reduce drawdowns.

This project uses a two-layer model:

- Layer 1: Risk regime (existing ARMA-GARCH volatility regimes).
- Layer 2: Directional return signal (new layer mapped below).

## Layer 2 Map: Directional Return Signal

### Purpose
Provide a positive-return engine that can be scaled up or down by the risk regime.
Volatility regimes alone reduce risk but rarely improve CAGR versus buy-and-hold.

### Core design principles
- Keep the directional layer simple enough to be robust across decades.
- Use the risk regime to control exposure size, not to time direction itself.
- Prefer signals with economic intuition (trend, momentum, breakouts) and low overfitting risk.
- Favor fewer knobs, then add complexity only if there is a clear failure mode.

### Definitions and inputs
- Price series: use total return if available; otherwise price-only with dividend caveats.
- Frequency: monthly or weekly to reduce noise and turnover.
- Signal window: 6-12 months for trend, 3-12 months for breakout bands.
- Regime window: based on the existing ARMA-GARCH regime assignments.

### Candidate signals (choose one to start)
- Trend filter: 6-12 month moving-average slope or MA cross.
- Time-series momentum: 12-month total return sign.
- Breakout regime: price above/below rolling channel (e.g., 6-month high/low).

### Directional states (more granular)
Define a trend strength score (e.g., slope of 6-12 month MA, or 12-month return z-score).

Example buckets:
- Strong uptrend: score >= +0.5
- Weak/neutral: -0.5 < score < +0.5
- Strong downtrend: score <= -0.5

### Trend strength score details
- Slope-based: compute slope of a 6-12 month moving average, then standardize it.
- Momentum-based: use 12-month return minus 1-month return (to avoid short-term mean reversion).
- Normalization: convert to z-score so thresholds are stable across time.
- Stability check: require persistence (e.g., 2 consecutive periods) before switching buckets.

### Combine layers (decision matrix)
Risk regime (Layer 1) x Directional state (Layer 2) -> exposure policy

Example 9-cell policy (exposure in % of equity):

| Vol \ Trend | Strong uptrend | Weak/neutral | Strong downtrend |
| --- | --- | --- | --- |
| Low vol | 110% | 80% | 50% |
| Mid vol | 90% | 60% | 30% |
| High vol | 70% | 40% | 10% |

Notes:
- Over 100% is optional (only if leverage is allowed).
- The worst cells can pair with an options overlay.

### Continuous sizing (alternative to buckets)
Instead of discrete cells, map exposure to a smooth function:
- Start with a base exposure.
- Add trend strength contribution.
- Subtract volatility contribution.

Example rule (conceptual):
exposure = clamp(base + a * trend_score - b * vol_score, min, max)

This keeps granularity without exploding the state grid.

### Risk regime details (Layer 1 input)
- Use the existing low/mid/high volatility regimes from the ARMA-GARCH layer.
- Regime definitions should be fixed out of sample (no peeking).
- If regime definitions drift over time, re-estimate only at predefined intervals (e.g., yearly).
- Treat regime transitions as risk events: reduce exposure when regime is unstable.

### Decision matrix refinements
- Add a "transition penalty" when regime changes in the last N periods.
- Add a "signal confidence" modifier to reduce exposure when trend score is weak.
- Cap maximum exposure in high volatility regimes even if trend is strong.
- Require a minimum holding period to avoid whipsawing in choppy markets.

### Exposure sizing policy (practical constraints)
- No leverage version: cap exposure at 100%.
- Moderate leverage version: cap at 110-130% only in low vol + strong uptrend.
- Defensive floor: set a minimum exposure (e.g., 10-20%) to avoid total market exit.
- Cash alternative: hold T-bills or cash if exposure is reduced.

### Optional options overlay (only in specific states)
- Trigger: high vol + downtrend + hedge cost not extreme.
- Instrument: put spread or collar (cap cost, keep some upside).
- Goal: reduce tail drawdown while preserving trend participation.

### Options overlay details
- Use put spreads to control cost and avoid constant bleed.
- Use collars if you accept a capped upside during the hedge window.
- Hedge cost gate: do not hedge when implied vol is very expensive vs realized.
- Roll schedule: monthly or quarterly; avoid overtrading.
- Hedge size: small and consistent (e.g., 10-30% notional) unless tail risk is extreme.

### Hedge cost signals (simple choices)
- Variance risk premium: implied vol minus realized vol.
- Skew: put skew relative to at-the-money implied vol.
- Term structure: front-month vs 3-month implied vol.

### Evaluation criteria (expanded)
- Total return vs S&P 500 buy-and-hold (CAGR, total return).
- Max drawdown and drawdown recovery time.
- Risk-adjusted metrics: Sharpe, Sortino, Calmar.
- Capture ratios: upside capture vs downside capture.
- Turnover and transaction cost sensitivity.
- Option carry cost and premium decay impact (if used).

### Diagnostics and sanity checks
- Check that performance is not isolated to one decade.
- Inspect exposure over time to confirm intuitive behavior.
- Stress test on crisis windows (2008, 2020) and calm periods.
- Compare against a simple trend-only strategy to see incremental value from regimes.

### Failure modes to watch
- Overfitting thresholds to one sample period.
- Over-hedging in benign volatility spikes.
- Whipsaw if trend signals are too fast.
- Hidden leverage risk if exposure increases during rising volatility.

### Proposed next decisions
1) Pick the directional signal:
   - 12-month momentum (robust, slow).
   - 6-month trend slope (balanced).
   - MA cross (intuitive but can be noisy).
2) Pick rebalance cadence:
   - Monthly (default).
   - Weekly (more responsive, higher turnover).
3) Pick exposure caps:
   - No leverage (max 100%).
   - Mild leverage (max 110-130% in the best state).
4) Decide if options are in-scope:
   - Overlay only in worst states.
   - No options, exposure-only.

### Implementation plan (conceptual, no code)
- Step 1: Choose the directional signal and define its score.
- Step 2: Lock the regime definitions from the existing model.
- Step 3: Define the matrix (or continuous sizing rule) and caps.
- Step 4: Run a simple backtest with reasonable transaction costs.
- Step 5: Compare to buy-and-hold and trend-only baseline.
- Step 6: Iterate on thresholds only if failure modes are clear.

### Parameter sweep (objective-driven)
Objective: maximize excess CAGR vs buy-and-hold, subject to max drawdown no worse than
the benchmark for the same window.

Suggested sweep grid (start small):
- trend_window: 21, 42, 63, 84, 126, 252
- trend_threshold: 0.15, 0.25, 0.35, 0.5
- rebalance: daily, weekly, monthly
- state_confirm: 1, 2
- matrix_set: default, aggressive

Runner (writes ranked tables to `reports/strategy_layered_sweep/`):
- Run `scripts/run_layered_strategy_sweep.py` directly in your IDE.
- Edit defaults in `scripts/run_layered_strategy_sweep.py` under `_default_config()`
  if you want to change the sweep grid or dates.

Inspect:
- `reports/strategy_layered_sweep/sweep_results.csv` for all configs
- `reports/strategy_layered_sweep/top_candidates.csv` for the top ranked configs
- `reports/strategy_layered_sweep/sweep_summary.txt` for the sweep setup

### Evaluation criteria
- Total return vs S&P 500 buy-and-hold.
- Max drawdown and drawdown duration.
- Upside capture and downside capture ratios.
- Hedge cost (if options used) and turnover.

### Next decision (to finalize Layer 2)
Pick the directional signal and its time horizon:
- Conservative: 12-month momentum (monthly rebalance).
- Faster: 3-6 month trend filter (monthly or biweekly).
- Balanced: 6-month trend slope (monthly).

Once chosen, the regime x direction matrix can be tuned for exposure sizing.
