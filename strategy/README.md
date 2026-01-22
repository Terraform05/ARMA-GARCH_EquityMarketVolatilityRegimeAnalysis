# Regime-Trend Strategy Notes

The regime-trend strategy combines:
- Volatility regime (ARMA-GARCH low/mid/high)
- Trend signal (directional return state)

Exposure is set either by a regime x trend matrix or a continuous sizing rule.

## Where the Code Lives

- Signals: `src/signals/` (trend, volatility)
- Strategy assembly: `src/strategies/regime_trend.py`
- Sizing logic: `src/strategies/sizing.py`

`build_regime_trend_frame(...)` accepts optional `trend_score`, `trend_state`,
and `vol_score`, so new signals can be injected without rewiring the pipeline.

## Run Workflow

### Optimize and Freeze
```bash
python scripts/run_regime_trend_optimize.py
```
Writes `configs/regime_trend_best.json`.

### Run Frozen Backtest
```bash
python scripts/run_regime_trend_backtest.py
```
Writes to `reports/strategy_regime_trend/`.

### Sweep (manual)
```bash
python scripts/run_regime_trend_sweep.py
```
Writes to `reports/strategy_regime_trend_sweep/`.

## Default Sweep Grid (edit in script)
Edit `_default_config()` in `scripts/run_regime_trend_sweep.py`:
- `trend_windows`
- `trend_thresholds`
- `rebalance`
- `state_confirms`
- `matrix_set`
- walk-forward split lengths and costs

## Design Principles

- Keep directional logic simple and interpretable.
- Use regimes for risk sizing, not direction prediction.
- Penalize turnover via cost assumptions, not hard caps.
- Prefer robust behavior across decades over a single best period.

## Evaluation Focus

- Total return vs benchmark
- Max drawdown and recovery
- Risk-adjusted metrics (Sharpe/Sortino)
- Turnover and cost sensitivity
- Stability across walk-forward windows
