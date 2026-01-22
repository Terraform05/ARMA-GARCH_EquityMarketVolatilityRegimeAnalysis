# ARMA-GARCH Equity Market Volatility Regime Analysis

Model equity-market volatility regimes using ARMA-GARCH and evaluate regime-aware strategies.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Everything (full pipeline)
```bash
python scripts/run_all.py
```

### Optimize + Frozen Backtest (main workflow)
```bash
python scripts/run_regime_trend_optimize.py
python scripts/run_regime_trend_backtest.py
```

### Regime Series Only (minimal dependency build)
```bash
python scripts/run_regime_series.py
```

## What Gets Generated

- `reports/` is the current, readable output set (overwritten on each run).
- `runs/` is timestamped history for reproducible audits.
- `configs/active/` snapshots the config used by `run_all`.

Start here for outputs: `reports/README.md`.

## Model Variant Selection

Variant selection is controlled by `src/config.py`:

- `VARIANT_SELECTION = "bic"` (best in-sample fit)
- `VARIANT_SELECTION = "tracking"` (best realized-vol tracking)
- `VARIANT_SELECTION = "hybrid"` (tracking-first, prefer lower BIC if close)

The model-variants step writes `reports/modeling_variants/data/best_variant.txt`,
which now includes BIC, tracking, hybrid, and the active selection.

## Repository Layout

- `configs/` config files, including frozen best config
- `configs/active/` config snapshots used by full runs
- `data/` raw and processed data
- `docs/` pipeline notes and plain-English summary
- `reports/` generated plots and tables (current run)
- `runs/` timestamped outputs for audit/repro
- `scripts/` CLI entry points
- `src/` reusable modeling, analysis, and strategy code
- `strategy/` strategy design notes
- `tests/` tests

## Documentation

- Pipeline docs: `docs/pipeline/`
- Plain-English summary: `docs/readmes/laymans_readme.md`
- Long-form: `docs/readmes/concepts.md`, `docs/readmes/math.md`, `docs/readmes/plots.md`
- Strategy notes: `strategy/README.md`
- Reports index: `reports/README.md`

## Notes

- `run_all.py` overwrites `reports/` outputs.
- If you want to preserve prior outputs, copy `reports/` before running.
