# ARMA-GARCH_EquityMarketVolatilityRegimeAnalysis

Model and interpret volatility regimes in equity markets by separating return dynamics from volatility dynamics, and assess how risk evolves across market conditions.

## TL;DR

- Builds ARMA‑GARCH volatility regimes on SPX and compares them to VIX.
- Default model selection prioritizes realized‑vol tracking (GARCH), with BIC‑best available via config.
- Regime labels drive a risk‑control overlay that reduces drawdowns at the cost of lower return.
- Hedge‑cost monitoring flags when protection is cheap vs expensive.

## Project Goal

This project focuses on regime interpretation, not trading or alpha generation. The core questions and answers are:

- **When is the market calm vs stressed?** Conditional volatility regimes split at low <= 0.006721 and high >= 0.010107, with mid‑regime as transitional.
- **How persistent are volatility shocks?** EGARCH_t indicates high persistence (beta[1] ≈ 0.964), so volatility shocks decay slowly.
- **How does implied volatility (VIX) compare to realized volatility?** VIX aligns best with 10‑day realized volatility in this sample; divergence periods indicate risk pricing mismatches.
- **What does this mean for risk and valuation confidence?** High‑vol regimes coincide with higher VIX and deeper drawdowns, implying lower valuation confidence and higher hedge costs.

## Quick Start

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/run_all.py

## Dependencies

- Python 3.10+
- `pip install -r requirements.txt`

## General Overview

This project takes daily S&P 500 prices and the VIX index, transforms them into returns, then models how average returns behave and how volatility changes over time. The model produces a time series of conditional volatility, which is used to label low‑ and high‑volatility regimes and compare to implied volatility (VIX). Diagnostics and validation checks show whether the model is a good fit or whether volatility dynamics remain unexplained.

The workflow is intentionally linear: data prep feeds diagnostics; diagnostics guide model choice; modeling produces conditional volatility; validation checks model adequacy; regime analysis interprets volatility states; the OOS check stress‑tests the model’s forecasting behavior.

## Math (Explained)

We work with log returns because they are additive over time and behave better statistically than prices. Let $P_t$ be the S&P 500 adjusted close on day $t$. The log return is:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

The mean process is modeled with ARMA $(p, q)$, which allows returns to depend on their own past values (AR terms) and on past shocks (MA terms). This captures short‑run autocorrelation:

$$
r_t = \mu + \sum_{i=1}^{p} \phi_i r_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t
$$

The variance process is modeled with GARCH(1,1), which allows volatility to change over time and respond to recent shocks. The conditional variance $\sigma_t^2$ depends on yesterday’s squared shock and yesterday’s variance:

$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

Key parameters:
- $\omega$ is the long‑run variance level.
- $\alpha$ controls how strongly volatility reacts to new shocks.
- $\beta$ controls how persistent volatility is.
- $\alpha + \beta$ near 1 implies slow decay of volatility shocks.

To compare model output with VIX, we compute a rolling realized volatility proxy and annualize it. For window length $w$ (in trading days), the realized volatility is:

$$
\text{RV}_{t,w} = \sqrt{252} \times \text{StdDev}(r_{t-w+1:t}) \times 100
$$

The $\sqrt{252}$ factor annualizes daily volatility using the standard number of trading days per year, and the $\times 100$ converts it to percent to match VIX units.

## Steps And How They Fit Together

1) **Data preparation**  
   Download SPX and VIX, align by NYSE trading days, and compute returns. This produces the single aligned dataset used everywhere else. Without this aligned dataset, every downstream step fails or becomes inconsistent.

2) **Preliminary diagnostics**  
   Returns are stationary (ADF statistic -13.7338, p < 0.001) and volatility clustering is strong (ARCH statistic 1270.0264, p < 0.001). This supports ARMA for the mean and GARCH‑family models for the variance.  
   The plots below show return clustering and short‑lag dependence.

![Returns and squared returns](reports/diagnostics/plots/returns_series.png)
![ACF and PACF](reports/diagnostics/plots/acf_pacf.png)

3) **Model variants and selection**  
   GARCH, GJR, and EGARCH variants are compared with normal vs Student‑t errors. By default the pipeline selects the best model by realized‑vol tracking (currently **GARCH**), which aligns the model to observed volatility behavior.  
   You can switch selection to BIC by setting `VARIANT_SELECTION = "bic"` in `src/config.py`.
   Use the charts below to see how top variants track realized volatility and how information criteria differ.
   If your goal is interpretability or in‑sample fit, BIC is the right compass; if your goal is volatility tracking (risk monitoring), the realized‑vol metrics are the better guide.

![Model variant comparison](reports/modeling_variants/plots/variant_comparison.png)
![Top variants vs realized](reports/modeling_variants/plots/variant_vs_realized.png)
![Variant metrics](reports/modeling_variants/plots/variant_metrics.png)
![Best variant volatility](reports/modeling_variants/plots/best_variant_volatility.png)
![BIC vs tracking](reports/modeling_variants/plots/bic_vs_tracking.png)

Variant accuracy vs realized volatility (lower RMSE/QLIKE and higher correlation are better):

| Variant | Corr | RMSE | QLIKE |
| --- | --- | --- | --- |
| GARCH | 0.9587 | 3.0347 | 6.1994 |
| GARCH_t | 0.9297 | 3.5652 | 6.2014 |
| GJR | 0.8901 | 4.3646 | 6.2286 |
| GJR_t | 0.8901 | 4.3646 | 6.2286 |
| EGARCH_t | 0.8173 | 5.4161 | 6.2747 |
| EGARCH | 0.8081 | 5.5010 | 6.2790 |

QLIKE (quasi‑likelihood) is a volatility‑forecast loss defined as: QLIKE = mean(realized_var / forecast_var + log(forecast_var)). Lower is better and it is robust to noise in realized variance.

Note: EGARCH_t wins on BIC (in‑sample fit), while GARCH has the best realized‑vol tracking metrics. This highlights the tradeoff between model fit and out‑of‑sample alignment.

To force the pipeline to use the BIC‑best model, set `VARIANT_SELECTION = "bic"` in `src/config.py` before running the pipeline.

4) **Modeling (mean + variance)**  
   The selected ARMA order is (2, 0) with BIC -24948.0268. The EGARCH_t variance model yields parameters: omega -0.335733, alpha[1] 0.178828, beta[1] 0.963976. This model’s conditional volatility is the core input for regimes and implied vs realized comparisons.

5) **Model validation**  
   Residual autocorrelation remains at lag 20 (Ljung‑Box p = 0.000110), but squared residuals and ARCH tests are no longer significant (p ≈ 0.79 and p ≈ 0.81). This indicates variance dynamics are well captured, with remaining structure mainly in the mean.  
   Residual plots confirm the remaining structure visually.
   Practical takeaway: the volatility model is adequate for regime labeling, but return predictability is still limited.

![Residual series](reports/validation/plots/residuals_series.png)
![Residual ACF](reports/validation/plots/residuals_acf.png)
![Residual Q-Q](reports/validation/plots/residuals_qq.png)

6) **Regime interpretation**  
   Conditional volatility is split into low/mid/high regimes using 33% and 66% quantiles (low <= 0.006721, high >= 0.010107). The realized volatility window that aligns best with VIX is 10 days.  
   The figures below show regime structure, implied vs realized comparison, and outcome summaries by regime.

![Regime scatter](reports/regime_analysis/plots/regimes.png)
![VIX vs realized](reports/regime_analysis/plots/vix_vs_realized.png)
![Window metrics](reports/regime_analysis/plots/realized_window_metrics.png)
![Regime outcomes](reports/regime_analysis/plots/regime_outcomes.png)

7) **Out‑of‑sample check**  
   The holdout window is 2024‑01‑01 to 2026‑01‑01 (502 rows). Static multi‑step forecasts are not available for EGARCH, so the rolling 1‑step forecast is the primary diagnostic. Rolling OOS correlation vs realized is 0.7286 with RMSE 5.5016.  
   The rolling forecast plot below shows how well volatility is tracked in the holdout.
   This is a monitoring check, not a trading signal: directional tracking is the key success criterion.

![OOS rolling forecast vs realized](reports/oos_check/plots/forecast_vs_realized_rolling.png)

8) **Hedge‑cost monitoring**  
   The hedge ratio (VIX / realized vol) flags expensive vs cheap hedging. Current thresholds are 1.056 (cheap) and 1.925 (expensive), with 20.00% of days cheap and 20.00% expensive. Average signal persistence is 4.6 days (cheap), 7.8 days (neutral), and 5.9 days (expensive).  
   Practical use: use the ratio as a budgeting signal (hedge more when cheap, hedge less or seek alternatives when expensive). See `reports/hedge_monitoring/README.md` for full tables and plots.

9) **Regime strategy backtest**  
   The baseline exposure rule (low=1.0, mid=0.75, high=0.25) yields annual return 0.0618, annual vol 0.0872, Sharpe 0.7085, and max drawdown -0.1234. Buy‑and‑hold returns 0.1128 with vol 0.1738 and max drawdown -0.3392.  
   The Sharpe‑best exposure map is low=1.0, mid=1.0, high=0.5 with Sharpe 0.7397.  
   Interpretation: the regime strategy sacrifices return for lower drawdowns and higher risk‑adjusted performance. If your objective is pure return, buy‑and‑hold wins; if your objective is risk control, the strategy helps. See `reports/strategy_backtest/README.md` for the equity curve and variant table.

Key short‑horizon visuals (last 12 months, rebased to the same start):
- Black line: buy‑and‑hold equity curve.
- Slate‑blue line: regime strategy equity curve.
- Regime strip colors: green = low, amber = mid, red = high volatility.
- Exposure values (1.0, 0.75, 0.25) are fractions of full equity exposure.

![Regime strategy equity curve (last year)](reports/strategy_backtest/plots/equity_curve_last_year.png)

The exposure overlay shows how the regime labels drive risk scaling: exposure rises in low‑volatility regimes and falls in high‑volatility regimes.

![Exposure overlay (last year)](reports/strategy_backtest/plots/exposure_overlay_last_year.png)

Full‑sample views for context:

![Regime strategy equity curve](reports/strategy_backtest/plots/equity_curve.png)

![Exposure overlay](reports/strategy_backtest/plots/exposure_overlay.png)

## Actionable Next Steps

- **Risk control:** use regime labels to scale exposure or to set hedge budgets before stress periods, then review hedge ratio signals for timing.
- **Model choice:** choose BIC‑best (EGARCH_t) for in‑sample fit or switch to tracking‑best (GARCH) when the goal is aligning with realized volatility.
- **Strategy tuning:** evaluate the Sharpe‑optimized exposure map from `reports/strategy_backtest/data/strategy_variants.csv`, then rerun to confirm stability.
- **Hold vs strategy:** if you only care about total return, buy‑and‑hold wins in this sample; if drawdown control matters, the regime strategy is the better fit.

## Scope

- Underlying: S&P 500 Index returns (SPX)
- Volatility benchmark: VIX Index (implied volatility)
- Horizon: 10–20+ years with low-vol, crisis, and transitional regimes

## Documentation

- [Data preparation](docs/01_data_prep.md)
- [Preliminary diagnostics](docs/02_diagnostics.md)
- [ARMA + GARCH modeling](docs/03_modeling.md)
- [Model validation](docs/04_validation.md)
- [Regime interpretation](docs/05_regime_analysis.md)
- [Out-of-sample check](docs/06_oos_check.md)
- [Executive summary](reports/summary.md)
- [Insights report](reports/insights.md)
- [Modeling variants](reports/modeling_variants/README.md)
- [Hedge cost monitoring](reports/hedge_monitoring/README.md)
- [Regime strategy backtest](reports/strategy_backtest/README.md)
- [Hedge + strategy overview](reports/hedge_strategy/README.md)

## Run All

- `python scripts/run_all.py`

## Key Outputs

- `data/processed/spx_vix_aligned.csv`
- `reports/diagnostics/README.md`
- `reports/modeling/README.md`
- `reports/modeling_variants/README.md`
- `reports/validation/README.md`
- `reports/regime_analysis/README.md`
- `reports/oos_check/README.md`
- `reports/insights.md`
- `reports/hedge_monitoring/README.md`
- `reports/strategy_backtest/README.md`
- `reports/hedge_strategy/README.md`
