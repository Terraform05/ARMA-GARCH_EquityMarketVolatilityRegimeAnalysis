# Math

This document collects the math definitions used in the pipeline.

## Log Returns

We work with log returns because they are additive over time and behave better statistically than prices. Let P_t be the S&P 500 adjusted close on day t. The log return is:

r_t = log(P_t / P_{t-1})

Key variables:
- r_t is the log return on day t.
- P_t and P_{t-1} are adjusted close prices on days t and t-1.
- t indexes trading days.

## ARMA Mean Process

The mean process is modeled with ARMA(p, q), which allows returns to depend on their own past values (AR terms) and on past shocks (MA terms). This captures short-run autocorrelation:

r_t = mu + sum_{i=1}^p phi_i r_{t-i} + sum_{j=1}^q theta_j epsilon_{t-j} + epsilon_t

Key variables:
- mu is the mean return.
- phi_i are AR coefficients on past returns r_{t-i}.
- theta_j are MA coefficients on past shocks epsilon_{t-j}.
- epsilon_t is the return shock (innovation) at time t.
- p and q are the AR and MA orders.

## ARCH/GARCH Variance Process

ARCH models volatility as a function of past squared shocks; big moves tend to be followed by bigger variance. GARCH extends this by including the previous day's variance, capturing persistence.

The variance process is modeled with GARCH(1,1):

sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

Key variables:
- sigma_t^2 is the conditional variance at time t.
- epsilon_{t-1} is yesterday’s return shock.
- omega is the long-run variance level.
- alpha controls how strongly volatility reacts to new shocks.
- beta controls how persistent volatility is.
- alpha + beta near 1 implies slow decay of volatility shocks.

## Realized Volatility Proxy

To compare model output with VIX, we compute rolling realized volatility and annualize it. For window length w (in trading days):

RV_{t,w} = sqrt(252) * StdDev(r_{t-w+1:t}) * 100

Key variables:
- RV_{t,w} is realized volatility at time t.
- w is the lookback window in trading days.
- StdDev(r_{t-w+1:t}) is the standard deviation of returns over the window.
- sqrt(252) annualizes daily volatility; * 100 converts to percent.

## Strategy Returns

Let E_t be exposure at time t and r_t be the log return. Strategy log return is:

r_t^{strat} = E_t * r_t

If transaction costs are included, they are applied as a per-turnover penalty:

r_t^{net} = r_t^{strat} - cost_bps * turnover_t / 10000

## Risk Metrics

Sharpe (annualized):
- Sharpe = (mean(r_t) * 252) / (std(r_t) * sqrt(252))

Sortino (annualized):
- Sortino = (mean(r_t) * 252) / (std(r_t | r_t < 0) * sqrt(252))

Calmar:
- Calmar = CAGR / |max_drawdown|

Max drawdown:
- The minimum of the cumulative return curve relative to its prior peaks.

## Forecast Loss (QLIKE)

QLIKE is a volatility-forecast loss defined as:

QLIKE = mean(realized_var / forecast_var + log(forecast_var))

Lower QLIKE indicates a better fit to realized variance and is robust to noise in realized variance.
