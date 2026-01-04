# Modeling Summary

## Key Results

- ARMA order selected by BIC: (2, 0) with BIC = -24948.026784
- GARCH parameters:
  - omega = 0.000002
  - alpha[1] = 0.100000
  - beta[1] = 0.880000
  - alpha + beta = 0.980000

## Interpretation

- The mean equation prefers an AR(2) structure with no MA terms, implying short‑lag dependence in returns.
- GARCH captures volatility clustering with high persistence (alpha + beta close to 1).
- The conditional volatility series is the core input for regime labeling and implied vs realized comparisons.
