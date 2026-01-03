# Modeling Summary

## Key Results

- ARMA order selected by BIC: (2, 0) with BIC = -24948.026784
- EGARCH_t parameters:
  - omega = -0.335733
  - alpha[1] = 0.178828
  - beta[1] = 0.963976
  - alpha + beta = 1.142804

## Interpretation

- The mean equation prefers an AR(2) structure with no MA terms, implying short‑lag dependence in returns.
- EGARCH with Student‑t errors allows asymmetric and heavy‑tailed volatility dynamics.
- The conditional volatility series is the core input for regime labeling and implied vs realized comparisons.
