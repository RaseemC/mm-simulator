"""CARA utility / certainty equivalent, for comparing strategies the way the
Avellaneda-Stoikov (2008) derivation does: maximising E[-exp(-gamma * wealth)],
not raw expected P&L or the Sharpe ratio this project has used elsewhere.

Concept: the certainty equivalent CE solves U(CE) = E[U(X)] for U(x) = -exp(-gamma x):
    -exp(-gamma*CE) = E[-exp(-gamma*X)]
    CE = -(1/gamma) * ln( E[exp(-gamma*X)] )
     = -(1/gamma) * ( logsumexp(-gamma*X) - ln(n) )
computed via a shifted log-sum-exp so it does not overflow for realistic P&L
magnitudes. CE is in the same units as X (dollars): a risk-averse agent (gamma > 0)
is indifferent between the random X and a SURE payoff of CE, so CE < E[X] whenever
X has any variance, with the gap growing in gamma and in Var(X). This is this
project's only place that maximises a DIFFERENT objective than Sharpe or mean
P&L; everywhere else continues to use whichever metric that section documents.

Weakness: for gamma*X large and very spread out, the logsumexp shift keeps this
numerically stable, but CE itself becomes extremely sensitive to the tail of X
when gamma is not small (a CARA agent cares a great deal about rare bad outcomes)
- the bootstrap CI on CE should be read with that in mind.
"""
from __future__ import annotations

import numpy as np


def certainty_equivalent(pnl: np.ndarray, gamma: float) -> float:
    """CE = -(1/gamma) ln( mean(exp(-gamma*pnl)) ), computed stably."""
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    x = -gamma * np.asarray(pnl, dtype=float)
    m = np.max(x)
    log_mean_exp = m + np.log(np.mean(np.exp(x - m)))
    return float(-log_mean_exp / gamma)


def cara_utility(pnl: np.ndarray, gamma: float) -> np.ndarray:
    """U(x) = -exp(-gamma x), elementwise."""
    return -np.exp(-gamma * np.asarray(pnl, dtype=float))
