"""Fill-probability model.

Concept: a passive quote placed `d` away from the mid is hit with a probability
that decays exponentially in d (far quotes rarely trade).

Equation (per step, each side independently):
    p(d) = min(1, A * exp(-k * d) * dt),   fill iff U < p,   U ~ Uniform[0,1)

Informed flow multiplies p by `mult` (>= 1) on the side that is about to be
picked off; the min(1, .) cap then applies to the product. `cap_binds` exposes
where the cap is active so the simulator can report how often it mattered.
Distances are floored at FillConfig.min_dist by core.step before they get here.
"""
from __future__ import annotations

import numpy as np

from .config import FillConfig


def raw_fill_prob(d: np.ndarray, fills: FillConfig, dt: float, mult=1.0) -> np.ndarray:
    """Uncapped intensity mult*A*exp(-k d)*dt."""
    return fills.A * np.exp(-fills.k * np.asarray(d, dtype=float)) * dt * mult


def fill_prob(d: np.ndarray, fills: FillConfig, dt: float, mult=1.0) -> np.ndarray:
    """p = min(1, mult*A*exp(-k d)*dt)."""
    return np.minimum(1.0, raw_fill_prob(d, fills, dt, mult))


def cap_binds(d: np.ndarray, fills: FillConfig, dt: float, mult=1.0) -> np.ndarray:
    """Boolean array: True where the min(1, .) cap is active (raw p > 1)."""
    return raw_fill_prob(d, fills, dt, mult) > 1.0


def draw_fills(d: np.ndarray, u: np.ndarray, fills: FillConfig, dt: float,
               mult=1.0) -> np.ndarray:
    """Return integer {0,1} fills: 1 where U < p."""
    return (u < fill_prob(d, fills, dt, mult)).astype(np.int64)
