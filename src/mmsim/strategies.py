"""Quoting rules.

Concept: a market maker posts a bid and an ask around a *reservation price*
that leans against its inventory (long -> quote lower to sell, short -> higher).

Equations:
    reservation price  r_n = m_n - eta * q_n
    bid                b_n = r_n - delta
    ask                a_n = r_n + delta
  so the distances from the mid are
    d_bid = m_n - b_n = delta + eta * q_n
    d_ask = a_n - m_n = delta - eta * q_n

eta = 0 is the naive symmetric maker; eta > 0 is inventory-aware. One class,
parameterised by eta (no duplicated code paths).

Interface for future rules (Avellaneda-Stoikov with eta_t = gamma*sigma^2*(T-t),
hard inventory caps, ...): subclass `Quoter` and implement `distances(m, q, t)`.
No lookahead: a quoter sees only (m_n, q_n, t_n), never future information.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import StrategyConfig


class Quoter(ABC):
    """Maps (mid, inventory, time) -> quote distances from the mid."""

    @abstractmethod
    def distances(self, m: np.ndarray, q: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (d_bid, d_ask) = (m - bid, ask - m), arrays shaped like m."""


class InventorySkewQuoter(Quoter):
    """Constant-eta reservation-price quoter. eta = 0 gives the naive maker."""

    def __init__(self, delta: float, eta: float) -> None:
        self.delta = float(delta)
        self.eta = float(eta)

    @classmethod
    def from_config(cls, cfg: StrategyConfig) -> "InventorySkewQuoter":
        return cls(cfg.delta, cfg.eta)

    def distances(self, m, q, t):
        skew = self.eta * q
        return self.delta + skew, self.delta - skew

    def __repr__(self) -> str:
        return f"InventorySkewQuoter(delta={self.delta}, eta={self.eta})"


def as_eta(gamma: float, sigma: float, T: float, t) -> np.ndarray:
    """eta_t = gamma * sigma^2 * (T - t): the AS time-varying inventory-skew
    coefficient. Pure function so scripts/tests can evaluate the formula without
    building a Quoter."""
    return gamma * sigma**2 * (T - np.asarray(t, dtype=float))


def as_half_spread(gamma: float, sigma: float, k: float, T: float, t) -> np.ndarray:
    """delta_t = [eta_t + (2/gamma) ln(1 + gamma/k)] / 2: the AS optimal
    half-spread, split symmetrically around the reservation price (Avellaneda &
    Stoikov 2008, eq. for the optimal bid-ask spread). As gamma -> 0+ (risk
    neutral) this converges to the MYOPIC optimum 1/k: maximising single-tick
    expected revenue R(d) = A*dt*d*exp(-k*d) over d gives R'(d) = 0
    => 1 - k*d = 0 => d* = 1/k, independent of gamma, sigma or A. That limit is
    checked directly (not assumed) in scripts/validate_closed_form.py."""
    eta_t = as_eta(gamma, sigma, T, t)
    return (eta_t + (2.0 / gamma) * np.log(1.0 + gamma / k)) / 2.0


class AvellanedaStoikovQuoter(Quoter):
    """Exact closed-form AS(2008) optimal policy: TIME-VARYING eta_t and delta_t.

    reservation price  r(s,q,t) = s - q * eta_t,         eta_t   = as_eta(...)
    half-spread        delta_t  = as_half_spread(...)
    quotes             bid = r - delta_t,  ask = r + delta_t
      so               d_bid = delta_t + eta_t*q,  d_ask = delta_t - eta_t*q

    Both InventorySkewQuoter's eta and delta are CONSTANTS approximating this; the
    true optimum decays toward (0, 1/k) as t -> T. This class is the honest
    closed-form policy against which the constant-parameter strategy is a
    simplification, used by validate_closed_form.py (C4) to check that
    policy against the simulator rather than merely assuming it.
    """

    def __init__(self, gamma: float, sigma: float, k: float, T: float) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma, self.sigma, self.k, self.T = float(gamma), float(sigma), float(k), float(T)

    def distances(self, m, q, t):
        eta_t = as_eta(self.gamma, self.sigma, self.T, t)
        delta_t = as_half_spread(self.gamma, self.sigma, self.k, self.T, t)
        skew = eta_t * np.asarray(q, dtype=float)
        return delta_t + skew, delta_t - skew

    def __repr__(self) -> str:
        return f"AvellanedaStoikovQuoter(gamma={self.gamma}, sigma={self.sigma}, k={self.k}, T={self.T})"
