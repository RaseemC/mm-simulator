"""Midprice processes (batch array AND tick-by-tick generator).

Concept: the "fair price" the market maker quotes around. Hidden behind an
abstract interface so jump / stochastic-vol / replayed-data processes can be
plugged in later without touching the engine.

Equation (arithmetic random walk, optional known drift):
    m_{n+1} = m_n + drift_n + sigma * sqrt(dt) * Z_{n+1},   Z ~ N(0,1) i.i.d.
`drift` (default none = zero drift) is a deterministic per-tick array; adverse
selection supplies it from the hidden informed episodes (see adverse.py).

Design: a process turns a block of random shocks into a path. Drawing the
shocks (`draw_shocks`) is separate from building the path, so batch and live
paths are built from literally the same numbers. Both builders add increments
left to right starting from m_0, so they agree to the last bit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from .config import MarketConfig


class MidpriceProcess(ABC):
    """Interface: shocks -> midprice path, as an array or as a generator."""

    def __init__(self, market: MarketConfig) -> None:
        self.market = market

    @abstractmethod
    def draw_shocks(self, rng: np.random.Generator) -> np.ndarray:
        """Consume `rng` and return the session's shocks, shape (N,)."""

    @abstractmethod
    def path_from_shocks(self, shocks: np.ndarray, drift: np.ndarray | None = None) -> np.ndarray:
        """Batch: shocks (..., N) -> mids (..., N+1) with mids[..., 0] = m_0."""

    @abstractmethod
    def stream(self, shocks: np.ndarray, drift: np.ndarray | None = None) -> Iterator[float]:
        """Live: yield m_0, m_1, ..., m_N one tick at a time."""


class ArithmeticRandomWalk(MidpriceProcess):
    """Zero-drift Gaussian random walk with constant sigma."""

    def draw_shocks(self, rng):
        return rng.standard_normal(self.market.N)

    def _increments(self, shocks, drift=None):
        inc = self.market.sigma * np.sqrt(self.market.dt) * shocks
        return inc if drift is None else np.asarray(drift, dtype=float) + inc

    def path_from_shocks(self, shocks, drift=None):
        inc = self._increments(np.asarray(shocks, dtype=float), drift)
        start = np.full(inc.shape[:-1] + (1,), self.market.m0)
        # cumsum adds sequentially from m_0, matching the generator exactly
        return np.cumsum(np.concatenate([start, inc], axis=-1), axis=-1)

    def stream(self, shocks, drift=None):
        inc = self._increments(np.asarray(shocks, dtype=float), drift)
        m = self.market.m0
        yield m
        for x in inc:
            m = m + float(x)
            yield m
