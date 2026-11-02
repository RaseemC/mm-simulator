"""Adverse selection: hidden informed episodes, shared by live and batch.

Concept: informed traders sometimes know where the mid is about to go. We model
this as a latent episode process per session. During an episode with direction
s = +1 (up) or -1 (down):
  * the mid drifts:   m_{n+1} = m_n + strength*sigma*sqrt(dt)*s + sigma*sqrt(dt)*Z
    (`informed_drift`; added by the MidpriceProcess), and
  * the side about to be wrong is hit fill_boost times as often (applied inside
    core.step through the `informed` array): s = +1 -> our ask is lifted,
    s = -1 -> our bid is hit.
The single toxicity parameter is `strength`, in units of the per-tick diffusion
scale sigma*sqrt(dt), so it means the same thing at every sigma. strength = 0
disables episodes entirely.

Lifecycle (a small state machine run over the session's pre-drawn uniforms):
    quiet --(V_start < p_episode)--> LEAD ticks (book tilt only, no drift)
          --> ACTIVE for 1 + floor(-mean_duration * ln(1 - V_dur)) ticks --> quiet
Outputs per tick:
    informed[n] in {-1, 0, +1}: direction while ACTIVE (drift + fill boost)
    tilt[n]     in {-1, 0, +1}: direction while LEAD or ACTIVE (the synthetic book
                                 tilts; also the ground-truth label for detector
                                 scoring, since the tilt starts at the lead-in)

Everything is a pure function of (draws, config), so a live session replays the
exact episodes a batch run sees.

Assumptions / weaknesses: one episode at a time; constant strength within an
episode; drift is a known constant (a trader who knew it would not be adversely
selected); fill_boost is a modelling constant independent of strength, so at
strength -> 0+ informed flow exists but carries no edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AdverseConfig, MarketConfig
from .draws import SessionDraws, draw_session
from .midprice import MidpriceProcess


@dataclass(frozen=True)
class Regime:
    informed: np.ndarray  # (n, N) int8, direction while ACTIVE else 0
    tilt: np.ndarray  # (n, N) int8, direction while LEAD or ACTIVE else 0


def build_regime(u_start: np.ndarray, u_dir: np.ndarray, u_dur: np.ndarray,
                 cfg: AdverseConfig) -> Regime:
    """Run the episode state machine; inputs shape (n, N), vectorised over n."""
    u_start, u_dir, u_dur = (np.atleast_2d(np.asarray(a, dtype=float))
                             for a in (u_start, u_dir, u_dur))
    n, N = u_start.shape
    informed = np.zeros((n, N), dtype=np.int8)
    tilt = np.zeros((n, N), dtype=np.int8)
    if cfg.strength == 0:
        return Regime(informed, tilt)
    direction = np.zeros(n, dtype=np.int64)
    pre = np.zeros(n, dtype=np.int64)       # lead ticks remaining
    left = np.zeros(n, dtype=np.int64)      # active ticks remaining
    for t in range(N):
        in_pre = pre > 0
        in_act = ~in_pre & (left > 0)
        quiet = ~in_pre & ~in_act
        # lead-in: count down; the tick the lead reaches 0 is already active
        pre = np.where(in_pre, pre - 1, pre)
        finishing = in_pre & (pre == 0)
        dur = 1 + np.floor(-cfg.mean_duration * np.log1p(-u_dur[:, t])).astype(np.int64)
        left = np.where(finishing, dur, left)
        # active: count down; direction clears when the episode ends
        left = np.where(in_act, left - 1, left)
        ended = in_act & (left == 0)
        direction = np.where(ended, 0, direction)
        # quiet: maybe start a new episode
        start = quiet & (u_start[:, t] < cfg.p_episode)
        direction = np.where(start, np.where(u_dir[:, t] < 0.5, 1, -1), direction)
        pre = np.where(start, cfg.lead, pre)
        informed[:, t] = np.where(left > 0, direction, 0)
        tilt[:, t] = np.where((pre > 0) | (left > 0), direction, 0)
    return Regime(informed, tilt)


def informed_drift(informed: np.ndarray, cfg: AdverseConfig, market: MarketConfig) -> np.ndarray:
    """Per-tick drift of the mid: strength * sigma * sqrt(dt) * s_n."""
    return cfg.strength * market.sigma * np.sqrt(market.dt) * np.asarray(informed, dtype=float)


@dataclass(frozen=True)
class Prepared:
    """Everything a run needs for a block of sessions (batch or one live session)."""

    mids: np.ndarray  # (n, N+1)
    u_bid: np.ndarray  # (n, N)
    u_ask: np.ndarray  # (n, N)
    informed: np.ndarray  # (n, N)
    tilt: np.ndarray  # (n, N)
    draws: SessionDraws

    @property
    def u_queue(self) -> np.ndarray:  # (n, N, 8)
        return self.draws.u_queue

    @property
    def u_depth0(self) -> np.ndarray:  # (n, 2, MAX_LEVELS)
        return self.draws.u_depth0


def prepare(process: MidpriceProcess, draws: SessionDraws, cfg: AdverseConfig) -> Prepared:
    """Draws (n, N) -> mids with informed drift, plus the episode arrays."""
    d = draws.map(np.atleast_2d)
    reg = build_regime(d.u_start, d.u_dir, d.u_dur, cfg)
    drift = informed_drift(reg.informed, cfg, process.market)
    mids = process.path_from_shocks(d.shocks, drift)
    return Prepared(mids, d.u_bid, d.u_ask, reg.informed, reg.tilt, d)


def prepare_batch(process: MidpriceProcess, seed: int, n_sessions: int,
                  cfg: AdverseConfig, first: int = 0,
                  want_queue: bool = True) -> Prepared:
    """Sessions first .. first+n-1, each drawn from its own (seed, i) stream.

    `want_queue=False` (pass this whenever the caller will run `queue=None`,
    i.e. the exponential fill model only) skips drawing the queue-model blocks
    entirely - see draws.py. Default True keeps old callers' behaviour and
    seeding unchanged.
    """
    stacked = SessionDraws.stack(draw_session(process, seed, i, want_queue)
                                 for i in range(first, first + n_sessions))
    return prepare(process, stacked, cfg)
