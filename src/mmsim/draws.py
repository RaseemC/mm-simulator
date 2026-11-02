"""Per-session random draws (the seeding contract).

Concept: session i's randomness depends ONLY on (seed, i) - never on how many
sessions run beside it, or on batch vs live mode. That is what makes the
live/batch equivalence test possible and sweeps noise-free across settings.

Seeding: SeedSequence(seed, spawn_key=(i,)) is exactly the i-th child of
SeedSequence(seed).spawn(n), but can be built lazily for any i (live mode).

Draw order within a session (documented, fixed):
    logically per step n:  Z_n, U_bid_n, U_ask_n, (V_start_n, V_dir_n, V_dur_n)
    physically per session: one block of N shocks Z, then one (N, 2) block of
    uniforms whose columns are (U_bid, U_ask), then one (N, 3) block of uniforms
    whose columns drive the hidden informed episodes (start, direction,
    duration), then (if `want_queue`) the limit-order-book blocks: an
    (N, N_QUEUE_COLS) block of per-tick uniforms (market-order arrival / size /
    cancel / add, both sides) and a (2, MAX_LEVELS) block of initial per-level
    depth multipliers.
Every block THAT IS DRAWN is appended at the END in this fixed order, so earlier
blocks are bit-identical to the streams that predate it, and are bit-identical
whether or not the book blocks are drawn (nothing downstream of a session's own
generator is affected by how much of it a given caller chooses to consume, since
each session owns an independent generator - see session_rng). The book blocks
have FIXED shapes (MAX_LEVELS, not the configured level count), so changing a
book parameter never shifts the stream either.

`want_queue` (default True, for backward compatibility with any caller that does
not pass it): when False, the book blocks are NOT drawn at all - `u_queue` and
`u_depth0` come back as None rather than allocated-and-unused. This matters at
scale: at n sessions x N=200 ticks, `u_queue` alone is n*200*8*8 bytes (12.8 MB
per 1,000 sessions), allocated whether or not the queue model is ever invoked,
which is wasteful and was observed to make very large exponential-only runs slow
and memory-heavy for no benefit (see docs/DESIGN_NOTES.md). Callers that will
pass `queue=None` to `core.step`/`simulate_paths` should pass `want_queue=False`
here; `tests/test_draws.py` checks the exponential-only path performs no queue
allocation at all.
Both strategies consume the SAME draws (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .midprice import MidpriceProcess


def session_rng(seed: int, i: int) -> np.random.Generator:
    """Independent generator for session i."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(i,))))


# Fixed widths so book parameters never shift the random stream.
MAX_LEVELS = 24
N_QUEUE_COLS = 8  # mo_buy, mo_sell arrival; mo_buy, mo_sell size; cancel bid/ask; add bid/ask


@dataclass(frozen=True)
class SessionDraws:
    shocks: np.ndarray  # (N,)  Z
    u_bid: np.ndarray  # (N,)
    u_ask: np.ndarray  # (N,)
    u_start: np.ndarray  # (N,)  episode-start uniforms (informed flow)
    u_dir: np.ndarray  # (N,)  direction uniforms
    u_dur: np.ndarray  # (N,)  duration uniforms
    u_queue: np.ndarray  # (N, N_QUEUE_COLS)  order-book flow uniforms
    u_depth0: np.ndarray  # (2, MAX_LEVELS)  initial depth multipliers

    FIELDS = ("shocks", "u_bid", "u_ask", "u_start", "u_dir", "u_dur",
              "u_queue", "u_depth0")

    def map(self, fn) -> "SessionDraws":
        """Apply `fn` to every array (used to stack sessions or add a leading axis).
        None fields (queue draws that were never requested) pass through as None."""
        return SessionDraws(*(None if (v := getattr(self, f)) is None else fn(v)
                              for f in self.FIELDS))

    @classmethod
    def stack(cls, sessions) -> "SessionDraws":
        """Stack per-session draws into leading-axis arrays.

        Assumes every session in the batch was drawn with the SAME `want_queue`
        (true for every caller in this project, which fixes it once per batch) -
        so a field is either None for every session or an array for every
        session, and only the first session needs checking.
        """
        seq = list(sessions)
        first = seq[0]
        return cls(*(None if getattr(first, f) is None else
                    np.stack([getattr(d, f) for d in seq]) for f in cls.FIELDS))


def draw_session(process: MidpriceProcess, seed: int, i: int,
                 want_queue: bool = True) -> SessionDraws:
    rng = session_rng(seed, i)
    shocks = process.draw_shocks(rng)
    u = rng.random((process.market.N, 2))
    v = rng.random((process.market.N, 3))
    w = rng.random((process.market.N, N_QUEUE_COLS)) if want_queue else None
    d0 = rng.random((2, MAX_LEVELS)) if want_queue else None
    return SessionDraws(shocks, u[:, 0].copy(), u[:, 1].copy(),
                        v[:, 0].copy(), v[:, 1].copy(), v[:, 2].copy(), w, d0)


def draw_batch(process: MidpriceProcess, seed: int, n_sessions: int,
               want_queue: bool = True) -> SessionDraws:
    """Stack draws of sessions 0..n-1 -> arrays with a leading session axis."""
    return SessionDraws.stack(draw_session(process, seed, i, want_queue)
                              for i in range(n_sessions))
