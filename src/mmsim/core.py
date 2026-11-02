"""The PURE one-tick step function shared by live and batch modes.

Concept: all accounting lives here and nowhere else. `step` maps
(state at n, next mid, uniform draws) -> state at n+1 without side effects.
It works on arrays of shape (n,) (n = 1 in live mode, n = n_sessions in batch).

Order within a step (matters):
  1. quotes from (m_n, q_n):   b = m - d_bid,  a = m + d_ask, with each distance
     floored at fills.min_dist (the floor lives HERE so live and batch share it)
  2. fills, from ONE OF TWO models:
     * exponential (default, `queue=None`): F = 1[U < p(d)], p = min(1, A e^{-k d} dt).
       If an informed episode is active (`informed` = -1/+1 for this session at
       this tick), p is multiplied by fill_boost on the picked-off side:
       informed = -1 -> bid, informed = +1 -> ask.
     * queue position (`queue=QueueConfig(...)`, see book.py): our order rests in
       a price-level queue; it fills only when market-order flow consumes
       everything ahead of it, and then only up to its remaining size. Fills are
       therefore SIZES, not indicators, and may be PARTIAL. Adverse selection
       enters as a higher market-order rate on the picked-off side.
     The mid's informed drift is NOT applied here: the caller supplies m_next.
  3-5. identical accounting for both models, driven by the executed prices and
     sizes. In the queue model the executed price is the grid price our order
     actually rests at, which is at least `d` from the mid but not exactly `d`,
     so spread capture uses the ACTUAL distance and is weighted by filled size.
  3. q_{n+1} = q_n + F_bid - F_ask
     X_{n+1} = X_n - F_bid*b + F_ask*a
  4. mid moves to m_{n+1} AFTER fills (caller supplies m_next)
  5. MTM  Pi_{n+1} = X_{n+1} + q_{n+1} * m_{n+1}

P&L decomposition (exact identity), tracked incrementally:
    spread_pnl    += F_bid*(m_n - bid) + F_ask*(ask - m_n)      (size-weighted)
    inventory_pnl += q_{n+1} * (m_{n+1} - m_n)
The identity needs no assumption about sizes being 1 or prices being m -+ delta:
    Pi_{n+1} - Pi_n = F_b*(m_n - b) + F_a*(a - m_n) + q_{n+1}*(m_{n+1} - m_n)
holds for any execution prices and any (fractional) sizes.
  Summing: Pi_N = spread_pnl + inventory_pnl, since
    Pi_{n+1} - Pi_n = F_bid*d_bid + F_ask*d_ask + q_{n+1}(m_{n+1} - m_n).

Terminal P&L (liquidate at mid at T): Pi_N - c*|q_N|.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from . import book as bk
from .config import FillConfig, QueueConfig
from .fills import cap_binds, draw_fills
from .strategies import Quoter

# warn if the min(1, .) cap binds on more than this fraction of quote-sides
CAP_WARN_FRACTION = 1e-3


@dataclass(frozen=True)
class State:
    """Session state; every field is an array of shape (n,)."""

    m: np.ndarray
    q: np.ndarray
    cash: np.ndarray
    spread_pnl: np.ndarray
    inventory_pnl: np.ndarray
    bid_fills: np.ndarray
    ask_fills: np.ndarray
    clamp_steps: np.ndarray  # steps where the min_dist floor bound on either side
    cap_bind_sides: np.ndarray  # quote-sides where min(1,.) was active
    book: "bk.BookState | None" = None  # queue model only
    requotes: np.ndarray | None = None  # queue model only: priority-losing requotes
    outside_steps: np.ndarray | None = None  # queue model only: quote-sides out of window

    @property
    def mtm(self) -> np.ndarray:
        """Pi = X + q*m."""
        return self.cash + self.q * self.m


@dataclass(frozen=True)
class StepInfo:
    """What happened this tick (for plotting / dashboards). Fills are SIZES."""

    bid: np.ndarray
    ask: np.ndarray
    f_bid: np.ndarray
    f_ask: np.ndarray


def initial_state(m0, n: int = 1, queue: QueueConfig | None = None,
                  u_depth0: np.ndarray | None = None) -> State:
    """Fresh state. With `queue`, also seed the order book from `u_depth0` (n, 2, MAX)."""
    m0 = np.full(n, float(m0)) if np.ndim(m0) == 0 else np.asarray(m0, dtype=float)
    z = np.zeros(n)
    zi = np.zeros(n, dtype=np.int64)
    book = requotes = outside = None
    if queue is not None:
        if u_depth0 is None:
            raise ValueError("the queue model needs u_depth0 to seed the book")
        book = bk.initial_state(m0, u_depth0, queue)
        requotes, outside = np.zeros((n, 2), dtype=np.int64), np.zeros((n, 2), dtype=np.int64)
    return State(m0, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(),
                 zi.copy(), zi.copy(), book, requotes, outside)


def step(state: State, m_next, u_bid, u_ask, quoter: Quoter, fills: FillConfig,
         dt: float, t: float, informed=None, fill_boost: float = 1.0,
         queue: QueueConfig | None = None, u_queue=None, depth_mult=None, tilt=None
         ) -> tuple[State, StepInfo]:
    """Advance one tick. Pure: inputs are not modified.

    informed: optional (n,) array in {-1, 0, +1}, the active informed direction
    for each session at this tick (None = no adverse selection).
    queue:    None  -> exponential fills (bit-identical to the pre-queue engine);
              QueueConfig -> queue-position fills, which need `u_queue` (n, 8),
              `depth_mult` (n, 2, levels) and `tilt` (n,) for this tick.
    """
    m_next = np.asarray(m_next, dtype=float)
    # 1. quotes (information at step n only), floored at min_dist
    d_bid_raw, d_ask_raw = quoter.distances(state.m, state.q, t)
    d_bid = np.maximum(d_bid_raw, fills.min_dist)
    d_ask = np.maximum(d_ask_raw, fills.min_dist)
    informed = None if informed is None else np.asarray(informed)
    book = state.book
    requotes, outside = state.requotes, state.outside_steps
    cap_sides = state.cap_bind_sides
    if queue is None:
        # 2a. exponential intensity; informed flow boosts the picked-off side
        bid, ask = state.m - d_bid, state.m + d_ask
        if informed is None:
            mult_bid = mult_ask = 1.0
        else:
            mult_bid = np.where(informed < 0, fill_boost, 1.0)
            mult_ask = np.where(informed > 0, fill_boost, 1.0)
        f_bid = draw_fills(d_bid, u_bid, fills, dt, mult_bid).astype(float)
        f_ask = draw_fills(d_ask, u_ask, fills, dt, mult_ask).astype(float)
        cap_sides = cap_sides + cap_binds(d_bid, fills, dt, mult_bid) \
            + cap_binds(d_ask, fills, dt, mult_ask)
    else:
        # 2b. queue position: post (losing priority if the price moved), let the
        #     book evolve, then let market-order flow consume it.
        if book is None:
            raise ValueError("queue fills need a State built with initial_state(queue=...)")
        zeros = np.zeros(state.m.shape[0])
        want = bk.quote_grids(d_bid, d_ask, state.m, queue)
        book = bk.place(book, want, queue)
        # measured HERE, while the order is certainly live: "resting but beyond the
        # modelled window, so unreachable". After `trade` a filled order is not live
        # and would be miscounted as outside.
        outside = outside + (bk.our_level(book, queue) < 0)
        book = bk.flow(book, u_queue, zeros if tilt is None else tilt, depth_mult, queue)
        book, filled, _ = bk.trade(book, u_queue,
                                   zeros if informed is None else informed, queue)
        bid = want[:, bk.BID] * queue.tick
        ask = want[:, bk.ASK] * queue.tick
        f_bid, f_ask = filled[:, bk.BID], filled[:, bk.ASK]
        requotes = requotes + (book.requotes - state.book.requotes)
    # 3. inventory and cash (sizes, so partial fills need no special case)
    q_next = state.q + f_bid - f_ask
    cash_next = state.cash - f_bid * bid + f_ask * ask
    # 4-5. mid moves after fills; decomposition increments, from EXECUTED prices
    new = State(
        m=m_next,
        q=q_next,
        cash=cash_next,
        spread_pnl=state.spread_pnl + f_bid * (state.m - bid) + f_ask * (ask - state.m),
        inventory_pnl=state.inventory_pnl + q_next * (m_next - state.m),
        bid_fills=state.bid_fills + f_bid,
        ask_fills=state.ask_fills + f_ask,
        clamp_steps=state.clamp_steps
        + ((d_bid_raw < fills.min_dist) | (d_ask_raw < fills.min_dist)),
        cap_bind_sides=cap_sides,
        book=book if queue is None else bk.reanchor(
            book, m_next, depth_mult, np.zeros(state.m.shape[0]) if tilt is None else tilt,
            queue),
        requotes=requotes,
        outside_steps=outside,
    )
    return new, StepInfo(bid, ask, f_bid, f_ask)


def terminal_pnl(state: State, liquidation_penalty: float = 0.0) -> np.ndarray:
    """Pi_N - c*|q_N| (inventory liquidated at the mid)."""
    return state.mtm - liquidation_penalty * np.abs(state.q)


def simulate_paths(mids: np.ndarray, u_bid: np.ndarray, u_ask: np.ndarray,
                   quoter: Quoter, fills: FillConfig, dt: float,
                   liquidation_penalty: float = 0.0, informed: np.ndarray | None = None,
                   fill_boost: float = 1.0, queue: QueueConfig | None = None,
                   u_queue: np.ndarray | None = None, u_depth0: np.ndarray | None = None,
                   tilt: np.ndarray | None = None) -> dict:
    """Run whole sessions by looping `step`. mids (n, N+1); u_* (n, N);
    informed (n, N) in {-1, 0, +1} or None. For the queue model also pass
    u_queue (n, N, 8), u_depth0 (n, 2, MAX_LEVELS) and tilt (n, N).

    Returns per-step paths (q, mtm, bid, ask, fills) plus final State and
    terminal P&L. This is a thin loop - all accounting is in `step`.
    """
    n, N = u_bid.shape
    depth_mult = None
    if queue is not None:
        depth_mult = bk.session_multipliers(u_depth0, queue)
    st = initial_state(mids[:, 0].copy(), n, queue, u_depth0)
    q = np.zeros((n, N + 1))
    mtm = np.zeros((n, N + 1))
    bid = np.zeros((n, N))
    ask = np.zeros((n, N))
    f_bid = np.zeros((n, N))
    f_ask = np.zeros((n, N))
    for i in range(N):
        st, info = step(st, mids[:, i + 1], u_bid[:, i], u_ask[:, i], quoter, fills, dt, i * dt,
                        None if informed is None else informed[:, i], fill_boost,
                        queue=queue,
                        u_queue=None if queue is None else u_queue[:, i, :],
                        depth_mult=depth_mult,
                        tilt=None if tilt is None else tilt[:, i])
        q[:, i + 1], mtm[:, i + 1] = st.q, st.mtm
        bid[:, i], ask[:, i] = info.bid, info.ask
        f_bid[:, i], f_ask[:, i] = info.f_bid, info.f_ask
    frac = st.cap_bind_sides.sum() / (2 * n * N)
    if queue is None and frac > CAP_WARN_FRACTION:
        warnings.warn(f"fill-probability cap min(1,.) bound on {frac:.2%} of quote-sides; "
                      "results are sensitive to the cap.", stacklevel=2)
    out = dict(state=st, q=q, mtm=mtm, bid=bid, ask=ask, f_bid=f_bid, f_ask=f_ask,
               terminal_pnl=terminal_pnl(st, liquidation_penalty), cap_fraction=frac)
    if queue is not None:
        out["requote_fraction"] = st.requotes.sum() / (2 * n * N)
        out["outside_fraction"] = st.outside_steps.sum() / (2 * n * N)
    return out
