"""Limit order book with queue position: the state model behind queue-driven fills.

Concept
-------
Prices live on an ABSOLUTE grid of size `tick`; grid index g means price g*tick.
That matters: queue position is only meaningful if a level is a fixed price, so
the book is stored relative to the mid but our resting order is tracked by its
absolute grid index. When the mid moves, the window slides over the order; the
order does not move with it.

    anchor      = floor(m / tick)              (grid index just below the mid)
    bid level j = anchor - j                   j = 0 .. levels-1  (0 = touch)
    ask level j = anchor + 1 + j

A quote further than `levels * tick` from the mid rests OUTSIDE the modelled
window: it keeps its price and its queue position but cannot be filled, since no
market order large enough to sweep that far is modelled. It becomes fillable
again if the mid comes back to it. `outside_fraction` reports how often this
happens, because a model boundary that binds often is a result, not a detail.

`depth[n, side, j]` is OTHER participants' resting size at level j (side 0 = bid,
1 = ask). Our own order is held separately:

    rest[n, side]    absolute grid index where our order rests
    ahead[n, side]   how much of that level's depth is in front of us
    size[n, side]    our remaining resting size
    live[n, side]    whether we have an order resting at all

so `behind = depth[our level] - ahead` is derived, never stored, and the
invariant 0 <= ahead <= depth[our level] is checked by the tests.

Order of operations in one tick (fixed, and no lookahead - every input is known
at the start of the tick):

  1. REQUOTE   the quoter's desired prices are rounded to the grid. If a desired
               price differs from where we rest, the old order is CANCELLED and
               the new one joins the BACK of the new level's queue:
                   ahead := depth[new level],  size := our_size
               If the price is unchanged, queue position is kept. This is the
               priority cost of requoting (B+2c).
  2. CANCELS   a fraction c ~ U(0, 2*cancel_rate) of resting size is cancelled at
               every level. Cancels are assumed uniform over the queue, so the
               same fraction comes out of `ahead`: our priority improves.
  3. ADDS      new size joins the BACK of each level: depth grows, `ahead` does
               not. Depth is also pulled toward its target by `replenish_rate`.
  4. TRADES    market orders arrive (probability `mo_rate` per side, size
               Exp(`mo_size_mean`)) and consume depth from the touch outward. A
               market BUY walks the ask side, a market SELL the bid side. Within
               our level the consumption order is: everything AHEAD of us, then
               US (this is the fill), then everything behind.
  5. the caller moves the mid, then calls `reanchor` so the window follows it.

Adverse selection enters at step 4: while an informed episode is ACTIVE the
market-order rate on the side about to be picked off is multiplied by
`informed_mo_mult` (direction +1 lifts our ask, -1 hits our bid). The depth tilt
(step 3, from `tilt`) starts in the lead-in window and is what the detector sees.

Everything is vectorised over a leading session axis and is a pure function of
(state, draws, config): batch and live consume identical numbers.

Weaknesses: one order per side; continuous sizes rather than integer lots;
cancels uniform over the queue (real cancels concentrate at the back); a single
cancel fraction and add amount per side per tick rather than per level, so levels
are not idiosyncratically noisy after initialisation; no hidden/iceberg orders,
no self-trade, and our own order never influences anyone else's behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import QueueConfig

BID, ASK = 0, 1
# column layout of SessionDraws.u_queue
MO_ARRIVE, MO_SIZE, CANCEL, ADD = (0, 1), (2, 3), (4, 5), (6, 7)


@dataclass(frozen=True)
class BookState:
    """All arrays carry a leading session axis of length n."""

    anchor: np.ndarray  # (n,) int64 grid index just below the mid
    depth: np.ndarray  # (n, 2, levels) other participants' resting size
    rest: np.ndarray  # (n, 2) int64 absolute grid index of our order
    ahead: np.ndarray  # (n, 2) size in front of us at our level
    size: np.ndarray  # (n, 2) our remaining resting size
    live: np.ndarray  # (n, 2) bool: do we have an order resting
    requotes: np.ndarray  # (n, 2) int64 count of priority-losing requotes
    ticks_resting: np.ndarray  # (n, 2) int64 ticks our current order has rested

    @property
    def n(self) -> int:
        return self.anchor.shape[0]


def grid_of(price: np.ndarray, cfg: QueueConfig) -> np.ndarray:
    """Grid index just below `price`."""
    return np.floor(np.asarray(price, dtype=float) / cfg.tick).astype(np.int64)


def level_prices(anchor: np.ndarray, cfg: QueueConfig) -> tuple[np.ndarray, np.ndarray]:
    """Prices of the modelled bid and ask levels, shape (n, levels) each."""
    j = np.arange(cfg.levels)
    return ((anchor[:, None] - j) * cfg.tick, (anchor[:, None] + 1 + j) * cfg.tick)


def our_level(state: BookState, cfg: QueueConfig) -> np.ndarray:
    """(n, 2) level index of our resting order, or -1 when it is outside the window."""
    j_bid = state.anchor - state.rest[:, BID]
    j_ask = state.rest[:, ASK] - state.anchor - 1
    j = np.stack([j_bid, j_ask], axis=1)
    return np.where(state.live & (j >= 0) & (j < cfg.levels), j, -1)


def target_depth(tilt_dir: np.ndarray, mult: np.ndarray, cfg: QueueConfig) -> np.ndarray:
    """Target profile (n, 2, levels): base * (1 + slope*j) * exp(+-tilt) * M_j."""
    j = np.arange(cfg.levels)
    base = cfg.depth_base * (1 + cfg.depth_slope * j)
    sign = np.array([1.0, -1.0])[None, :, None]  # bid thickens, ask thins, when tilt > 0
    lean = np.exp(cfg.tilt * sign * np.asarray(tilt_dir, dtype=float)[:, None, None])
    return base[None, None, :] * lean * mult


def initial_state(m0: np.ndarray, u_depth0: np.ndarray, cfg: QueueConfig) -> BookState:
    """Fresh book around m0. `u_depth0` is (n, 2, MAX_LEVELS) uniforms."""
    m0 = np.atleast_1d(np.asarray(m0, dtype=float))
    n = m0.shape[0]
    anchor = grid_of(m0, cfg)
    mult = session_multipliers(u_depth0, cfg)
    depth = target_depth(np.zeros(n), mult, cfg)
    z2, zi = np.zeros((n, 2)), np.zeros((n, 2), dtype=np.int64)
    return BookState(anchor=anchor, depth=depth, rest=zi.copy(), ahead=z2.copy(),
                     size=z2.copy(), live=np.zeros((n, 2), dtype=bool),
                     requotes=zi.copy(), ticks_resting=zi.copy())


def session_multipliers(u_depth0: np.ndarray, cfg: QueueConfig) -> np.ndarray:
    """Per-session, per-level lognormal depth multipliers (n, 2, levels)."""
    return _lognormal(np.asarray(u_depth0)[:, :, :cfg.levels], cfg.depth_cv)


def _lognormal(u: np.ndarray, cv: float) -> np.ndarray:
    """Lognormal multipliers of log-sd `cv` from uniforms, mean-one."""
    if cv == 0:
        return np.ones_like(u)
    z = _ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return np.exp(cv * z - 0.5 * cv**2)


def _ppf(u: np.ndarray) -> np.ndarray:
    """Standard-normal quantile (Acklam's rational approximation, |err| < 1.2e-9).

    Used instead of a Generator so the SAME pre-drawn uniforms give the same
    numbers in batch and live mode.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    u = np.asarray(u, dtype=float)
    lo, hi = u < 0.02425, u > 1 - 0.02425
    mid = ~(lo | hi)
    out = np.empty_like(u)
    q = np.sqrt(-2 * np.log(np.where(lo, u, 0.5)))
    out = np.where(lo, (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                   / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), out)
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - u, 0.5)))
    out = np.where(hi, -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                   / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), out)
    q = np.where(mid, u, 0.5) - 0.5
    r = q * q
    out = np.where(mid, (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
                   / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1), out)
    return out


def quote_grids(d_bid, d_ask, m, cfg: QueueConfig) -> np.ndarray:
    """Desired absolute grid indices (n, 2) for quote distances d_bid / d_ask.

    The bid rounds DOWN and the ask rounds UP, so a quote is never placed closer
    to the mid than the quoter asked for.
    """
    m = np.asarray(m, dtype=float)
    g_bid = np.floor((m - np.asarray(d_bid, dtype=float)) / cfg.tick).astype(np.int64)
    g_ask = np.ceil((m + np.asarray(d_ask, dtype=float)) / cfg.tick).astype(np.int64)
    return np.stack([g_bid, g_ask], axis=1)


def place(state: BookState, want: np.ndarray, cfg: QueueConfig) -> BookState:
    """Step 1: post at `want` (n, 2). A changed price loses queue priority."""
    # Posting the SAME price keeps priority, whether or not that price is currently
    # inside the modelled window (an order does not lose its place in the queue just
    # because the mid wandered away from it).
    price_same = state.live & (state.rest == want)
    # The order ALWAYS moves to the wanted price. What varies is whether it keeps
    # its place in the queue: normally only if the price did not change, and under
    # the `keep_priority` counterfactual whenever it was already resting.
    carry = price_same | (state.live if cfg.keep_priority else np.zeros_like(price_same))
    depth_at_want = _depth_at(state, want, cfg)
    return replace(
        state,
        rest=want,
        ahead=np.where(carry, state.ahead, depth_at_want),   # else join the BACK
        size=np.where(carry, state.size, cfg.our_size),
        live=np.ones_like(state.live),
        # count ONLY priority lost by moving a live order's price; re-posting after
        # a complete fill is not a requote cost (there was nothing left to lose)
        requotes=state.requotes + (state.live & (state.rest != want)).astype(np.int64),
        ticks_resting=np.where(price_same, state.ticks_resting + 1, 0),
    )


def clamp_ahead(state: BookState, cfg: QueueConfig) -> BookState:
    """Enforce `ahead <= depth at our level`.

    A level shrinking (replenishment pulling an over-full level down, or the
    window sliding over our order and re-seeding its level) IS, in aggregate,
    size disappearing from in front of us, so our priority improves. Without this
    the queue state can claim more size ahead of us than the level holds.
    """
    inside = our_level(state, cfg) >= 0
    at = _depth_at(state, state.rest, cfg)
    return replace(state, ahead=np.where(inside, np.minimum(state.ahead, at), state.ahead))


def _depth_at(state: BookState, grid: np.ndarray, cfg: QueueConfig) -> np.ndarray:
    """Depth (n, 2) at the given absolute grid indices; 0 outside the window."""
    j_bid = state.anchor - grid[:, BID]
    j_ask = grid[:, ASK] - state.anchor - 1
    out = np.zeros((state.n, 2))
    for side, j in ((BID, j_bid), (ASK, j_ask)):
        ok = (j >= 0) & (j < cfg.levels)
        idx = np.where(ok, j, 0)
        out[:, side] = np.where(ok, np.take_along_axis(
            state.depth[:, side, :], idx[:, None], axis=1)[:, 0], 0.0)
    return out


def flow(state: BookState, u: np.ndarray, tilt_dir: np.ndarray, mult: np.ndarray,
         cfg: QueueConfig) -> BookState:
    """Steps 2-3: cancels (we move up the queue), adds and replenishment (behind us)."""
    u = np.asarray(u, dtype=float)
    frac = np.stack([u[:, CANCEL[0]], u[:, CANCEL[1]]], axis=1) * 2 * cfg.cancel_rate
    depth = state.depth * (1 - frac)[:, :, None]
    ahead = state.ahead * (1 - frac)                     # cancels are uniform over the queue
    target = target_depth(tilt_dir, mult, cfg)
    add = np.stack([u[:, ADD[0]], u[:, ADD[1]]], axis=1) * 2 * cfg.add_rate
    depth = depth + add[:, :, None] * target             # joins the BACK: `ahead` unchanged
    depth = depth + cfg.replenish_rate * (target - depth)
    out = replace(state, depth=np.maximum(depth, 0.0), ahead=np.maximum(ahead, 0.0))
    return clamp_ahead(out, cfg)


def trade(state: BookState, u: np.ndarray, informed: np.ndarray,
          cfg: QueueConfig) -> tuple[BookState, np.ndarray, np.ndarray]:
    """Step 4: market orders consume the book from the touch outward.

    Returns (state, filled (n, 2), traded (n, 2)) where `filled` is OUR filled size
    per side and `traded` is the total size each side's market orders consumed.
    A market BUY (ASK side) is boosted while informed = +1, a SELL while -1.
    """
    u = np.asarray(u, dtype=float)
    informed = np.asarray(informed)
    boost = np.stack([np.where(informed < 0, cfg.informed_mo_mult, 1.0),
                      np.where(informed > 0, cfg.informed_mo_mult, 1.0)], axis=1)
    arrive = np.stack([u[:, MO_ARRIVE[0]], u[:, MO_ARRIVE[1]]], axis=1) < np.minimum(
        1.0, cfg.mo_rate * boost)
    usz = np.stack([u[:, MO_SIZE[0]], u[:, MO_SIZE[1]]], axis=1)
    size = -cfg.mo_size_mean * np.log1p(-np.clip(usz, 0, 1 - 1e-12))   # Exp(mo_size_mean)
    remaining = np.where(arrive, size, 0.0)
    traded = remaining.copy()

    depth = state.depth.copy()
    ahead = state.ahead.copy()
    our_sz = state.size.copy()
    filled = np.zeros((state.n, 2))
    j_ours = our_level(state, cfg)
    for side in (BID, ASK):
        rem = remaining[:, side]
        for j in range(cfg.levels):
            here = j_ours[:, side] == j
            d = depth[:, side, j]
            a = np.where(here, ahead[:, side], d)        # size in front of us at this level
            eat = np.minimum(rem, a)
            rem = rem - eat
            depth[:, side, j] = d - eat
            ahead[:, side] = np.where(here, ahead[:, side] - eat, ahead[:, side])
            f = np.where(here, np.minimum(rem, our_sz[:, side]), 0.0)   # then US
            rem = rem - f
            our_sz[:, side] = our_sz[:, side] - f
            filled[:, side] += f
            behind = np.where(here, depth[:, side, j] - ahead[:, side], 0.0)
            eat2 = np.minimum(rem, np.maximum(behind, 0.0))             # then behind us
            rem = rem - eat2
            depth[:, side, j] -= eat2
        remaining[:, side] = rem
    live = state.live & (our_sz > 1e-12)
    return (replace(state, depth=np.maximum(depth, 0.0), ahead=np.maximum(ahead, 0.0),
                    size=our_sz, live=live),
            filled, traded - remaining)


def reanchor(state: BookState, m_next: np.ndarray, mult: np.ndarray,
             tilt_dir: np.ndarray, cfg: QueueConfig) -> BookState:
    """Step 5: slide the window to the new mid; newly exposed levels start at target.

    Our resting order does NOT move: it is held by absolute grid index, so a mid
    move changes which level it sits at (and can slide it out of the window).
    """
    new_anchor = grid_of(m_next, cfg)
    shift = new_anchor - state.anchor                      # bid level j -> j + shift
    j = np.arange(cfg.levels)[None, :]
    target = target_depth(tilt_dir, mult, cfg)
    depth = state.depth.copy()
    for side, src in ((BID, j - shift[:, None]), (ASK, j + shift[:, None])):
        ok = (src >= 0) & (src < cfg.levels)
        gathered = np.take_along_axis(state.depth[:, side, :], np.clip(src, 0, cfg.levels - 1),
                                      axis=1)
        depth[:, side, :] = np.where(ok, gathered, target[:, side, :])
    # An order outside the modelled window keeps its price AND its queue position:
    # it simply cannot be reached, because no market order large enough to sweep
    # that far is modelled. It becomes fillable again if the mid comes back.
    return clamp_ahead(replace(state, anchor=new_anchor, depth=depth), cfg)

