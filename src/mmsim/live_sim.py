"""Live tick-by-tick market maker: a replay of the batch machinery, one tick at a time.

Concept
-------
Nothing here re-implements accounting. A session is set up exactly as a batch
session is (same per-session draws, same hidden informed episodes, same midprice
with informed drift via `adverse.build_regime`) and is then advanced one tick at
a time through `core.step` with n = 1. So live and batch agree by construction
(tests/test_equivalence.py checks it, including with adverse selection on and at
large eta). Only the synthetic book, the detector and the widening quoter are
live-only.

Paired strategies: every session runs TWO accounting states through `core.step` on
identical mids, fill draws and informed episodes (common random numbers):
"aware" = the configured strategy (eta slider, Auto-eta, Auto-widen) and "naive"
= eta 0, same delta, no detector, no widening. The detector reads the AWARE
strategy's own fills only. Paired difference = aware - naive terminal P&L.

Sessions: SESSION_STEPS ticks (T = 1). At each boundary inventory, cash, the P&L
accumulators, the mid (back to M0) and all detector state reset. Session i draws
from session_rng(seed, i) (draws.py); the synthetic book draws from a separate
stream, spawn_key (i, 1), so book randomness never perturbs the core draws.

Epochs: sliders never write into the running parameters. `request` queues a
change; `apply_pending` applies the queue at a TICK BOUNDARY (top of `step`, or
between frames). A changed config closes the epoch (epochs.py) and discards the
session in flight as incomplete. Because params are constant within a session,
the whole session (draws, regime, midprice path) is prepared at session start.

Adverse selection: see adverse.py. The single toxicity parameter is
params["tox"] = AdverseConfig.strength (drift per informed tick in units of
sigma*sqrt(dt)); 0 turns it off.

Synthetic book (live-only, derived from the hidden direction)
    LEVELS levels per side; depth ~ BOOK_BASE*(1+0.3 i) * lognormal noise, tilted
    by  BOOK_TILT*tilt_direction + AR(1) noise (the noise is a decoy).
    NOTE: the book is GENERATED from the hidden direction, so detection quality
    reflects this book's leakage (BOOK_TILT), not a general adverse-selection
    signal. See docs/DESIGN_NOTES.md.

Detector (inputs: the book now, own fills >= HORIZON ticks old, mid increments)
    imbalance  I  = EWMA of (bid depth - ask depth) / (bid depth + ask depth)
    markout    mk = EWMA of side*(m_now - m_at_fill)/(sd*sqrt(H)) over fills
                    that happened >= H ticks ago (a LAGGED window)
    score         = min(1, 0.6*book_term + 0.6*markout_term); alert >= ALERT
    Scoring label: a tick is "informed" from the start of the lead-in (when the
    book tilt starts) through the end of the active episode.
Response (Auto-widen): half-spread delta*(1 + WIDEN_MAX*score), quotes shifted by
    SKEW_COEF * I * delta.
"""
from __future__ import annotations

import dataclasses
from collections import deque, namedtuple

import numpy as np

from . import book as bk
from .adverse import build_regime, informed_drift
from .config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from .core import initial_state, step as core_step
from .draws import draw_session
from .epochs import EpochLog, SessionRecord
from .midprice import ArithmeticRandomWalk
from .strategies import InventorySkewQuoter, Quoter

M0 = 100.0
SESSION_STEPS = 200  # N, T = 1
DT = 1.0 / SESSION_STEPS
WINDOW = 400        # number of ticks kept for plotting
SEED = 42

# Synthetic book
LEVELS = 5
TICK = 0.10
BOOK_BASE = 50.0
BOOK_TILT = 0.7
BOOK_NOISE = 0.18
LEVEL_NOISE = 0.30

# Detector / response
HORIZON = 20
IMB_ALPHA = 0.25
ALERT = 0.5
WIDEN_MAX = 2.0
SKEW_COEF = 0.5

PARAM_KEYS = ("sigma", "A", "k", "delta", "eta", "tox")
FLAG_KEYS = ("auto_eta", "auto_widen", "queue_fills")
# Queue-model parameters exposed as live settings. Part of the config (so changing
# one opens a new epoch) whether or not the queue model is currently on.
QUEUE_KEYS = ("depth_base", "mo_rate", "mo_size_mean", "informed_mo_mult")
DEFAULT_PARAMS = dict(sigma=2.0, A=140.0, k=1.5, delta=0.75, eta=0.10,
                      tox=AdverseConfig().strength)

# What one tick did (values are post-fill, post-move). `informed` is the scoring
# label: lead-in OR active.
Tick = namedtuple("Tick", "session m_old m_new q f_bid f_ask alert informed completed "
                          "u_bid u_ask pnl q_naive pnl_naive")


@dataclasses.dataclass(frozen=True)
class DisplayBook:
    """What the Book tab draws for one tick. Read-only; never feeds a strategy.

    source "queue": the fill-driving limit order book of the AWARE strategy
                    (book.BookState) - real depth, real queue position.
    source "synthetic": exponential fills have no book; this is the DETECTOR's
                    synthetic book (generated from the hidden direction - see the
                    module docstring), shown for display only.
    Prices are absolute; sizes are other participants' resting size per level.
    `ahead`/`size` are our queue position (nan when not applicable).
    """

    n: int
    source: str
    tick: float
    mid: float
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    our_bid: float
    our_ask: float
    ahead_bid: float
    ahead_ask: float
    size_bid: float
    size_ask: float
    depth_at_bid: float
    depth_at_ask: float
    score: float
    truth: float


class WideningQuoter(Quoter):
    """Live-only: constant-eta skew around an externally set half-spread/shift."""

    def __init__(self, half_spread: float, eta: float, shift: float) -> None:
        self.half_spread, self.eta, self.shift = half_spread, eta, shift

    def distances(self, m, q, t):
        skew = self.eta * q
        return self.half_spread + skew - self.shift, self.half_spread - skew + self.shift


class LiveMarketMaker:
    """Holds parameters and state; step() advances the world by one tick."""

    def __init__(self, seed: int = SEED, session_steps: int = SESSION_STEPS,
                 params: dict | None = None, auto_eta: bool = False,
                 auto_widen: bool = False, first_session: int = 0,
                 queue_fills: bool = False, queue: QueueConfig | None = None) -> None:
        self.first_session = first_session      # index of session 0 of this run (reproduction)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.auto_eta = auto_eta
        self.auto_widen = auto_widen
        self.queue_fills = queue_fills          # False -> exponential fills
        self.queue_cfg = queue or QueueConfig()
        self.paused = False
        self.seed = seed
        self.session_steps = session_steps
        self.reset()

    # -- state views -------------------------------------------------------
    @property
    def m(self) -> float:
        return float(self.state.m[0])

    @property
    def q(self) -> float:
        """Inventory. A float: the queue model allows partial fills."""
        return float(self.state.q[0])

    @property
    def X(self) -> float:
        return float(self.state.cash[0])

    @property
    def spread_pnl(self) -> float:
        return float(self.state.spread_pnl[0])

    @property
    def inventory_pnl(self) -> float:
        return float(self.state.inventory_pnl[0])

    @property
    def n_bid_fills(self) -> float:
        """Total filled SIZE on the bid (integral under the exponential model)."""
        return float(self.state.bid_fills[0])

    @property
    def n_ask_fills(self) -> float:
        return float(self.state.ask_fills[0])

    @property
    def pnl(self) -> float:
        return float(self.state.mtm[0])

    @property
    def q_n(self) -> float:
        return float(self.state_n.q[0])

    @property
    def pnl_n(self) -> float:
        return float(self.state_n.mtm[0])

    # -- config / epochs ---------------------------------------------------
    def config(self) -> dict:
        return dict(**{k: float(self.params[k]) for k in PARAM_KEYS},
                    auto_eta=self.auto_eta, auto_widen=self.auto_widen,
                    queue_fills=self.queue_fills,
                    **{k: float(getattr(self.queue_cfg, k)) for k in QUEUE_KEYS},
                    seed=self.seed, session_steps=self.session_steps)

    def request(self, key: str, value) -> None:
        """Queue a change (called from GUI callbacks). Applied at a tick boundary."""
        if key not in PARAM_KEYS and key not in FLAG_KEYS and key not in QUEUE_KEYS:
            raise KeyError(key)
        self._pending[key] = value

    def apply_pending(self) -> bool:
        """Apply queued changes. Call ONLY between ticks. True if an epoch opened."""
        if not self._pending:
            return False
        for k, v in self._pending.items():
            if k in PARAM_KEYS:
                self.params[k] = float(v)
            elif k in QUEUE_KEYS:
                self.queue_cfg = dataclasses.replace(self.queue_cfg, **{k: float(v)})
            else:
                setattr(self, k, bool(v))
        self._pending.clear()
        in_flight = self.session if self.session_step > 0 else None
        start = self.session + (1 if in_flight is not None else 0)
        changed = self.epochs.update_config(self.config(), interrupted=in_flight,
                                            start_session=start)
        if changed and in_flight is not None:
            self.session += 1          # discard the interrupted session; next index
            self._start_session()
        elif changed:
            self._start_session()      # at a boundary: re-prepare under the new config
        return changed

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._pending = {}
        self.session = self.first_session
        self.n = 0                      # total ticks, x-axis of the plots
        self.epochs = EpochLog(self.config(), start_session=self.first_session)
        self.hist = {key: deque(maxlen=WINDOW) for key in
                     ("n", "mid", "bid", "ask", "bid_n", "ask_n", "q", "pnl", "spread",
                      "inv", "q_n", "pnl_n", "spread_n", "inv_n",
                      "score", "imb", "truth", "sess")}
        self.bid_fill_pts = deque(maxlen=WINDOW)
        self.ask_fill_pts = deque(maxlen=WINDOW)
        self.bid_fill_pts_n = deque(maxlen=WINDOW)
        self.ask_fill_pts_n = deque(maxlen=WINDOW)
        # one DisplayBook per tick, for the ladder / depth heatmap / bubble chart
        self.book_hist = deque(maxlen=WINDOW)
        # Run-wide extremes: every tick of every session so far, interrupted ones
        # included (they happened), across all epochs.
        self.run_max_abs_q = self.run_max_abs_q_n = 0.0
        self.run_worst_dd = self.run_worst_dd_n = 0.0
        self._start_session()

    def _start_session(self) -> None:
        """Prepare session `self.session` exactly as a batch session is prepared."""
        p = self.params
        self.market = MarketConfig(m0=M0, T=1.0, N=self.session_steps, sigma=p["sigma"])
        self.dt = self.market.dt
        self.fills = FillConfig(A=p["A"], k=p["k"])
        self.adverse = AdverseConfig(strength=p["tox"])
        proc = ArithmeticRandomWalk(self.market)
        d = draw_session(proc, self.seed, self.session, want_queue=self.queue_fills)
        reg = build_regime(d.u_start, d.u_dir, d.u_dur, self.adverse)
        drift = informed_drift(reg.informed[0], self.adverse, self.market)
        self._mids = proc.stream(d.shocks, drift)
        next(self._mids)                              # m_0
        self._u_bid, self._u_ask = d.u_bid, d.u_ask
        self._informed, self._tilt = reg.informed[0], reg.tilt[0]
        self._u_queue = d.u_queue
        qc = self.queue_cfg if self.queue_fills else None
        self._depth_mult = (bk.session_multipliers(d.u_depth0[None], qc)
                            if self.queue_fills else None)
        book_rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence(self.seed, spawn_key=(self.session, 1))))
        self._tilt_z = book_rng.standard_normal(self.session_steps)
        self._level_z = book_rng.standard_normal((self.session_steps, 2, LEVELS))

        u0 = d.u_depth0[None] if self.queue_fills else None
        self.state = initial_state(M0, 1, qc, u0)
        self.state_n = initial_state(M0, 1, qc, u0)
        self.session_step = 0
        self.eta_eff = p["eta"]
        self.tilt_noise = 0.0
        self.book = (np.zeros(LEVELS), np.zeros(LEVELS))
        self.imb = 0.0
        self.mk = 0.0
        self.score = 0.0
        self.delta_eff = p["delta"]
        self.pending = deque()                        # (due_step, side, mid_at_fill)
        self.dm = deque(maxlen=100)                   # recent mid increments
        self.n_active = self.n_quiet = self.n_hit = self.n_false = 0
        self.peak = self.peak_n = 0.0
        self.max_dd = self.max_dd_n = 0.0
        self.max_abs_q = self.max_abs_q_n = 0.0

    def display_book(self) -> DisplayBook:
        """Snapshot of the book for display (Book tab). Never read by any strategy."""
        bid_q, ask_q = (self.hist["bid"][-1], self.hist["ask"][-1]) if self.hist["bid"] \
            else (np.nan, np.nan)
        truth = self.hist["truth"][-1] if self.hist["truth"] else 0.0
        if self.queue_fills and self.state.book is not None:
            cfg, bs = self.queue_cfg, self.state.book
            bpx, apx = bk.level_prices(bs.anchor, cfg)
            j = bk.our_level(bs, cfg)[0]
            dep = bs.depth[0]
            at = [float(dep[s, j[s]]) if j[s] >= 0 else np.nan for s in (0, 1)]
            live = bs.live[0]
            return DisplayBook(
                self.n, "queue", cfg.tick, self.m, bpx[0], dep[0].copy(), apx[0],
                dep[1].copy(),
                float(bs.rest[0, 0] * cfg.tick) if live[0] else np.nan,
                float(bs.rest[0, 1] * cfg.tick) if live[1] else np.nan,
                float(bs.ahead[0, 0]) if live[0] else np.nan,
                float(bs.ahead[0, 1]) if live[1] else np.nan,
                float(bs.size[0, 0]) if live[0] else np.nan,
                float(bs.size[0, 1]) if live[1] else np.nan,
                at[0], at[1], self.score, truth)
        bd, ad = self.book
        j = np.arange(LEVELS)
        nan = float("nan")
        return DisplayBook(self.n, "synthetic", TICK, self.m,
                           self.m - TICK * (j + 0.5), np.asarray(bd).copy(),
                           self.m + TICK * (j + 0.5), np.asarray(ad).copy(),
                           bid_q, ask_q, nan, nan, nan, nan, nan, nan, self.score, truth)

    # -- strategy helpers --------------------------------------------------
    def _auto_adjust_eta(self) -> None:
        """Toy supervisor: lean harder when inventory is big, relax when small.
        Acts on eta_eff (state), not on params, so it never opens an epoch."""
        if abs(self.q) >= 6:
            self.eta_eff = min(0.5, self.eta_eff + 0.002)
        elif abs(self.q) <= 2:
            self.eta_eff = max(0.0, self.eta_eff - 0.002)

    # -- synthetic book + detector ----------------------------------------
    def _observe_book(self, n: int, tilt_dir: int) -> None:
        self.tilt_noise = (0.9 * self.tilt_noise
                           + BOOK_NOISE * np.sqrt(1 - 0.81) * self._tilt_z[n])
        tilt = BOOK_TILT * tilt_dir + self.tilt_noise
        base = BOOK_BASE * (1 + 0.3 * np.arange(LEVELS))
        bid = base * np.exp(tilt + LEVEL_NOISE * self._level_z[n, 0])
        ask = base * np.exp(-tilt + LEVEL_NOISE * self._level_z[n, 1])
        self.book = (bid, ask)
        raw = (bid.sum() - ask.sum()) / (bid.sum() + ask.sum())
        self.imb += IMB_ALPHA * (raw - self.imb)

    def _sd(self) -> float:
        if len(self.dm) >= 20:
            return float(np.std(self.dm))
        return self.params["sigma"] * np.sqrt(self.dt)

    def _detect(self) -> None:
        s_book = np.clip((abs(self.imb) - 0.15) / 0.45, 0.0, 1.0)
        s_mark = np.clip(-self.mk, 0.0, 1.0)
        self.score = float(min(1.0, 0.6 * s_book + 0.6 * s_mark))

    def _mature_markouts(self) -> None:
        sd = self._sd()
        while self.pending and self.pending[0][0] <= self.session_step:
            _, side, m0 = self.pending.popleft()
            mo = side * (self.m - m0) / (sd * np.sqrt(HORIZON))
            self.mk += 0.2 * (mo - self.mk)
        self.mk *= 0.995

    # -- one tick ------------------------------------------------------------
    def step(self) -> Tick:
        self.apply_pending()                 # tick boundary: safe place for changes
        p = self.params
        n = self.session_step
        if self.auto_eta:
            self._auto_adjust_eta()
        else:
            self.eta_eff = p["eta"]

        # 0. the book reflects the hidden direction; the detector reads the book
        tilt_dir = int(self._tilt[n])
        self._observe_book(n, tilt_dir)
        self._detect()
        alert = self.score >= ALERT
        informed_label = tilt_dir != 0       # lead-in OR active
        if informed_label:
            self.n_active += 1
            self.n_hit += alert
        else:
            self.n_quiet += 1
            self.n_false += alert

        # 1. quoter from information available NOW (no lookahead)
        w = self.score if self.auto_widen else 0.0
        self.delta_eff += 0.2 * (p["delta"] * (1 + WIDEN_MAX * w) - self.delta_eff)
        if self.auto_widen:
            quoter = WideningQuoter(self.delta_eff, self.eta_eff,
                                    SKEW_COEF * self.imb * p["delta"])
        else:
            quoter = InventorySkewQuoter(p["delta"], self.eta_eff)

        # 2-5. the shared pure step: quotes (floored), fills, accounting
        m_old = self.m
        m_next = next(self._mids)
        u_bid, u_ask = float(self._u_bid[n]), float(self._u_ask[n])
        qc = self.queue_cfg if self.queue_fills else None
        kw = dict(informed=np.array([self._informed[n]]),
                  fill_boost=self.adverse.fill_boost, queue=qc,
                  u_queue=None if qc is None else self._u_queue[n][None, :],
                  depth_mult=self._depth_mult,
                  tilt=None if qc is None else np.array([self._tilt[n]]))
        new, info = core_step(
            self.state, np.array([m_next]), np.array([u_bid]), np.array([u_ask]),
            quoter, self.fills, self.dt, n * self.dt, **kw)
        self.state = new
        # the paired naive strategy: same mid, same draws, same informed flag
        self.state_n, info_n = core_step(
            self.state_n, np.array([m_next]), np.array([u_bid]), np.array([u_ask]),
            InventorySkewQuoter(p["delta"], 0.0), self.fills, self.dt, n * self.dt, **kw)
        f_bid, f_ask = float(info.f_bid[0]), float(info.f_ask[0])
        bid, ask = float(info.bid[0]), float(info.ask[0])
        self.dm.append(m_next - m_old)
        if f_bid:
            self.pending.append((n + HORIZON, +1, m_old))
        if f_ask:
            self.pending.append((n + HORIZON, -1, m_old))

        pnl = self.pnl
        self.n += 1
        self.session_step += 1
        self._mature_markouts()
        pnl_n = self.pnl_n
        self.peak = max(self.peak, pnl)
        self.max_dd = max(self.max_dd, self.peak - pnl)
        self.max_abs_q = max(self.max_abs_q, abs(self.q))
        self.peak_n = max(self.peak_n, pnl_n)
        self.max_dd_n = max(self.max_dd_n, self.peak_n - pnl_n)
        self.max_abs_q_n = max(self.max_abs_q_n, abs(self.q_n))
        self.run_max_abs_q = max(self.run_max_abs_q, self.max_abs_q)
        self.run_max_abs_q_n = max(self.run_max_abs_q_n, self.max_abs_q_n)
        self.run_worst_dd = max(self.run_worst_dd, self.max_dd)
        self.run_worst_dd_n = max(self.run_worst_dd_n, self.max_dd_n)

        h = self.hist
        bid_n, ask_n = float(info_n.bid[0]), float(info_n.ask[0])
        h["n"].append(self.n)
        h["mid"].append(self.m)
        h["bid"].append(bid)
        h["ask"].append(ask)
        h["bid_n"].append(bid_n)
        h["ask_n"].append(ask_n)
        h["q"].append(self.q)
        h["pnl"].append(pnl)
        h["spread"].append(self.spread_pnl)
        h["inv"].append(self.inventory_pnl)
        h["q_n"].append(self.q_n)
        h["pnl_n"].append(pnl_n)
        h["spread_n"].append(float(self.state_n.spread_pnl[0]))
        h["inv_n"].append(float(self.state_n.inventory_pnl[0]))
        h["score"].append(self.score)
        h["imb"].append(self.imb)
        h["truth"].append(1.0 if informed_label else 0.0)
        h["sess"].append(self.session)
        if f_bid:
            self.bid_fill_pts.append((self.n, bid))
        if f_ask:
            self.ask_fill_pts.append((self.n, ask))
        if info_n.f_bid[0] > 0:
            self.bid_fill_pts_n.append((self.n, bid_n))
        if info_n.f_ask[0] > 0:
            self.ask_fill_pts_n.append((self.n, ask_n))
        self.book_hist.append(self.display_book())

        tick = Tick(self.session, m_old, self.m, self.q, f_bid, f_ask, alert, informed_label,
                    self.session_step == self.session_steps, u_bid, u_ask, pnl,
                    self.q_n, pnl_n)
        if tick.completed:
            self._finish_session(pnl, pnl_n)
        return tick

    def _finish_session(self, terminal_pnl: float, terminal_pnl_n: float) -> None:
        self.epochs.add_session(SessionRecord(
            index=self.session, terminal_pnl=terminal_pnl, spread_pnl=self.spread_pnl,
            inventory_pnl=self.inventory_pnl, max_abs_q=self.max_abs_q,
            max_drawdown=self.max_dd, fills=self.n_bid_fills + self.n_ask_fills,
            n_active=self.n_active, n_hit=self.n_hit,
            n_quiet=self.n_quiet, n_false=self.n_false,
            clamp_steps=int(self.state.clamp_steps[0]),
            naive_terminal_pnl=terminal_pnl_n,
            naive_spread_pnl=float(self.state_n.spread_pnl[0]),
            naive_inventory_pnl=float(self.state_n.inventory_pnl[0]),
            naive_max_abs_q=self.max_abs_q_n, naive_max_drawdown=self.max_dd_n))
        self.session += 1
        self._start_session()
