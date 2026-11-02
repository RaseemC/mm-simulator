"""Freeze and reproduce: turn an epoch into a headless, self-checking script.

Concept: a live session is only evidence if it can be re-run without the GUI. An
epoch fixes everything that determines its sessions: the config, the seed and the
index of its first session (session i draws from session_rng(seed, i), so
sessions are reproducible without replaying earlier ones). `reproduction_script`
renders that as Python source that
  * rebuilds the run from the config (exact float reprs, no rounding),
  * re-runs the epoch's completed sessions, and
  * ASSERTS that the aware / naive / paired-difference statistics equal the ones
    recorded live (np.allclose, rtol 1e-9),
so a script that runs silently has reproduced the epoch.

Two forms:
  "live"  : drives `LiveMarketMaker` tick by tick (always possible, covers
            Auto-eta and Auto-widen, which are live-only strategies).
  "batch" : `prepare_batch` + `simulate_paths`, the vectorised engine. Only
            possible when Auto-eta and Auto-widen are off, because the batch
            engine has no detector or hysteretic quoter.
The live detector/book constants are module-level in live_sim; they are listed in
a comment so a change in code version is visible.

Limitation: reproduces from the code version that generated it (same
draws.py/adverse.py/live_sim.py constants); there is no version stamp beyond the
constants listed.
"""
from __future__ import annotations

import numpy as np

from . import live_sim as ls
from .config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from .epochs import Epoch

_STAT_KEYS = ("n", "mean", "naive_mean", "diff_mean", "diff_se")


def _expected(epoch: Epoch) -> dict | None:
    if not epoch.sessions:
        return None
    st = epoch.stats()
    return {k: (int(st[k]) if k == "n" else float(st[k])) for k in _STAT_KEYS}


def _header(epoch: Epoch, n: int) -> str:
    cfg = epoch.config
    first = epoch.start_session if epoch.start_session is not None else 0
    last = first + n - 1
    return (f"# Reproduce epoch #{epoch.id}: seed={cfg['seed']!r}, sessions {first}..{last} "
            f"({n} completed, {len(epoch.interrupted)} interrupted)\n"
            f"# config: {cfg!r}\n")


def _py(v) -> str:
    """repr that stays valid Python for nan."""
    return "float('nan')" if isinstance(v, float) and v != v else repr(v)


def _check(expected: dict | None) -> str:
    if expected is None:
        return ("# (this epoch had no completed sessions when frozen: nothing to compare against)\n"
                "print(result)\n")
    body = "{" + ", ".join(f"{k!r}: {_py(v)}" for k, v in expected.items()) + "}"
    return (f"expected = {body}\n"
            "print('reproduced:', result)\n"
            "print('expected  :', expected)\n"
            "assert result['n'] == expected['n']\n"
            "for k in ('mean', 'naive_mean', 'diff_mean', 'diff_se'):\n"
            "    assert np.allclose(result[k], expected[k], rtol=1e-9, atol=1e-9, equal_nan=True), k\n"
            "print('OK: epoch reproduced')\n")


_RESULT = (
    "aware = np.array(aware, dtype=float)\n"
    "naive = np.array(naive, dtype=float)\n"
    "d = aware - naive\n"
    "result = dict(n=len(d), mean=float(aware.mean()), naive_mean=float(naive.mean()),\n"
    "              diff_mean=float(d.mean()),\n"
    "              diff_se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'))\n"
)


def _queue_cfg(cfg: dict) -> QueueConfig | None:
    """The epoch's queue-model config, or None if it ran exponential fills.
    Epochs recorded before queue settings existed reproduce with the defaults."""
    if not cfg.get("queue_fills", False):
        return None
    return QueueConfig(**{k: cfg[k] for k in ls.QUEUE_KEYS if k in cfg})


def batch_possible(epoch: Epoch) -> bool:
    return not (epoch.config["auto_eta"] or epoch.config["auto_widen"])


def live_script(epoch: Epoch) -> str:
    cfg = epoch.config
    n = len(epoch.sessions) or 20
    params = {k: cfg[k] for k in ls.PARAM_KEYS}
    first = epoch.start_session if epoch.start_session is not None else 0
    consts = {k: getattr(ls, k) for k in ("BOOK_TILT", "BOOK_NOISE", "LEVEL_NOISE", "HORIZON",
                                          "IMB_ALPHA", "ALERT", "WIDEN_MAX", "SKEW_COEF")}
    qc = _queue_cfg(cfg)
    return (
        _header(epoch, n) +
        f"# live_sim constants at freeze time: {consts!r}\n"
        "import numpy as np\n"
        "from mmsim.config import QueueConfig\n"
        "from mmsim.live_sim import LiveMarketMaker\n\n"
        f"mm = LiveMarketMaker(seed={cfg['seed']!r}, session_steps={cfg['session_steps']!r},\n"
        f"                     params={params!r},\n"
        f"                     auto_eta={cfg['auto_eta']!r}, auto_widen={cfg['auto_widen']!r},\n"
        f"                     queue_fills={qc is not None!r}, queue={qc!r},\n"
        f"                     first_session={first!r})\n"
        f"for _ in range({n} * mm.session_steps):\n"
        "    mm.step()\n"
        "recs = mm.epochs.current.sessions\n"
        "aware = [r.terminal_pnl for r in recs]\n"
        "naive = [r.naive_terminal_pnl for r in recs]\n"
        + _RESULT + _check(_expected(epoch)))


def batch_script(epoch: Epoch) -> str:
    if not batch_possible(epoch):
        raise ValueError("batch form needs auto_eta and auto_widen off (live-only strategies)")
    cfg = epoch.config
    n = len(epoch.sessions) or 20
    first = epoch.start_session if epoch.start_session is not None else 0
    market = MarketConfig(m0=ls.M0, T=1.0, N=int(cfg["session_steps"]), sigma=cfg["sigma"])
    fills = FillConfig(A=cfg["A"], k=cfg["k"])
    adverse = AdverseConfig(strength=cfg["tox"])
    qc = _queue_cfg(cfg)
    queue_kw = ("queue=queue, u_queue=prep.u_queue, u_depth0=prep.u_depth0, tilt=prep.tilt"
                if qc is not None else "queue=None")
    return (
        _header(epoch, n) +
        "import numpy as np\n"
        "from mmsim.adverse import prepare_batch\n"
        "from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig\n"
        "from mmsim.core import simulate_paths\n"
        "from mmsim.midprice import ArithmeticRandomWalk\n"
        "from mmsim.strategies import InventorySkewQuoter\n\n"
        f"market = {market!r}\n"
        f"fills = {fills!r}\n"
        f"adverse = {adverse!r}\n"
        f"queue = {qc!r}   # None = exponential fills\n"
        f"prep = prepare_batch(ArithmeticRandomWalk(market), {cfg['seed']!r}, {n}, adverse, "
        f"first={first!r}, want_queue=queue is not None)\n\n"
        "def terminal(eta):\n"
        f"    quoter = InventorySkewQuoter({cfg['delta']!r}, eta)\n"
        "    return simulate_paths(prep.mids, prep.u_bid, prep.u_ask, quoter, fills, market.dt,\n"
        "                          informed=prep.informed, fill_boost=adverse.fill_boost,\n"
        f"                          {queue_kw})['terminal_pnl']\n\n"
        f"aware, naive = terminal({cfg['eta']!r}), terminal(0.0)\n"
        + _RESULT + _check(_expected(epoch)))


def reproduction_report(epoch: Epoch) -> str:
    """Everything the Freeze button prints."""
    parts = ["=" * 78,
             f"FREEZE: epoch #{epoch.id}  (paste into a script; it asserts it reproduces the live numbers)",
             "=" * 78, "# ---- live form ----", live_script(epoch)]
    if batch_possible(epoch):
        parts += ["# ---- batch form (same numbers, vectorised engine) ----", batch_script(epoch)]
    else:
        parts.append("# (batch form unavailable: Auto-eta / Auto-widen are live-only strategies)")
    return "\n".join(parts)
