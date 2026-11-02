"""Phase A verification (A2 accounting, A3 detector). Prints a report; writes nothing.

Usage: python scripts/verify_phase_a.py [--sessions 1000]
Seeded, fixed-parameter runs (no slider changes): reproducible. Seeds 0-7 are the
DEV seeds (looked at while the detector constants were chosen), 1000-1007 are
HELD-OUT. Live/batch equivalence now lives in tests/test_equivalence.py; the
drift calibration lives in scripts/calibrate_toxicity.py.
"""
import argparse
import math

import numpy as np

from mmsim import live_sim as ls
from mmsim.config import AdverseConfig

N = ls.SESSION_STEPS
DEFAULT_TOX = AdverseConfig().strength


def run(seed, sessions, **params):
    """Fixed-parameter run; per-tick arrays incl. the active-only ground truth."""
    mm = ls.LiveMarketMaker(seed=seed, params=params)
    cols = {k: [] for k in ls.Tick._fields}
    cols["active_only"] = []
    for _ in range(sessions * N):
        cols["active_only"].append(bool(mm._informed[mm.session_step] != 0))   # pre-step read
        t = mm.step()
        for k, v in zip(ls.Tick._fields, t):
            cols[k].append(v)
        if t.completed:                                  # boundary must be a clean slate
            assert (mm.q, mm.X, mm.spread_pnl, mm.inventory_pnl) == (0, 0.0, 0.0, 0.0)
            assert mm.session_step == 0 and mm.m == ls.M0
    return mm, {k: np.array(v) for k, v in cols.items()}


def a2(sessions):
    print("=" * 78 + "\nA2  P&L vs generator over the whole run\n" + "=" * 78)
    for tox in (0.0, DEFAULT_TOX):
        mm, c = run(12345, sessions, tox=tox)
        sig = mm.params["sigma"]
        recs = mm.epochs.current.sessions
        aq = np.abs(c["q"])
        dm = c["m_new"] - c["m_old"]
        t_stat = dm.mean() / (dm.std(ddof=1) / math.sqrt(dm.size))
        inv_counter = sum(r.inventory_pnl for r in recs)
        inv_offline = float(np.sum(c["q"] * dm))          # sum q_{n+1}*(m_{n+1}-m_n)
        sd_inv = math.sqrt(np.sum(c["q"].astype(float) ** 2) * sig**2 * ls.DT)
        print(f"\n--- tox={tox}  seed=12345  sessions={sessions}  ticks={dm.size}  sigma={sig}")
        print(f"completed sessions        {len(recs)} (expected {sessions}); incomplete "
              f"{mm.epochs.n_incomplete}; q/cash/accumulators == 0 asserted at every boundary")
        print(f"|q| over WHOLE run        max {aq.max():d}   p1 {np.percentile(aq, 1):.0f}   "
              f"p50 {np.percentile(aq, 50):.0f}   p99 {np.percentile(aq, 99):.0f}")
        print(f"inventory P&L (counters)  {inv_counter:12.3f}")
        print(f"inventory P&L (offline)   {inv_offline:12.3f}   diff {inv_counter - inv_offline:+.2e}")
        print(f"terminal P&L identity     max|Pi - (spread+inv)| = "
              f"{max(abs(r.terminal_pnl - r.spread_pnl - r.inventory_pnl) for r in recs):.1e}")
        print(f"mean dm per tick          {dm.mean():+.5f}   t-stat vs 0 = {t_stat:+.2f}")
        print(f"std dm per tick           {dm.std(ddof=1):.5f}   theory sigma*sqrt(dt) = "
              f"{sig * math.sqrt(ls.DT):.5f}   ratio {dm.std(ddof=1) / (sig * math.sqrt(ls.DT)):.3f}")
        print(f"inventory P&L z-score     {inv_offline / sd_inv:+.2f}   "
              f"(total / sqrt(sum q^2 sigma^2 dt); |z|<3 = consistent with a driftless mid)")


class Poison:
    """Stands in for the hidden episode arrays; ANY use raises."""
    def _boom(self, *a, **k):
        raise AssertionError("detector read hidden regime state")
    __getitem__ = __iter__ = __len__ = __bool__ = _boom


def poison_test():
    """Run the detector's functions with the hidden episode arrays replaced."""
    mm = ls.LiveMarketMaker(seed=3, params=dict(tox=1.0))
    originals = {n: getattr(ls.LiveMarketMaker, n) for n in ("_detect", "_mature_markouts", "_sd")}

    def wrap(fn):
        def inner(self, *a, **k):
            saved = (self._informed, self._tilt)
            self._informed = self._tilt = Poison()
            try:
                return fn(self, *a, **k)
            finally:
                self._informed, self._tilt = saved
        return inner
    for n, fn in originals.items():
        setattr(ls.LiveMarketMaker, n, wrap(fn))
    try:
        for _ in range(5 * N):
            mm.step()
        return "PASS: _detect, _mature_markouts, _sd never touched the hidden episode arrays"
    except AssertionError as e:
        return f"FAIL: {e}"
    finally:
        for n, fn in originals.items():
            setattr(ls.LiveMarketMaker, n, fn)


def detection(seeds, sessions, label, detector=None, **params):
    cols = {"alert": [], "informed": [], "active_only": []}
    for s in seeds:
        mm = ls.LiveMarketMaker(seed=s, params=params)
        if detector:
            mm._detect = detector.__get__(mm)
        for _ in range(sessions * N):
            act = bool(mm._informed[mm.session_step] != 0)
            t = mm.step()
            cols["alert"].append(t.alert)
            cols["informed"].append(t.informed)
            cols["active_only"].append(act)
    a, inf, act = (np.array(cols[k]) for k in ("alert", "informed", "active_only"))
    print(f"{label:<36} detect {a[inf].mean():6.1%}   false(corrected) {a[~inf].mean():6.2%}   "
          f"false(old: lead-in counted quiet) {a[~act].mean():6.2%}   alert base rate {a.mean():6.1%}")
    return a, inf


def a3(sessions):
    print("\n" + "=" * 78 + "\nA3  Detector audit (corrected label = lead-in + active = 'informed')\n" + "=" * 78)
    print(poison_test())
    dev, held = tuple(range(8)), tuple(range(1000, 1008))
    a, y = detection(dev, sessions, "DEV seeds 0-7, full detector")
    detection(held, sessions, "HELD-OUT seeds 1000-1007")
    rng = np.random.default_rng(0)
    hits = [a[rng.permutation(y)].mean() for _ in range(500)]
    print(f"labels randomly permuted (500x):     detect {np.mean(hits):6.1%} "
          f"[{np.percentile(hits, 2.5):.1%}, {np.percentile(hits, 97.5):.1%}]   <- the alert base rate")
    print("\nAblations on HELD-OUT seeds (what is the book, what is the fills?)")

    def book_only(self):
        self.score = float(min(1.0, 0.6 * np.clip((abs(self.imb) - 0.15) / 0.45, 0, 1)))

    def mark_only(self):
        self.score = float(min(1.0, 0.6 * np.clip(-self.mk, 0, 1)))
    detection(held, sessions, "book term only", book_only)
    detection(held, sessions, "markout term only", mark_only)
    old = ls.BOOK_TILT
    ls.BOOK_TILT = 0.0
    try:
        detection(held, sessions, "full detector, BOOK_TILT=0")
    finally:
        ls.BOOK_TILT = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=1000)
    args = ap.parse_args()
    a2(args.sessions)
    a3(100)


if __name__ == "__main__":
    main()
