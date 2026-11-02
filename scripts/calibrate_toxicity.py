"""Calibrate the default informed-drift strength. Prints a table; writes nothing.

For each strength s (drift per active tick = s * sigma * sqrt(dt)) report, from
seeded batch draws:
  RV infl.    E[sum_n (dm_n)^2] / E0[...] - 1           (realised variance, the
              sum of squared tick changes over a session)
  Var(dm_T)   E[(m_N - m_0)^2] / E0[(m_N - m_0)^2] - 1  (variance of the session's
              net move; this is what inventory risk feels)
E0 is the same draws with strength 0 (common random numbers), so the columns
are paired differences, not noisy absolute estimates.
  informed %  share of ticks that are active / lead-in
  P(fall>=48) chance a 380-tick window at sigma = 4.86 falls by >= 48 points
              (the move you flagged), from a long continuous stream
Realised variance is nearly blind to drift (a persistent drift adds only s^2 to
the per-tick second moment but compounds in the net move), which is why the two
inflation columns disagree so much.

Usage: python scripts/calibrate_toxicity.py [--sessions 20000]
"""
import argparse
import math

import numpy as np

from mmsim.adverse import build_regime, informed_drift
from mmsim.config import AdverseConfig, MarketConfig
from mmsim.draws import SessionDraws, draw_session
from mmsim.midprice import ArithmeticRandomWalk

GRID = (0.0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.00)


def session_stats(s, draws, mk):
    cfg = AdverseConfig(strength=s)
    proc = ArithmeticRandomWalk(mk)
    reg = build_regime(draws.u_start, draws.u_dir, draws.u_dur, cfg)
    mids = proc.path_from_shocks(draws.shocks, informed_drift(reg.informed, cfg, mk))
    rv = (np.diff(mids, axis=1) ** 2).sum(axis=1).mean()
    disp = np.mean((mids[:, -1] - mids[:, 0]) ** 2)
    return rv, disp, (reg.informed != 0).mean(), (reg.tilt != 0).mean()


def stack_draws(proc, seed, n):
    return SessionDraws.stack(draw_session(proc, seed, i) for i in range(n))


def window_prob(s, draws, mk, window=380, thresh=48.0):
    cfg = AdverseConfig(strength=s)
    reg = build_regime(draws.u_start, draws.u_dir, draws.u_dur, cfg)
    mids = ArithmeticRandomWalk(mk).path_from_shocks(
        draws.shocks, informed_drift(reg.informed, cfg, mk))
    d = (mids[:, window:] - mids[:, :-window])[:, 500:]          # drop burn-in
    return (d <= -thresh).mean(), (np.abs(d) >= thresh).mean(), d.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20000)
    args = ap.parse_args()
    sd_win = 4.86 * math.sqrt(0.005) * math.sqrt(380)
    mk_s = MarketConfig(sigma=2.0)                                  # N = 200, dt = 0.005
    d_s = stack_draws(ArithmeticRandomWalk(mk_s), 2024, args.sessions)
    ticks = 20_000                                                  # long rows, dt = 0.005
    mk_w = MarketConfig(sigma=4.86, T=ticks * 0.005, N=ticks)
    d_w = stack_draws(ArithmeticRandomWalk(mk_w), 77, 400)
    rv0, disp0, _, _ = session_stats(0.0, d_s, mk_s)
    print(f"{'s':>5} | {'RV infl.':>8} {'Var(dm_T)':>9} {'active%':>8} {'lead+act%':>9} | "
          f"{'P(fall>=48)':>11} {'P(|mv|>=48)':>11} {'sd380':>7}   (driftless sd380 = {sd_win:.2f})")
    for s in GRID:
        rv, disp, act, tilt = session_stats(s, d_s, mk_s)
        pf, pa, sd = window_prob(s, d_w, mk_w)
        print(f"{s:>5.2f} | {rv / rv0 - 1:>+8.1%} {disp / disp0 - 1:>+9.1%} {act:>8.1%} {tilt:>9.1%} | "
              f"{pf:>11.2e} {pa:>11.2e} {sd:>7.2f}")


if __name__ == "__main__":
    main()
