"""M1 demo: simulate one session for both strategies and save figures/single_session.png.

Usage: python scripts/run_single_session.py [--seed 12345] [--session 0] [--eta 0.1]
Prints the config first; writes only the PNG.
"""
import argparse
import dataclasses

import numpy as np

from mmsim.config import SimConfig, StrategyConfig
from mmsim.core import simulate_paths
from mmsim.draws import draw_session
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.plots import plot_single_session
from mmsim.strategies import InventorySkewQuoter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--session", type=int, default=0)
    p.add_argument("--eta", type=float, default=0.1)
    args = p.parse_args()

    cfg = SimConfig(seed=args.seed, aware=StrategyConfig(eta=args.eta))
    print("mode: single session (M1)")
    print(f"config: {cfg}")
    print(f"session index: {args.session}")

    proc = ArithmeticRandomWalk(cfg.market)
    d = draw_session(proc, cfg.seed, args.session)
    mids = proc.path_from_shocks(d.shocks)[None, :]
    ub, ua = d.u_bid[None, :], d.u_ask[None, :]
    out = {}
    for name, sc in (("naive", cfg.naive), ("aware", cfg.aware)):
        out[name] = simulate_paths(mids, ub, ua, InventorySkewQuoter.from_config(sc),
                                   cfg.fills, cfg.market.dt, cfg.liquidation_penalty)
        s = out[name]["state"]
        print(f"{name:>5}: terminal P&L={out[name]['terminal_pnl'][0]:8.3f}  "
              f"spread_pnl={s.spread_pnl[0]:7.3f}  inventory_pnl={s.inventory_pnl[0]:7.3f}  "
              f"bid fills={s.bid_fills[0]}  ask fills={s.ask_fills[0]}  final q={s.q[0]:.0f}")
    path = plot_single_session(mids[0], out["naive"], out["aware"], cfg.market.dt)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
