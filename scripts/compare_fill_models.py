"""Exponential vs queue-position fills: composition, priority cost, and when it matters.

Prints three tables and writes figures/fill_models.png. Nothing else is written.

  COMPOSITION (B+2b)  the two models can be made to agree on the MEAN fill rate;
                      this shows where they still disagree - fill size
                      distribution, burstiness (clustering of fills in time) and
                      the correlation between fills and the hidden informed state.
  PRIORITY COST (B+2c) the same run with QueueConfig.keep_priority True vs False.
                      Every draw is shared, so the paired difference is exactly
                      the cost of losing queue position when a quote moves.
  WHEN IT MATTERS      sweep sigma*sqrt(dt)/tick. Priority can only matter if a
                      quote survives long enough to climb a queue, so the cost
                      should grow as the mid becomes slow relative to the grid.

Usage: python scripts/compare_fill_models.py [--sessions 2000]
"""
import argparse

import numpy as np

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from mmsim.core import simulate_paths
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.plots import FIG_DIR
from mmsim.strategies import InventorySkewQuoter


def run(seed, sessions, eta, queue, sigma=2.0, delta=0.75, strength=None, tick=None):
    market = MarketConfig(sigma=sigma)
    adv = AdverseConfig() if strength is None else AdverseConfig(strength=strength)
    qc = queue if queue is None else (queue if tick is None else
                                      QueueConfig(**{**queue.__dict__, "tick": tick}))
    prep = prepare_batch(ArithmeticRandomWalk(market), seed, sessions, adv,
                         want_queue=qc is not None)
    res = simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(delta, eta),
                         FillConfig(), market.dt, informed=prep.informed,
                         fill_boost=adv.fill_boost, queue=qc, u_queue=prep.u_queue,
                         u_depth0=prep.u_depth0, tilt=prep.tilt)
    res["informed"] = prep.informed
    res["prep"] = prep
    return res


def composition(res, label):
    """Fill-size distribution, burstiness, and fill/informed correlation."""
    f = res["f_bid"] + res["f_ask"]
    sizes = np.concatenate([res["f_bid"].ravel(), res["f_ask"].ravel()])
    sizes = sizes[sizes > 0]
    traded = (f > 0).astype(float)
    # burstiness: lag-1 autocorrelation of "did we trade this tick"
    a, b = traded[:, :-1].ravel(), traded[:, 1:].ravel()
    rho = float(np.corrcoef(a, b)[0, 1])
    # does informed flow pick us off? signed fill vs the informed direction
    inf = res["informed"]
    signed = res["f_ask"] - res["f_bid"]        # +1 we sold, -1 we bought
    picked = signed * inf                        # >0 when we traded the wrong way
    mask = inf != 0
    return dict(
        label=label,
        rate=float(f.sum() / (2 * f.size)),
        mean_size=float(sizes.mean()),
        partial=float(np.mean(sizes < 0.999)),
        p99_size=float(np.percentile(sizes, 99)),
        rho=rho,
        picked=float(picked[mask].mean()) if mask.any() else float("nan"),
        rate_informed=float(f[mask].mean() / 2) if mask.any() else float("nan"),
        rate_quiet=float(f[~mask].mean() / 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=2000)
    args = ap.parse_args()
    n, seed = args.sessions, 2024
    qc = QueueConfig()

    print("=" * 100)
    print("COMPOSITION  (eta = 0.1, delta = 0.75, default toxicity; "
          f"{n} sessions x 200 ticks)")
    print("=" * 100)
    rows = []
    for label, q in (("exponential", None), ("queue", qc)):
        rows.append(composition(run(seed, n, 0.1, q), label))
    hdr = (f"{'model':<12} {'fills/t/side':>12} {'mean size':>10} {'partial':>8} {'p99 size':>9} "
           f"{'burstiness':>11} {'picked off':>11} {'rate|informed':>14} {'rate|quiet':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<12} {r['rate']:>12.4f} {r['mean_size']:>10.3f} {r['partial']:>8.1%} "
              f"{r['p99_size']:>9.3f} {r['rho']:>11.4f} {r['picked']:>11.4f} "
              f"{r['rate_informed']:>14.4f} {r['rate_quiet']:>11.4f}")
    print("burstiness = lag-1 autocorrelation of 'traded this tick'; picked off = "
          "E[signed fill x informed direction] (>0 = we trade the wrong way)")

    print("\n" + "=" * 100)
    print("PRIORITY COST  (paired: identical draws, requoting free vs costly)")
    print("=" * 100)
    free = QueueConfig(**{**qc.__dict__, "keep_priority": True})
    hdr = (f"{'eta':>5} {'P&L costly':>11} {'P&L free':>10} {'cost':>9} {'se':>7} {'t':>7} "
           f"{'requotes/t':>11} {'fills costly':>13} {'fills free':>11}")
    print(hdr)
    print("-" * len(hdr))
    cost_by_eta = {}
    for eta in (0.0, 0.1, 0.2, 0.3):
        a = run(seed, n, eta, qc)
        b = run(seed, n, eta, free)
        d = a["terminal_pnl"] - b["terminal_pnl"]
        se = d.std(ddof=1) / np.sqrt(n)
        cost_by_eta[eta] = (d.mean(), se)
        fa = (a["state"].bid_fills + a["state"].ask_fills).sum() / (2 * n * 200)
        fb = (b["state"].bid_fills + b["state"].ask_fills).sum() / (2 * n * 200)
        print(f"{eta:>5} {a['terminal_pnl'].mean():>11.2f} {b['terminal_pnl'].mean():>10.2f} "
              f"{d.mean():>9.2f} {se:>7.2f} {d.mean() / se:>7.2f} "
              f"{a['requote_fraction']:>11.4f} {fa:>13.4f} {fb:>11.4f}")
    print("cost = costly - free, so a NEGATIVE number is what losing priority costs.")

    print("\n" + "=" * 100)
    print("WHEN DOES PRIORITY MATTER?  sweep the mid's speed against the grid")
    print("=" * 100)
    hdr = (f"{'sigma':>6} {'sig*sqrt(dt)/tick':>18} {'ticks per requote':>18} {'requotes/t':>11} "
           f"{'priority cost':>14} {'se':>7} {'as % of P&L':>12}")
    print(hdr)
    print("-" * len(hdr))
    sweep = []
    for sigma in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        a = run(seed, n, 0.1, qc, sigma=sigma)
        b = run(seed, n, 0.1, free, sigma=sigma)
        d = a["terminal_pnl"] - b["terminal_pnl"]
        se = d.std(ddof=1) / np.sqrt(n)
        ratio = sigma * np.sqrt(MarketConfig().dt) / qc.tick
        pct = 100 * d.mean() / abs(b["terminal_pnl"].mean()) if b["terminal_pnl"].mean() else 0
        sweep.append((ratio, d.mean(), se, pct, a["requote_fraction"]))
        print(f"{sigma:>6} {ratio:>18.3f} {1 / a['requote_fraction']:>18.2f} "
              f"{a['requote_fraction']:>11.4f} {d.mean():>14.2f} {se:>7.2f} {pct:>11.1f}%")
    print("sig*sqrt(dt)/tick is how far the mid moves per tick in grid units. Note the two")
    print("questions this separates: a quote SURVIVES longer when the mid is slow (left of the")
    print("table), but the P&L LOST to requoting is largest when the mid is fast, because the")
    print("cost is (how often you requote) x (value of a queue place), and the first term wins.")
    _figure(sweep, cost_by_eta)


def _figure(sweep, cost_by_eta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ratio = [s[0] for s in sweep]
    cost = [s[1] for s in sweep]
    se = [s[2] for s in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(ratio, cost, yerr=[1.96 * x for x in se], marker="o", color="#2a6fbb")
    ax[0].axhline(0, color="#999999", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\sigma\sqrt{dt}\,/\,$tick   (mid speed in grid units)")
    ax[0].set_ylabel("priority cost (P&L)")
    ax[0].set_title("Requote cost vs how fast the mid moves", fontsize=10)
    etas = sorted(cost_by_eta)
    ax[1].errorbar(etas, [cost_by_eta[e][0] for e in etas],
                   yerr=[1.96 * cost_by_eta[e][1] for e in etas], marker="s", color="#d9730d")
    ax[1].axhline(0, color="#999999", lw=0.8)
    ax[1].set_xlabel(r"inventory skew $\eta$")
    ax[1].set_ylabel("priority cost (P&L)")
    ax[1].set_title("Requote cost vs skew strength", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.2)
    fig.suptitle("Cost of losing queue priority when requoting (95% CI, paired draws)",
                 fontsize=11)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "fill_models.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
