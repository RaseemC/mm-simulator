"""C1: sweep adverse selection and measure the inventory-aware strategy's advantage.

Hypothesis under test: the aware strategy's advantage GROWS with toxicity.
Registered prediction (docs/DESIGN_NOTES.md): the Sharpe gap rises more steeply
under the queue fill model than under the exponential one, because the queue model
makes informed flow structurally more damaging (bigger, deeper, one-sided market
orders rather than a raised fill probability).

Design
------
* Toxicity is `AdverseConfig.strength`: the drift per informed tick in units of
  sigma*sqrt(dt). Swept on a fixed grid from 0 (no informed flow) to 0.5.
* At every point both strategies run on IDENTICAL draws - same mids, same book,
  same informed episodes (common random numbers) - so the comparison is paired
  and the difference has far less variance than either strategy alone.
* Sharpe here is the per-session ratio mean(terminal P&L) / sd(terminal P&L),
  sample sd, no annualisation. The Sharpe GAP is aware minus naive.
* CIs come from a PAIRED bootstrap over sessions: resample session indices once
  and apply the same indices to both strategies, so the pairing is preserved.
  Percentile interval, 2000 resamples.
* Both fill models are run on the same sessions, so their slopes are comparable.

Everything is seeded and deterministic. Writes only figures/toxicity_sweep.png.

Usage: python scripts/sweep_toxicity.py [--sessions 10000] [--boot 2000]
"""
import argparse
import time

import numpy as np

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from mmsim.core import simulate_paths
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.plots import FIG_DIR
from mmsim.strategies import InventorySkewQuoter

GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
SEED, ETA, DELTA = 2024, 0.1, 0.75


def sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def point(strength, sessions, queue, boot, rng):
    """Run both strategies at one toxicity level; return paired statistics."""
    market = MarketConfig(sigma=2.0)
    adv = AdverseConfig(strength=strength)
    prep = prepare_batch(ArithmeticRandomWalk(market), SEED, sessions, adv)

    def run(eta):
        return simulate_paths(
            prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(DELTA, eta),
            FillConfig(), market.dt, informed=prep.informed, fill_boost=adv.fill_boost,
            queue=queue, u_queue=prep.u_queue, u_depth0=prep.u_depth0, tilt=prep.tilt)

    ra, rn = run(ETA), run(0.0)
    a, nv = ra["terminal_pnl"], rn["terminal_pnl"]
    d = a - nv
    # paired bootstrap: one index draw applied to BOTH strategies (drawn per
    # iteration rather than all at once, which would be a 160MB index array)
    gaps = np.empty(boot)
    diffs = np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, sessions, size=sessions)
        gaps[b] = sharpe(a[i]) - sharpe(nv[i])
        diffs[b] = (a[i] - nv[i]).mean()
    return dict(
        strength=strength,
        aware=a.mean(), naive=nv.mean(),
        diff=d.mean(), diff_se=d.std(ddof=1) / np.sqrt(sessions),
        sh_a=sharpe(a), sh_n=sharpe(nv), gap=sharpe(a) - sharpe(nv),
        gap_lo=np.percentile(gaps, 2.5), gap_hi=np.percentile(gaps, 97.5),
        diff_lo=np.percentile(diffs, 2.5), diff_hi=np.percentile(diffs, 97.5),
        q_aware=float(np.abs(ra["q"]).mean()), q_naive=float(np.abs(rn["q"]).mean()),
    )


def slope(xs, ys):
    """OLS slope of y on x."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10_000)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    results = {}
    for label, queue in (("queue", QueueConfig()), ("exponential", None)):
        rng = np.random.default_rng(99)      # same bootstrap draws for both models
        t0 = time.time()
        rows = [point(s, args.sessions, queue, args.boot, rng) for s in GRID]
        results[label] = rows
        print("=" * 108)
        print(f"{label.upper()} FILL MODEL   n = {args.sessions} paired sessions per point, "
              f"eta = {ETA}, {args.boot} bootstrap resamples   [{time.time() - t0:.0f}s]")
        print("=" * 108)
        hdr = (f"{'toxicity':>9} {'aware':>8} {'naive':>8} {'diff':>8} {'se':>6} "
               f"{'diff 95% CI':>18} {'Sh aware':>9} {'Sh naive':>9} {'Sh gap':>8} "
               f"{'gap 95% CI':>18} {'mean|q| a/n':>13}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['strength']:>9.2f} {r['aware']:>8.2f} {r['naive']:>8.2f} "
                  f"{r['diff']:>8.2f} {r['diff_se']:>6.2f} "
                  f"[{r['diff_lo']:>7.2f},{r['diff_hi']:>7.2f}] "
                  f"{r['sh_a']:>9.3f} {r['sh_n']:>9.3f} {r['gap']:>8.3f} "
                  f"[{r['gap_lo']:>7.3f},{r['gap_hi']:>7.3f}] "
                  f"{r['q_aware']:>5.1f}/{r['q_naive']:<6.1f}")
        g = slope(GRID, [r["gap"] for r in rows])
        dd = slope(GRID, [r["diff"] for r in rows])
        print(f"OLS slope of Sharpe gap on toxicity: {g:+.3f} per unit strength   "
              f"(P&L diff slope {dd:+.2f})")
    _compare(results)
    _figure(results)


def _compare(results):
    print("\n" + "=" * 108)
    print("PREDICTION SCORED: is the Sharpe gap steeper under the queue model?")
    print("=" * 108)
    for label in ("queue", "exponential"):
        rows = results[label]
        print(f"{label:<12} Sharpe gap {rows[0]['gap']:+.3f} at toxicity 0 -> "
              f"{rows[-1]['gap']:+.3f} at 0.5   (change {rows[-1]['gap'] - rows[0]['gap']:+.3f}, "
              f"OLS slope {slope(GRID, [r['gap'] for r in rows]):+.3f})")
    qs = slope(GRID, [r["gap"] for r in results["queue"]])
    es = slope(GRID, [r["gap"] for r in results["exponential"]])
    verdict = "CONFIRMED" if qs > es else "NOT CONFIRMED"
    print(f"\nqueue slope {qs:+.3f} vs exponential slope {es:+.3f}  ->  prediction {verdict}")
    # A single OLS slope is a poor summary of a non-monotonic curve, so also report
    # the rising region and where (if anywhere) each model's gap turns over.
    k = GRID.index(0.20) + 1
    lo_q = slope(GRID[:k], [r["gap"] for r in results["queue"][:k]])
    lo_e = slope(GRID[:k], [r["gap"] for r in results["exponential"][:k]])
    print(f"restricted to toxicity <= 0.20: queue {lo_q:+.3f} vs exponential {lo_e:+.3f}  ->  "
          f"{'CONFIRMED' if lo_q > lo_e else 'NOT CONFIRMED'} on the rising region")
    for label in ("queue", "exponential"):
        gaps = [r["gap"] for r in results[label]]
        peak = int(np.argmax(gaps))
        print(f"{label:<12} peak Sharpe gap {gaps[peak]:+.3f} at toxicity {GRID[peak]:.2f}"
              + ("  (turns over before the end of the grid)" if peak < len(GRID) - 1 else
                 "  (still rising at the end of the grid)"))


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"queue": "#2a6fbb", "exponential": "#d9730d"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, rows in results.items():
        x = [r["strength"] for r in rows]
        c = colours[label]
        gap = [r["gap"] for r in rows]
        lo = [r["gap"] - r["gap_lo"] for r in rows]
        hi = [r["gap_hi"] - r["gap"] for r in rows]
        ax[0].errorbar(x, gap, yerr=[lo, hi], marker="o", color=c, label=label, capsize=3)
        d = [r["diff"] for r in rows]
        dlo = [r["diff"] - r["diff_lo"] for r in rows]
        dhi = [r["diff_hi"] - r["diff"] for r in rows]
        ax[1].errorbar(x, d, yerr=[dlo, dhi], marker="s", color=c, label=label, capsize=3)
        ax[2].plot(x, [r["sh_a"] for r in rows], marker="o", color=c, label=f"{label} aware")
        ax[2].plot(x, [r["sh_n"] for r in rows], marker="o", ls="--", color=c, alpha=0.55,
                   label=f"{label} naive")
    ax[0].set_ylabel("Sharpe gap (aware - naive)")
    ax[0].set_title("Sharpe gap vs toxicity", fontsize=10)
    ax[1].set_ylabel("paired P&L difference")
    ax[1].set_title("P&L advantage vs toxicity", fontsize=10)
    ax[2].set_ylabel("Sharpe per session")
    ax[2].set_title("Level, by strategy", fontsize=10)
    for a in ax:
        a.axhline(0, color="#999999", lw=0.8)
        a.set_xlabel("toxicity (informed drift, units of $\\sigma\\sqrt{dt}$)")
        a.legend(fontsize=7, frameon=False)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.2)
    fig.suptitle("C1: does inventory-aware quoting gain more as adverse selection rises? "
                 "(paired sessions, 95% bootstrap CI)", fontsize=11)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "toxicity_sweep.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
