"""Five-parameter sensitivity sweep (sigma, A, k, delta, eta), one at a time.

Completes the sensitivity analysis originally asked for. Unlike C1 and C4, this
is not testing a specific hypothesis - it is a systematic check of whether the
headline naive-vs-aware comparison holds up, and how, as each market/strategy
parameter moves away from its project default, holding the other four fixed.

Regime: exponential fill model (the project's original regime, predating the
queue model), DEFAULT toxicity (strength = AdverseConfig().strength, i.e. the
project's normal operating point, NOT the zero-toxicity regime C4 uses - this
sweep is about the headline comparison's robustness in the regime the project
actually runs in day to day). Aware = eta 0.10 unless eta itself is the swept
parameter, in which case naive is always eta = 0.0 and the "aware" leg is the
swept value.

For each parameter: paired P&L gap and Sharpe gap (aware - naive), with 95%
paired-bootstrap CIs, at every grid point, holding the other four at their
project defaults (sigma=2.0, A=140.0, k=1.5, delta=0.75, eta=0.10).

Writes figures/sensitivity_sweep.png. Nothing else to disk.

Usage: python scripts/sweep_sensitivity.py [--sessions 5000] [--boot 1000]
"""
import argparse

import numpy as np

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig
from mmsim.core import simulate_paths
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.plots import FIG_DIR
from mmsim.strategies import InventorySkewQuoter

SEED = 2024
DEFAULTS = dict(sigma=2.0, A=140.0, k=1.5, delta=0.75, eta=0.10)
TOX = AdverseConfig().strength       # project default toxicity (0.12), not zero

GRIDS = {
    "sigma": (0.5, 1.0, 2.0, 4.0, 6.0, 8.0),
    "A": (40.0, 80.0, 140.0, 200.0, 280.0),
    "k": (0.5, 1.0, 1.5, 2.5, 4.0),
    "delta": (0.3, 0.5, 0.75, 1.0, 1.5),
    "eta": (0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
}


def sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def point(param, value, sessions, boot, rng):
    p = dict(DEFAULTS)
    p[param] = value
    market = MarketConfig(sigma=p["sigma"])
    adv = AdverseConfig(strength=TOX)
    prep = prepare_batch(ArithmeticRandomWalk(market), SEED, sessions, adv,
                         want_queue=False)   # exponential model only
    fills = FillConfig(A=p["A"], k=p["k"])

    def run(eta):
        return simulate_paths(prep.mids, prep.u_bid, prep.u_ask,
                              InventorySkewQuoter(p["delta"], eta), fills, market.dt,
                              informed=prep.informed, fill_boost=adv.fill_boost)
    aware_eta = value if param == "eta" else DEFAULTS["eta"]
    a = run(aware_eta)["terminal_pnl"]
    nv = run(0.0)["terminal_pnl"]
    d = a - nv
    gaps, diffs = np.empty(boot), np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, sessions, size=sessions)
        gaps[b] = sharpe(a[i]) - sharpe(nv[i])
        diffs[b] = (a[i] - nv[i]).mean()
    return dict(value=value, diff=d.mean(), diff_se=d.std(ddof=1) / np.sqrt(sessions),
               diff_lo=np.percentile(diffs, 2.5), diff_hi=np.percentile(diffs, 97.5),
               gap=sharpe(a) - sharpe(nv), gap_lo=np.percentile(gaps, 2.5),
               gap_hi=np.percentile(gaps, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(99)
    results = {}
    print(f"Sensitivity sweep: exponential model, default toxicity {TOX}, n={args.sessions} "
          f"paired sessions per point, {args.boot} bootstrap resamples.")
    print(f"Project defaults held fixed except the swept parameter: {DEFAULTS}\n")
    for param, grid in GRIDS.items():
        rows = [point(param, v, args.sessions, args.boot, rng) for v in grid]
        results[param] = rows
        print("=" * 100)
        print(f"PARAMETER: {param}   (others held at default; default value "
              f"{DEFAULTS.get(param, 'n/a (eta sweep: naive is always eta=0)')})")
        print("=" * 100)
        hdr = (f"{'value':>8} {'P&L gap':>9} {'se':>6} {'P&L 95% CI':>18} {'Sharpe gap':>11} "
               f"{'Sharpe 95% CI':>18}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['value']:>8.3g} {r['diff']:>9.2f} {r['diff_se']:>6.2f} "
                  f"[{r['diff_lo']:>7.2f},{r['diff_hi']:>7.2f}] {r['gap']:>11.3f} "
                  f"[{r['gap_lo']:>7.3f},{r['gap_hi']:>7.3f}]")
        diffs = [r["diff"] for r in rows]
        gaps = [r["gap"] for r in rows]
        sign_changes_pnl = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
        sign_changes_sh = sum(1 for i in range(len(gaps) - 1) if gaps[i] * gaps[i + 1] < 0)
        print(f"P&L gap sign changes across the grid: {sign_changes_pnl}   "
              f"Sharpe gap sign changes: {sign_changes_sh}")
        if sign_changes_pnl or sign_changes_sh:
            print("  -> the headline comparison's DIRECTION is not stable across this "
                  "parameter's range; see which grid point(s) flip above.")
    _figure(results)
    return results


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for j, (param, rows) in enumerate(results.items()):
        x = [r["value"] for r in rows]
        for row_i, (key, lo_key, hi_key, ylabel) in enumerate(
                (("diff", "diff_lo", "diff_hi", "paired P&L gap"),
                 ("gap", "gap_lo", "gap_hi", "Sharpe gap"))):
            ax = axes[row_i, j]
            y = [r[key] for r in rows]
            lo = [r[key] - r[lo_key] for r in rows]
            hi = [r[hi_key] - r[key] for r in rows]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", color="#2a6fbb", capsize=2)
            ax.axhline(0, color="#999999", lw=0.8)
            if param in DEFAULTS:
                ax.axvline(DEFAULTS[param], color="#d9730d", ls=":", lw=1)
            ax.set_title(param, fontsize=10)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.2)
    fig.suptitle("Sensitivity sweep: paired P&L gap and Sharpe gap (aware - naive) vs each "
                 "parameter, others at default (orange dotted line)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "sensitivity_sweep.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
