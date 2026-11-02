"""C1 (2-D): separate the two things 'toxicity' was conflating.

The 1-D sweep (sweep_toxicity.py) found that `AdverseConfig.strength` (the
informed price DRIFT) and `QueueConfig.informed_mo_mult` (the one-sided market-
order FLOW boost) are different knobs that the queue model's single toxicity
parameter accidentally moved together: `strength = 0` disables episodes
entirely, which also switches the flow boost off, so the "toxicity = 0" point
is a different regime, not a small-drift limit of the others.

This script sweeps them independently:
    rows    informed_mo_mult in {1.0 (off), 1.25, 1.5, 1.75, 2.0 (saturated+)}
    columns strength        in {0.05, 0.10, 0.20, 0.30, 0.50}  (strength = 0 is
                             reported separately: with no episodes, mo_mult
                             cannot have an effect, so it collapses to one point)

IMPORTANT: the informed arrival probability is min(1, mo_rate * informed_mo_mult).
At the default mo_rate = 0.5, this SATURATES at mo_mult = 2.0 - the C1 default of
2.5 was already past the cap. The first version of this grid used {1.0, 1.5, 2.0,
2.5, 3.5} and produced three IDENTICAL rows (2.0, 2.5, 3.5), which looked like a
bug before the cap was found. The grid below is chosen to lie entirely in the
UNSATURATED region (1.0 to 2.0) so each row is a genuinely different flow strength;
2.0 is kept as the saturation point, reported explicitly rather than past it.
for BOTH the Sharpe gap and the paired P&L gap, each its own panel/table, since
C1 found they rank the models differently and that difference is itself a
result. Same n, seed and paired bootstrap as sweep_toxicity.py.

The exponential model has no separate flow/drift split (`fill_boost` plays both
roles at once), so it is reported only as the 1-D marginal over strength, for
comparison with the queue model's informed_mo_mult = 2.5 row (its own default).

Writes figures/toxicity_sweep_2d.png. Nothing else to disk.

Usage: python scripts/sweep_toxicity_2d.py [--sessions 10000] [--boot 2000]
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

STRENGTHS = (0.05, 0.10, 0.20, 0.30, 0.50)
MULTS = (1.0, 1.25, 1.5, 1.75, 2.0)
SEED, ETA, DELTA = 2024, 0.1, 0.75


def sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def point(strength, mo_mult, sessions, boot, rng):
    market = MarketConfig(sigma=2.0)
    adv = AdverseConfig(strength=strength)
    qc = QueueConfig(informed_mo_mult=mo_mult)
    prep = prepare_batch(ArithmeticRandomWalk(market), SEED, sessions, adv)

    def run(eta):
        return simulate_paths(
            prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(DELTA, eta),
            FillConfig(), market.dt, informed=prep.informed, fill_boost=adv.fill_boost,
            queue=qc, u_queue=prep.u_queue, u_depth0=prep.u_depth0, tilt=prep.tilt)

    a, nv = run(ETA)["terminal_pnl"], run(0.0)["terminal_pnl"]
    d = a - nv
    gaps, diffs = np.empty(boot), np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, sessions, size=sessions)
        gaps[b] = sharpe(a[i]) - sharpe(nv[i])
        diffs[b] = (a[i] - nv[i]).mean()
    return dict(
        strength=strength, mo_mult=mo_mult,
        diff=d.mean(), diff_se=d.std(ddof=1) / np.sqrt(sessions),
        diff_lo=np.percentile(diffs, 2.5), diff_hi=np.percentile(diffs, 97.5),
        gap=sharpe(a) - sharpe(nv),
        gap_lo=np.percentile(gaps, 2.5), gap_hi=np.percentile(gaps, 97.5),
    )


def zero_point(sessions, boot, rng):
    """strength = 0: no episodes at all, so informed_mo_mult has no effect.
    Reported once, not as a row of the grid (see module docstring)."""
    return point(0.0, 2.5, sessions, boot, rng)   # mo_mult value is irrelevant here


def exponential_marginal(sessions, boot, rng):
    market = MarketConfig(sigma=2.0)
    rows = []
    for strength in (0.0,) + STRENGTHS:
        adv = AdverseConfig(strength=strength)
        prep = prepare_batch(ArithmeticRandomWalk(market), SEED, sessions, adv)

        def run(eta):
            return simulate_paths(prep.mids, prep.u_bid, prep.u_ask,
                                  InventorySkewQuoter(DELTA, eta), FillConfig(), market.dt,
                                  informed=prep.informed, fill_boost=adv.fill_boost)
        a, nv = run(ETA)["terminal_pnl"], run(0.0)["terminal_pnl"]
        d = a - nv
        gaps, diffs = np.empty(boot), np.empty(boot)
        for b in range(boot):
            i = rng.integers(0, sessions, size=sessions)
            gaps[b] = sharpe(a[i]) - sharpe(nv[i])
            diffs[b] = (a[i] - nv[i]).mean()
        rows.append(dict(strength=strength, diff=d.mean(), diff_se=d.std(ddof=1) / np.sqrt(sessions),
                         diff_lo=np.percentile(diffs, 2.5), diff_hi=np.percentile(diffs, 97.5),
                         gap=sharpe(a) - sharpe(nv), gap_lo=np.percentile(gaps, 2.5),
                         gap_hi=np.percentile(gaps, 97.5)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10_000)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    n, boot = args.sessions, args.boot

    t0 = time.time()
    rng = np.random.default_rng(99)
    z = zero_point(n, boot, rng)
    print("=" * 100)
    print(f"QUEUE MODEL, strength = 0 (NO EPISODES AT ALL: informed_mo_mult has no effect "
          f"here, so this is reported once, not as a grid row)")
    print("=" * 100)
    print(f"paired P&L diff {z['diff']:+.2f} [{z['diff_lo']:+.2f}, {z['diff_hi']:+.2f}]   "
          f"Sharpe gap {z['gap']:+.3f} [{z['gap_lo']:+.3f}, {z['gap_hi']:+.3f}]")
    print("(naive beats aware here on both fill models -- see the zero-toxicity validation "
          "result below)")

    grid = {}
    for mo_mult in MULTS:
        for strength in STRENGTHS:
            grid[(mo_mult, strength)] = point(strength, mo_mult, n, boot, rng)
    print(f"\n2-D grid computed in {time.time() - t0:.0f}s "
          f"({len(MULTS)} x {len(STRENGTHS)} points, n={n}, boot={boot})")

    def table(key, fmt, width):
        print("\n" + "=" * 100)
        print(f"{'PAIRED P&L GAP' if key == 'diff' else 'SHARPE GAP'}  "
              f"(rows = informed_mo_mult [flow], columns = strength [drift])")
        print("=" * 100)
        hdr = f"{'mo_mult':>9}" + "".join(f"{s:>{width}.2f}" for s in STRENGTHS)
        print(hdr)
        print("-" * len(hdr))
        for mo_mult in MULTS:
            row = f"{mo_mult:>9.1f}"
            for strength in STRENGTHS:
                row += fmt.format(grid[(mo_mult, strength)][key])
            tag = "  <- flow boost OFF (drift only)" if mo_mult == 1.0 else \
                  "  <- arrival probability SATURATES here (mo_rate=0.5 cap)" \
                  if mo_mult == 2.0 else ""
            print(row + tag)

    table("diff", "{:>" + "12.2f" + "}", 12)
    table("gap", "{:>" + "12.3f" + "}", 12)

    print("\n" + "=" * 100)
    print("WHAT THE GRID SHOWS")
    print("=" * 100)
    flow_off = [grid[(1.0, s)]["gap"] for s in STRENGTHS]
    flow_on = [grid[(2.0, s)]["gap"] for s in STRENGTHS]
    print(f"Sharpe gap with informed_mo_mult = 1.0 (drift only, flow boost OFF): "
          f"{min(flow_off):.3f} to {max(flow_off):.3f} across strength 0.05-0.50 "
          f"(range {max(flow_off) - min(flow_off):+.3f})")
    print(f"Sharpe gap with informed_mo_mult = 2.0 (arrival probability saturated):"
          f"{flow_on[0]:.3f} to {max(flow_on):.3f} across strength 0.05-0.50 "
          f"(range {max(flow_on) - min(flow_on):+.3f})")
    print("If the drift-only row is nearly flat while mo_mult clearly shifts the level "
          "between rows, that confirms the 1-D finding: informed_mo_mult (flow), not "
          "strength (drift), is what drives the queue model's toxicity effect.")
    print(f"\nNOTE: informed arrival probability = min(1, mo_rate * informed_mo_mult); "
          f"at mo_rate = {QueueConfig().mo_rate}, this saturates at mo_mult = "
          f"{1.0 / QueueConfig().mo_rate:.2f}. The C1 default (2.5) is PAST the cap, so "
          f"raising it further has NO additional effect -- under this model, one-sided "
          f"flow toxicity cannot be made arbitrarily large through informed_mo_mult alone "
          f"once arrival is already near-certain; only larger fill SIZE could do more, "
          f"and mo_size_mean does not depend on the informed state. This is a model "
          f"limitation, not a tuning choice, and belongs in B+3.")

    print("\n" + "=" * 100)
    print("EXPONENTIAL MODEL MARGINAL (1-D over strength, for comparison; no flow/drift split)")
    print("=" * 100)
    exp_rows = exponential_marginal(n, boot, np.random.default_rng(99))
    hdr = f"{'strength':>9} {'P&L diff':>10} {'Sharpe gap':>11}"
    print(hdr)
    for r in exp_rows:
        print(f"{r['strength']:>9.2f} {r['diff']:>10.2f} {r['gap']:>11.3f}")

    _figure(z, grid, exp_rows)


def _figure(z, grid, exp_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    for i, mo_mult in enumerate(MULTS):
        c = cmap(i / (len(MULTS) - 1))
        ls = "--" if mo_mult == 1.0 else "-"
        lbl = f"mo_mult={mo_mult:.1f}" + (" (flow off)" if mo_mult == 1.0 else
                                          " (saturates)" if mo_mult == 2.0 else "")
        diffs = [grid[(mo_mult, s)]["diff"] for s in STRENGTHS]
        dlo = [grid[(mo_mult, s)]["diff"] - grid[(mo_mult, s)]["diff_lo"] for s in STRENGTHS]
        dhi = [grid[(mo_mult, s)]["diff_hi"] - grid[(mo_mult, s)]["diff"] for s in STRENGTHS]
        ax[0].errorbar(STRENGTHS, diffs, yerr=[dlo, dhi], marker="o", color=c, ls=ls,
                       label=lbl, capsize=2)
        gaps = [grid[(mo_mult, s)]["gap"] for s in STRENGTHS]
        glo = [grid[(mo_mult, s)]["gap"] - grid[(mo_mult, s)]["gap_lo"] for s in STRENGTHS]
        ghi = [grid[(mo_mult, s)]["gap_hi"] - grid[(mo_mult, s)]["gap"] for s in STRENGTHS]
        ax[1].errorbar(STRENGTHS, gaps, yerr=[glo, ghi], marker="o", color=c, ls=ls,
                       label=lbl, capsize=2)
    for a in ax:
        a.axhline(0, color="#999999", lw=0.8)
        a.axhline(z["gap"] if a is ax[1] else z["diff"], color="black", lw=0.8, ls=":")
        a.set_xlabel("strength (informed drift, units of $\\sigma\\sqrt{dt}$)")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.2)
    ax[0].set_ylabel("paired P&L gap (aware - naive)")
    ax[0].set_title("P&L gap: flow (colour) vs drift (x-axis)", fontsize=10)
    ax[1].set_ylabel("Sharpe gap (aware - naive)")
    ax[1].set_title("Sharpe gap: flow (colour) vs drift (x-axis)", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("C1 (2-D): separating the flow boost (informed_mo_mult) from the drift "
                 "(strength) -- dotted black line = strength 0 (no episodes)", fontsize=11)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "toxicity_sweep_2d.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
