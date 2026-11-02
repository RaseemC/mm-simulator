"""C8: statistical power. For each headline paired comparison reported elsewhere in
this project, compute the number of SESSIONS needed to detect the observed paired
effect at 95% confidence / 80% power, and confirm the n actually used exceeds it.

Standard paired-difference power formula:
    n_required = ((z_{a/2} + z_{b})^2 * sd_d^2) / mean_d^2
    z_{a/2} = 1.959964 (two-sided 95% confidence), z_b = 0.841621 (80% power)
derived from treating the paired difference d_i = aware_i - naive_i as
approximately normal (justified here by the same N-step CLT argument as C4
section 2 - each d_i is itself built from ~200 per-tick fill/price increments).

Each comparison is RECOMPUTED fresh here (not pulled from an earlier printed
number) at the SAME n used when that result was originally reported, so the
power conclusion is about the actual run being cited, with its own seed and
sample. Headline comparisons covered:
  1. C1 2D (queue model), default toxicity 0.12, eta 0.1 vs 0        (n=5000)
  2. C1 2D (exponential model), default toxicity 0.12, eta 0.1 vs 0  (n=5000)
  3. C1 2D (queue model), zero toxicity, eta 0.1 vs 0                (n=10000)
  4. C1 2D (queue model), strength=0.5, informed_mo_mult saturated   (n=10000)
  5. B+2(c) priority-cost differential (costly-free gap, aware - naive), eta 0.1 (n=1500)
  6. C4 Q1 (exponential, zero toxicity), eta 0.10 vs 0: mean P&L gap  (n=10000)

Writes nothing to disk; prints a table.

Usage: python scripts/power_analysis.py
"""
import numpy as np

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from mmsim.core import simulate_paths
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.strategies import InventorySkewQuoter

Z_ALPHA2, Z_BETA = 1.959964, 0.841621


def required_n(mean_d, sd_d):
    if mean_d == 0:
        return float("inf")
    return ((Z_ALPHA2 + Z_BETA) ** 2) * sd_d**2 / mean_d**2


def paired_run(n, strength, eta, queue_fills, delta=0.75, sigma=2.0, seed=2024):
    market = MarketConfig(sigma=sigma)
    adv = AdverseConfig(strength=strength)
    qc = QueueConfig() if queue_fills else None
    prep = prepare_batch(ArithmeticRandomWalk(market), seed, n, adv, want_queue=queue_fills)

    def run(e):
        return simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(delta, e),
                              FillConfig(), market.dt, informed=prep.informed,
                              fill_boost=adv.fill_boost, queue=qc, u_queue=prep.u_queue,
                              u_depth0=prep.u_depth0, tilt=prep.tilt)
    a, nv = run(eta)["terminal_pnl"], run(0.0)["terminal_pnl"]
    return a - nv


def priority_cost_differential(n, eta, seed=2024):
    """B+2(c): (costly - free) requote cost, for aware and for naive; return the
    DIFFERENCE of those two costs (the quantity the research question needs)."""
    market = MarketConfig(sigma=2.0)
    adv = AdverseConfig()
    qc_costly = QueueConfig()
    qc_free = QueueConfig(keep_priority=True)
    prep = prepare_batch(ArithmeticRandomWalk(market), seed, n, adv, want_queue=True)

    def run(e, qc):
        return simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(0.75, e),
                              FillConfig(), market.dt, informed=prep.informed,
                              fill_boost=adv.fill_boost, queue=qc, u_queue=prep.u_queue,
                              u_depth0=prep.u_depth0, tilt=prep.tilt)["terminal_pnl"]
    aware_cost = run(eta, qc_costly) - run(eta, qc_free)
    naive_cost = run(0.0, qc_costly) - run(0.0, qc_free)
    return aware_cost - naive_cost


def report(label, d, n_actual):
    mean_d, sd_d = d.mean(), d.std(ddof=1)
    se = sd_d / np.sqrt(n_actual)
    n_req = required_n(mean_d, sd_d)
    margin = n_actual / n_req if np.isfinite(n_req) and n_req > 0 else float("inf")
    print(f"{label}")
    print(f"  mean diff = {mean_d:+.3f}  (se {se:.3f} at n={n_actual})   sd(diff) = {sd_d:.2f}")
    print(f"  n required for 80% power at 95% confidence: {n_req:,.0f}")
    status = "SUFFICIENT" if n_actual >= n_req else "INSUFFICIENT"
    print(f"  n actually used: {n_actual:,}  ->  {status}  (margin {margin:.1f}x)\n")
    return dict(label=label, mean=mean_d, sd=sd_d, n_req=n_req, n_actual=n_actual, margin=margin)


def main():
    import warnings
    warnings.filterwarnings("ignore", message="fill-probability cap")
    rows = []
    rows.append(report(
        "1. C1 2D, QUEUE model, default toxicity (0.12), eta=0.1 vs 0",
        paired_run(5000, 0.12, 0.1, True), 5000))
    rows.append(report(
        "2. C1 2D, EXPONENTIAL model, default toxicity (0.12), eta=0.1 vs 0",
        paired_run(5000, 0.12, 0.1, False), 5000))
    rows.append(report(
        "3. C1 2D, QUEUE model, ZERO toxicity, eta=0.1 vs 0",
        paired_run(10000, 0.0, 0.1, True), 10000))
    rows.append(report(
        "4. C1 2D, QUEUE model, strength=0.5 (informed_mo_mult saturated), eta=0.1 vs 0",
        paired_run(10000, 0.5, 0.1, True), 10000))
    rows.append(report(
        "5. B+2(c) priority-cost DIFFERENTIAL (aware cost - naive cost), eta=0.1",
        priority_cost_differential(1500, 0.1), 1500))
    rows.append(report(
        "6. C4 Q1, EXPONENTIAL model, ZERO toxicity, eta=0.10 vs 0 (mean P&L gap)",
        paired_run(10000, 0.0, 0.1, False), 10000))

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = f"{'comparison':<70} {'n_req':>10} {'n_used':>8} {'margin':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:70]:<70} {r['n_req']:>10,.0f} {r['n_actual']:>8,} {r['margin']:>7.1f}x")
    under = [r for r in rows if r["margin"] < 1.0]
    if under:
        print(f"\n{len(under)} comparison(s) UNDERPOWERED at their originally-reported n:")
        for r in under:
            print(f"  - {r['label']}")
    else:
        print("\nEvery headline comparison checked here used more sessions than its 80%-power "
              "requirement.")


if __name__ == "__main__":
    main()
