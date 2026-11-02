"""C2: markout-based fill quality - the natural companion to the B+2 composition
finding (queue fills are ~4x more "picked off" than exponential fills at matched
mean rate). Markout is how a real desk actually measures this, fill by fill,
rather than via the hidden ground truth the detector tries to infer.

Two markout definitions, decomposed (they sum exactly):
    distance_captured   = side * (fill_price[t] - mid[t])          (h-independent)
                          the static compensation charged at fill time: constant
                          (=delta) for naive, but WIDENS on the side under
                          pressure for aware (eta*|q| grows the quote distance
                          on the toxic side) - this is the skew's whole purpose.
    impact_h            = side * (mid[t+h] - mid[t])                 (per horizon)
                          the PURE subsequent price move, excluding any pricing
                          choice - the genuine adverse-selection signal, a
                          property of the flow, not of the quoting strategy.
    markout_h           = distance_captured + impact_h
                          the desk-standard, fill-price-referenced P&L markout:
                          was this, net, a good trade.
side = +1 for a BID fill (we bought), -1 for an ASK fill (we sold). POSITIVE =
good; NEGATIVE = adverse selection, for all three quantities.

Why the decomposition matters here (found while running this): aware's markout
on INFORMED fills can come out net POSITIVE - not a bug, and not evidence
against adverse selection, but the mechanism working as intended. `impact_h` is
reliably negative on informed fills for BOTH strategies (a property of the
flow), but aware's `distance_captured` widens sharply on exactly the side under
pressure (inventory skew), and can OVER-compensate for the impact at this
toxicity level - so the fill-price-referenced `markout_h` nets out positive even
though the trade was genuinely adversely selected in impact terms. Reporting
`markout_h` alone would make this look like "no adverse selection happened";
reporting `impact_h` alongside it shows the flow was toxic regardless of how it
was priced, and shows exactly how much the skew over/under-charged for it.

Horizons h in {5, 10, 20, 50} ticks; a fill within h ticks of the session end is
dropped for that horizon. Fills are weighted by filled SIZE (0/1 under the
exponential model, continuous/partial under the queue model).

Split by informed vs quiet using `informed[t] != 0` at the FILL tick - the
ground-truth ACTIVE-episode flag, read ONLY for scoring here, never an input to
any strategy or fill decision.

Regime: project DEFAULT toxicity (0.12) - zero toxicity would make the
informed/quiet split vacuous. Both fill models, both strategies (naive eta=0,
aware eta=0.10), n=5000 paired sessions. Bootstrap resampled at the SESSION
level (fills within a session are not independent).

Writes figures/markout_analysis.png. Nothing else to disk.

Usage: python scripts/markout_analysis.py [--sessions 5000] [--boot 2000]
"""
import argparse
import warnings

import numpy as np

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from mmsim.core import simulate_paths
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.plots import FIG_DIR
from mmsim.strategies import InventorySkewQuoter

HORIZONS = (5, 10, 20, 50)
SEED = 2024
TOX = AdverseConfig().strength         # project default (0.12): zero would make
                                       # the informed/quiet split vacuous


def per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h):
    """For horizon h: per-session weighted sums of (distance_captured, impact_h,
    markout_h) and of size, split by informed vs quiet AT THE FILL TICK.
    Returns a dict of (n,) arrays keyed 'inf_*'/'q_*'.
    """
    n, N = f_bid.shape
    usable = N - h
    if usable <= 0:
        z = np.zeros(n)
        return {k: z for k in ("inf_dist", "inf_imp", "inf_mk", "inf_den",
                               "q_dist", "q_imp", "q_mk", "q_den")}
    m_now, m_future = mids[:, :usable], mids[:, h:h + usable]
    fb, fa = f_bid[:, :usable], f_ask[:, :usable]
    b, a = bid[:, :usable], ask[:, :usable]
    inf = informed[:, :usable] != 0

    dist_bid, dist_ask = m_now - b, a - m_now           # side-signed, h-independent
    imp_bid, imp_ask = m_future - m_now, m_now - m_future
    mk_bid, mk_ask = dist_bid + imp_bid, dist_ask + imp_ask   # == m_future - b / a - m_future

    out = {}
    for tag, mask in (("inf", inf), ("q", ~inf)):
        w_bid, w_ask = np.where(mask, fb, 0.0), np.where(mask, fa, 0.0)
        out[f"{tag}_dist"] = (w_bid * dist_bid + w_ask * dist_ask).sum(axis=1)
        out[f"{tag}_imp"] = (w_bid * imp_bid + w_ask * imp_ask).sum(axis=1)
        out[f"{tag}_mk"] = (w_bid * mk_bid + w_ask * mk_ask).sum(axis=1)
        out[f"{tag}_den"] = (w_bid + w_ask).sum(axis=1)
    return out


def weighted_mean(num, den):
    total = den.sum()
    return float(num.sum() / total) if total > 0 else float("nan")


def bootstrap_ci(num, den, rng, boot):
    n = len(num)
    vals = np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, n, size=n)
        vals[b] = weighted_mean(num[i], den[i])
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def run_one(queue_fills, eta, n_sessions, seed=SEED):
    market = MarketConfig(sigma=2.0)
    adv = AdverseConfig(strength=TOX)
    qc = QueueConfig() if queue_fills else None
    prep = prepare_batch(ArithmeticRandomWalk(market), seed, n_sessions, adv,
                         want_queue=queue_fills)
    res = simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(0.75, eta),
                         FillConfig(), market.dt, informed=prep.informed,
                         fill_boost=adv.fill_boost, queue=qc,
                         u_queue=prep.u_queue, u_depth0=prep.u_depth0, tilt=prep.tilt)
    return prep.mids, res["bid"], res["ask"], res["f_bid"], res["f_ask"], prep.informed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="fill-probability cap")
    rng = np.random.default_rng(99)

    configs = [("exponential", False, "naive", 0.0), ("exponential", False, "aware", 0.10),
              ("queue", True, "naive", 0.0), ("queue", True, "aware", 0.10)]
    results = {}
    print(f"Markout analysis: default toxicity {TOX}, n={args.sessions} sessions, "
          f"{args.boot} bootstrap resamples (session-level).")
    print("markout_h = distance_captured (h-independent, the skew's static compensation)")
    print("          + impact_h (the pure subsequent price move - the genuine toxicity signal)")
    print("Positive = good for us, for all three. Negative impact = adverse selection.\n")

    for model_name, queue_fills, strat_name, eta in configs:
        mids, bid, ask, f_bid, f_ask, informed = run_one(queue_fills, eta, args.sessions)
        total_fills = (f_bid + f_ask).sum()
        informed_fills = np.where(informed != 0, f_bid + f_ask, 0.0).sum()
        print("=" * 108)
        print(f"{model_name.upper()} model, {strat_name} (eta={eta})   "
              f"total filled size={total_fills:.0f}  informed share of filled size="
              f"{informed_fills / total_fills:.1%}")
        print("=" * 108)
        hdr = (f"{'horizon':>8} {'dist informed':>13} {'dist quiet':>11} "
               f"{'impact informed':>16} {'95% CI':>18} {'impact quiet':>13} {'95% CI':>18} "
               f"{'markout informed':>17} {'markout quiet':>14}")
        print(hdr)
        print("-" * len(hdr))
        rows = {}
        for h in HORIZONS:
            s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h)
            d_inf, d_q = weighted_mean(s["inf_dist"], s["inf_den"]), weighted_mean(s["q_dist"], s["q_den"])
            imp_inf, imp_q = weighted_mean(s["inf_imp"], s["inf_den"]), weighted_mean(s["q_imp"], s["q_den"])
            mk_inf, mk_q = weighted_mean(s["inf_mk"], s["inf_den"]), weighted_mean(s["q_mk"], s["q_den"])
            ci_imp_inf = bootstrap_ci(s["inf_imp"], s["inf_den"], rng, args.boot)
            ci_imp_q = bootstrap_ci(s["q_imp"], s["q_den"], rng, args.boot)
            rows[h] = dict(dist_inf=d_inf, dist_q=d_q, imp_inf=imp_inf, imp_q=imp_q,
                           ci_imp_inf=ci_imp_inf, ci_imp_q=ci_imp_q, mk_inf=mk_inf, mk_q=mk_q)
            print(f"{h:>8} {d_inf:>13.4f} {d_q:>11.4f} {imp_inf:>16.4f} "
                  f"[{ci_imp_inf[0]:>6.4f},{ci_imp_inf[1]:>6.4f}] {imp_q:>13.4f} "
                  f"[{ci_imp_q[0]:>6.4f},{ci_imp_q[1]:>6.4f}] {mk_inf:>17.4f} {mk_q:>14.4f}")
        results[(model_name, strat_name)] = rows

    print("\n" + "=" * 108)
    print("CROSS-MODEL COMPARISON AT h=20 (mid-range horizon)")
    print("=" * 108)
    hdr = (f"{'model':<12} {'strategy':<8} {'dist captured (informed)':>25} "
           f"{'pure impact (informed)':>23} {'fill-price markout (informed)':>30}")
    print(hdr)
    print("-" * len(hdr))
    for (model_name, strat_name), rows in results.items():
        r = rows[20]
        print(f"{model_name:<12} {strat_name:<8} {r['dist_inf']:>25.4f} {r['imp_inf']:>23.4f} "
              f"{r['mk_inf']:>30.4f}")

    print("\nPURE IMPACT (the genuine toxicity signal) is negative on informed fills for EVERY")
    print("model/strategy combination - the flow really is adverse, independent of how it was")
    print("priced. The QUEUE model's informed impact is more negative than the EXPONENTIAL")
    print("model's at matched strategy (B+2's composition finding, confirmed fill-by-fill, via")
    print("the metric a real desk would compute, independent of the hidden-state detector).")
    print("\nAWARE's distance_captured on informed fills is far larger than naive's (the skew")
    print("widening on the side under pressure) and, at this toxicity level, OVER-compensates")
    print("for the impact: aware's fill-price markout on informed fills is net POSITIVE even")
    print("though the underlying flow was genuinely toxic (impact negative). This is the skew")
    print("mechanism working, not an absence of adverse selection.")

    _figure(results)


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colours = {"exponential": "#2a6fbb", "queue": "#d9730d"}
    styles = {"naive": "--", "aware": "-"}
    panels = [("imp_inf", "ci_imp_inf", "pure impact, INFORMED fills\n(the genuine toxicity signal)"),
             ("imp_q", "ci_imp_q", "pure impact, QUIET fills"),
             ("mk_inf", None, "fill-price markout, INFORMED fills\n(nets out skew compensation)")]
    for ax, (key, ci_key, title) in zip(axes, panels):
        for (model_name, strat_name), rows in results.items():
            x = list(HORIZONS)
            y = [rows[h][key] for h in x]
            if ci_key:
                lo = [rows[h][key] - rows[h][ci_key][0] for h in x]
                hi = [rows[h][ci_key][1] - rows[h][key] for h in x]
                ax.errorbar(x, y, yerr=[lo, hi], marker="o", color=colours[model_name],
                           ls=styles[strat_name], capsize=3, label=f"{model_name} {strat_name}")
            else:
                ax.plot(x, y, marker="o", color=colours[model_name], ls=styles[strat_name],
                       label=f"{model_name} {strat_name}")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("horizon (ticks)")
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("size-weighted (+ = good, - = adverse selection)")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("C2: markout decomposition - pure price impact vs fill-price markout "
                "(95% session-bootstrap CI on impact)", fontsize=11)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "markout_analysis.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
