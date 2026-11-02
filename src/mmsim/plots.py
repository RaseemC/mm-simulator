"""All charts: take data in, save a PNG under figures/ with a FIXED filename.

Files are overwritten on each run (never appended or timestamped). Colours:
two-series palette (naive = blue, aware = orange), distinguishable under
colour-vision deficiency and additionally separated by line style/markers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; live GUI view (M4) selects its own backend
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")
NAIVE_C, AWARE_C, MID_C = "#2a6fbb", "#d9730d", "#444444"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_single_session(mids, naive: dict, aware: dict, dt: float,
                        name: str = "single_session.png") -> Path:
    """One session, both strategies. mids (N+1,); naive/aware are simulate_paths
    outputs for a single session (row 0).

    Panel 1: mid, the inventory-aware maker's quotes, and its fills.
    Panel 2: inventory. Panel 3: mark-to-market P&L.
    """
    N = len(mids) - 1
    t = np.arange(N + 1) * dt
    ts = t[:-1]  # quotes/fills at step n happen at time t_n
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 1]))

    a = ax[0]
    a.plot(t, mids, color=MID_C, lw=1.5, label="mid")
    a.step(ts, aware["bid"][0], where="post", color=AWARE_C, lw=1, alpha=0.8, label="aware bid")
    a.step(ts, aware["ask"][0], where="post", color=AWARE_C, lw=1, ls="--", alpha=0.8, label="aware ask")
    bf, af = aware["f_bid"][0] == 1, aware["f_ask"][0] == 1
    a.scatter(ts[bf], aware["bid"][0][bf], marker="^", s=30, color=AWARE_C, edgecolor="white",
              linewidth=0.6, zorder=3, label="bid filled (buy)")
    a.scatter(ts[af], aware["ask"][0][af], marker="v", s=30, color=AWARE_C, edgecolor="white",
              linewidth=0.6, zorder=3, label="ask filled (sell)")
    a.set_ylabel("price")
    a.set_title("Single session: midprice, inventory-aware quotes and fills")
    a.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02))

    ax[1].step(t, naive["q"][0], where="post", color=NAIVE_C, lw=1.5, label="naive")
    ax[1].step(t, aware["q"][0], where="post", color=AWARE_C, lw=1.5, ls="--", label="aware")
    ax[1].axhline(0, color="#999999", lw=0.8)
    ax[1].set_ylabel("inventory q")

    ax[2].plot(t, naive["mtm"][0], color=NAIVE_C, lw=1.5, label="naive")
    ax[2].plot(t, aware["mtm"][0], color=AWARE_C, lw=1.5, ls="--", label="aware")
    ax[2].axhline(0, color="#999999", lw=0.8)
    ax[2].set_ylabel("MTM P&L")
    ax[2].set_xlabel("time t")
    for a_ in ax[1:]:
        a_.legend(fontsize=8, frameon=False, loc="upper left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=0.2)
    return _save(fig, name)
