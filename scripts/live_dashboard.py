"""Live market-making dashboard (synthetic data): paired strategies, order book, epochs.

Usage (from the project root, with the project venv):
    python scripts/live_dashboard.py [--seed 42] [--exponential]

The simulation lives in mmsim.live_sim; this file is only the GUI.

Fill model: the QUEUE model (a limit order book with queue position - the model the
headline findings come from) is ON by default; untick "Queue fills" or pass
--exponential for the textbook exponential intensity. Switching opens a new epoch.

Four tabs, one shared animation loop. Tabs change what is DRAWN, never what is
COMPUTED: every tick of both strategies is simulated whichever tab is showing, and
the cheap line data of every panel is refreshed every frame; only the expensive
per-frame rebuilds (book ladder, depth heatmap, bubble chart, shading) are limited
to the visible tab.
  Market      midprice with BOTH strategies' quotes and fills; detector panel.
  Book        order book ladder with our queue position; depth-over-time heatmap;
              our queue position over time; order-cluster bubble chart.
  Strategies  inventory, MTM P&L, spread P&L, inventory P&L - naive vs aware, each
              panel on its own y-scale.
  Stats       current epoch (per-strategy stats, paired difference, n), run-wide
              extremes, closed epochs, paired-difference chart.

Sliders and buttons are visible on every tab. Sliders QUEUE a change; it applies at
a tick boundary and opens a new epoch (statistics are per epoch, never pooled
across configs). The red tick on each slider marks the value the CURRENT epoch
started with; the grey "def" beside the value is the project default.
Any statistic from fewer than 30 completed sessions is greyed and asterisked.
Freeze pauses and prints the current epoch to the TERMINAL as a self-checking
reproduction script. Space or p pauses/resumes. Every click is echoed on the status
line, so a dead handler is visible rather than silent.

Truth (the hidden informed state) is shown only where labelled as such: the shaded
bands on the Market tab, and the bubble chart's DEBUG colouring. Neither ever feeds a
strategy or a statistic. Writes nothing to disk; the epoch table prints on Reset and
on exit.
"""
import argparse
import itertools
import time
import traceback
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider

from mmsim import live_sim as ls
from mmsim.config import QueueConfig
from mmsim.epochs import config_str
from mmsim.repro import reproduction_report

STEPS_PER_FRAME = 2
NAIVE_C, AWARE_C, MID_C = "#2a6fbb", "#d9730d", "#222222"
BID_C, ASK_C = "#2e8b57", "#c0392b"
TRUTH_C = "#b8a2d6"
INK, MUTED = "#222222", "#9a9a9a"
WARN = "#b3261e"                 # under-powered banner only
MIN_N = 30                       # fewer completed sessions than this: banner, read as noise
TABS = ("Market", "Book", "Strategies", "Stats")
LADDER_LEVELS = 8                # levels per side on the ladder
HEAT_LEVELS = 10                 # levels per side in the depth heatmap
BUBBLE_LEVELS = 6                # levels per side in the bubble chart
BUBBLE_EVERY = 4                 # one bubble column every N ticks

_QDEF = QueueConfig()
# key, symbol, plain-language meaning, lo, hi, project default, model it belongs to
SLIDER_SPECS = [
    ("eta", "eta", "inventory skew strength", 0.0, 0.5, ls.DEFAULT_PARAMS["eta"], "both"),
    ("delta", "delta", "half-spread", 0.05, 2.0, ls.DEFAULT_PARAMS["delta"], "both"),
    ("sigma", "sigma", "price volatility", 0.1, 8.0, ls.DEFAULT_PARAMS["sigma"], "both"),
    ("A", "A", "order arrival rate (exponential fills)", 10.0, 400.0,
     ls.DEFAULT_PARAMS["A"], "exp"),
    ("k", "k", "fill decay with distance (exponential fills)", 0.2, 4.0,
     ls.DEFAULT_PARAMS["k"], "exp"),
    ("tox", "toxicity", "informed-flow strength (price drift)", 0.0, 0.5,
     ls.DEFAULT_PARAMS["tox"], "both"),
    ("depth_base", "depth", "resting depth at the touch, lots", 1.0, 30.0,
     _QDEF.depth_base, "queue"),
    ("mo_rate", "MO rate", "market-order arrival prob. per tick", 0.05, 1.0,
     _QDEF.mo_rate, "queue"),
    ("mo_size_mean", "MO size", "mean market-order size, lots", 5.0, 120.0,
     _QDEF.mo_size_mean, "queue"),
    ("informed_mo_mult", "inf. flow", "informed flow boost (saturates at 1/MO rate)",
     1.0, 4.0, _QDEF.informed_mo_mult, "queue"),
]
_UNUSED = "  (unused by current fill model)"

# Stats tab, tier 3: the config as grouped key/value pairs rather than one wall of text.
# (key, label, format) - "flag" renders a bool, "d" an integer.
CONFIG_GROUPS = [
    ("Market", [("sigma", "volatility", ".2f"), ("tox", "toxicity", ".3f")]),
    ("Strategy", [("eta", "eta (skew)", ".3f"), ("delta", "delta (half-spread)", ".2f")]),
    ("Exponential fills", [("A", "A (arrival)", ".0f"), ("k", "k (decay)", ".2f")]),
    ("Queue model", [("depth_base", "depth at touch", ".1f"), ("mo_rate", "MO rate", ".2f"),
                     ("mo_size_mean", "MO size", ".1f"),
                     ("informed_mo_mult", "inf. flow", ".2f")]),
    ("Run", [("seed", "seed", "d"), ("session_steps", "session steps", "d"),
             ("queue_fills", "fill model", "flag"), ("auto_eta", "auto-eta", "flag"),
             ("auto_widen", "auto-widen", "flag")]),
]
# What a value is marked against. queue_fills defaults to the QUEUE model here because that
# is the dashboard's default and the model the headline findings came from - the batch
# engine's own default is the exponential one (see live_sim.LiveMarketMaker).
PROJECT_DEFAULTS = dict(ls.DEFAULT_PARAMS,
                        **{k: float(getattr(_QDEF, k)) for k in ls.QUEUE_KEYS},
                        seed=ls.SEED, session_steps=ls.SESSION_STEPS,
                        queue_fills=True, auto_eta=False, auto_widen=False)


def cfg_value(key, v, spec) -> str:
    if spec == "flag":
        return ("queue book" if v else "exponential") if key == "queue_fills" else \
            ("on" if v else "off")
    return f"{int(v)}" if spec == "d" else f"{v:{spec}}"


def config_columns(cfg: dict) -> list[str]:
    """The config as two columns of grouped key/value lines, changed values marked."""
    cols = []
    for groups in (CONFIG_GROUPS[:3], CONFIG_GROUPS[3:]):
        lines = []
        for title, items in groups:
            if lines:
                lines.append("")
            lines.append(title.upper())
            for key, label, spec in items:
                if key not in cfg:
                    continue
                v = cfg[key]
                mark = " •" if PROJECT_DEFAULTS.get(key, v) != v else ""
                lines.append(f"  {label:<18}{cfg_value(key, v, spec) + mark:>13}")
        cols.append("\n".join(lines))
    return cols


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def fmt(x, spec="+.2f") -> str:
    """A number, or an em dash when it is undefined (nan/inf) - never 'nan'."""
    return format(x, spec) if _finite(x) else "—"


def pct(x) -> str:
    return f"{x:.1%}" if _finite(x) else "—"


def _set_ylim(ax, *series):
    """Independent, padded y-limits from this axis' own data."""
    vals = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return
    lo, hi = vals.min(), vals.max()
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    ax.set_ylim(lo - pad, hi + pad)


def _style(ax, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, lw=0.6)
    ax.tick_params(labelsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)


def _legend_above(ax, ncol):
    """Legend in the margin ABOVE the axes, so it never sits on the data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol, fontsize=7.5,
              frameon=False, borderaxespad=0.15, handlelength=1.8, columnspacing=1.2)


def _title(ax, text, color=MUTED):
    ax.set_title(text, loc="right", fontsize=8, color=color, pad=3)


def build_dashboard(seed: int = ls.SEED, queue_fills: bool = True) -> SimpleNamespace:
    # 'p' is matplotlib's pan key by default; here it pauses, so unbind it from pan
    plt.rcParams["keymap.pan"] = [k for k in plt.rcParams["keymap.pan"] if k != "p"]
    mm = ls.LiveMarketMaker(seed=seed, queue_fills=queue_fills)
    ui = dict(tab="Market", truth_debug=False, epoch_id=None)

    fig = plt.figure(figsize=(17, 10.5), facecolor="white")
    fig.text(0.03, 0.972, "Live market-making dashboard", fontsize=13, weight="bold",
             va="center")
    fig.text(0.03, 0.948, "naive vs inventory-aware on identical prices, fills and informed "
             "flow  ·  synthetic data  ·  statistics are per epoch, never pooled across "
             "settings", fontsize=8.5, color=MUTED, va="center")

    # ------------------------------------------------------------ Market tab
    ax_price = fig.add_axes([0.05, 0.44, 0.59, 0.38])
    ax_tox = fig.add_axes([0.05, 0.08, 0.59, 0.25], sharex=ax_price)
    (line_mid,) = ax_price.plot([], [], color=MID_C, lw=1.3, label="mid")
    (line_bid,) = ax_price.plot([], [], color=AWARE_C, lw=0.9, label="aware bid")
    (line_ask,) = ax_price.plot([], [], color=AWARE_C, lw=0.9, ls="--", label="aware ask")
    (line_bid_n,) = ax_price.plot([], [], color=NAIVE_C, lw=0.7, alpha=0.8, label="naive bid")
    (line_ask_n,) = ax_price.plot([], [], color=NAIVE_C, lw=0.7, alpha=0.8, ls="--",
                                  label="naive ask")
    scat_bid = ax_price.scatter([], [], marker="^", color=AWARE_C, s=24, zorder=4,
                                label="aware buy / sell")
    scat_ask = ax_price.scatter([], [], marker="v", color=AWARE_C, s=24, zorder=4)
    scat_bid_n = ax_price.scatter([], [], marker="^", facecolor="none", edgecolor=NAIVE_C,
                                  s=18, lw=0.8, zorder=3, label="naive buy / sell")
    scat_ask_n = ax_price.scatter([], [], marker="v", facecolor="none", edgecolor=NAIVE_C,
                                  s=18, lw=0.8, zorder=3)
    ax_price.fill_between([], [], color=TRUTH_C, alpha=0.35, lw=0,
                          label="hidden informed period (truth, display only)")
    _style(ax_price, "price")
    _legend_above(ax_price, 5)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    (line_score,) = ax_tox.plot([], [], color=INK, lw=1.2, label="detector toxicity score")
    (line_imb,) = ax_tox.plot([], [], color="#1a9988", lw=0.9, label="book imbalance")
    ax_tox.axhline(ls.ALERT, color=ASK_C, ls="--", lw=0.8, label=f"alert threshold {ls.ALERT}")
    ax_tox.axhline(0, color=MUTED, lw=0.6)
    ax_tox.set_ylim(-1.0, 1.05)
    _style(ax_tox, "detector")
    ax_tox.set_xlabel("tick  (dotted lines = session boundaries)", fontsize=8)
    _legend_above(ax_tox, 3)

    # ------------------------------------------------------------ Book tab
    ax_ladder = fig.add_axes([0.04, 0.07, 0.22, 0.78])
    ax_ladder.set_axis_off()
    ax_heat = fig.add_axes([0.32, 0.61, 0.27, 0.22])
    ax_heat_cb = fig.add_axes([0.597, 0.61, 0.007, 0.22])
    ax_queue = fig.add_axes([0.32, 0.375, 0.27, 0.15], sharex=ax_heat)
    ax_bub = fig.add_axes([0.32, 0.08, 0.27, 0.22], sharex=ax_heat)
    ax_bub_cb = fig.add_axes([0.597, 0.08, 0.007, 0.22])
    cmap_depth = plt.get_cmap("viridis").with_extremes(bad=(1, 1, 1, 0))
    heat_im = ax_heat.imshow(np.full((2, 2), np.nan), origin="lower", aspect="auto",
                             cmap=cmap_depth, interpolation="nearest", extent=(0, 1, 0, 1))
    heat_cb = fig.colorbar(heat_im, cax=ax_heat_cb)
    heat_cb.set_label("resting size (lots)", fontsize=7.5)
    heat_cb.ax.tick_params(labelsize=7)
    (heat_bid,) = ax_heat.plot([], [], color="white", lw=1.6)
    (heat_bid2,) = ax_heat.plot([], [], color=AWARE_C, lw=1.0, label="our bid")
    (heat_ask,) = ax_heat.plot([], [], color="white", lw=1.6)
    (heat_ask2,) = ax_heat.plot([], [], color=AWARE_C, lw=1.0, ls="--", label="our ask")
    (heat_mid,) = ax_heat.plot([], [], color="white", lw=0.8, alpha=0.7)
    ax_heat.plot([], [], color=MID_C, lw=0.8, label="mid (white)")
    _style(ax_heat, "price")
    ax_heat.grid(False)
    _legend_above(ax_heat, 3)
    plt.setp(ax_heat.get_xticklabels(), visible=False)

    (q_bid,) = ax_queue.plot([], [], color=BID_C, lw=1.2, label="our bid")
    (q_ask,) = ax_queue.plot([], [], color=ASK_C, lw=1.2, label="our ask")
    ax_queue.set_ylim(-0.05, 1.05)
    _style(ax_queue, "share ahead")
    _legend_above(ax_queue, 2)
    queue_na = ax_queue.text(0.5, 0.5, "", transform=ax_queue.transAxes, ha="center",
                             va="center", fontsize=8.5, color=MUTED)
    plt.setp(ax_queue.get_xticklabels(), visible=False)

    cmap_score = plt.get_cmap("plasma")
    cmap_truth = ListedColormap(["#8fb3d9", "#c0392b"])
    bub = ax_bub.scatter([], [], c=[], s=[], cmap=cmap_score, vmin=0, vmax=1,
                         alpha=0.7, edgecolor="white", linewidths=0.3)
    bub_cb = fig.colorbar(bub, cax=ax_bub_cb)
    bub_cb.ax.tick_params(labelsize=7)
    (bub_mid,) = ax_bub.plot([], [], color=MID_C, lw=0.7, alpha=0.6)
    _style(ax_bub, "price")
    ax_bub.set_xlabel("tick", fontsize=8)

    # ------------------------------------------------------------ Strategies tab
    strat_rows = [(0.675, "inventory"), (0.475, "MTM P&L"), (0.275, "spread P&L"),
                  (0.075, "inventory P&L")]
    strat_axes = []
    for i, (y, lab) in enumerate(strat_rows):
        ax = fig.add_axes([0.05, y, 0.59, 0.165],
                          sharex=strat_axes[0] if strat_axes else None)
        _style(ax, lab)
        ax.axhline(0, color=MUTED, lw=0.6)
        if i < len(strat_rows) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        strat_axes.append(ax)
    ax_inv, ax_mtm, ax_spr, ax_ipnl = strat_axes

    def pair(ax):
        (ln,) = ax.plot([], [], color=NAIVE_C, lw=1.4, label="naive (eta = 0)")
        (la,) = ax.plot([], [], color=AWARE_C, lw=1.4, ls="--", label="aware (configured eta)")
        return ln, la

    inv_n, inv_a = pair(ax_inv)
    inv_n.set_drawstyle("steps-post")
    inv_a.set_drawstyle("steps-post")
    mtm_n, mtm_a = pair(ax_mtm)
    spr_n, spr_a = pair(ax_spr)
    ipl_n, ipl_a = pair(ax_ipnl)
    _legend_above(ax_inv, 2)
    _title(ax_inv, "each panel has its own y-scale · P&L and inventory reset each session")
    ax_ipnl.set_xlabel("tick", fontsize=8)

    # ------------------------------------------------------------ Stats tab
    ax_diff = fig.add_axes([0.05, 0.065, 0.59, 0.295])
    ax_diff.axhline(0, color=MUTED, lw=0.8)
    (line_dmean,) = ax_diff.plot([], [], color=INK, lw=1.5, label="running mean")
    scat_d = ax_diff.scatter([], [], s=12, color=MUTED, alpha=0.7, label="one session")
    ax_diff.fill_between([], [], color=INK, alpha=0.15, lw=0, label="± 1 se")
    _style(ax_diff, "aware − naive\nterminal P&L")
    ax_diff.set_xlabel("completed sessions in the current epoch", fontsize=8)
    _legend_above(ax_diff, 3)
    mono = dict(family="monospace", va="top")
    # Three tiers: the answer (large), its supporting table, then the config (smallest).
    head_text = fig.text(0.05, 0.845, "", fontsize=13, **mono)
    banner_text = fig.text(0.05, 0.803, "", fontsize=8.8, color=WARN, weight="bold", **mono)
    epoch_text = fig.text(0.05, 0.768, "", fontsize=8.3, **mono)       # tier 2
    cfg_head = fig.text(0.05, 0.652, "CONFIGURATION  (• differs from project default)",
                        fontsize=7.0, color=MUTED, **mono)
    cfg_texts = [fig.text(x, 0.632, "", fontsize=6.8, color=MUTED, linespacing=1.45, **mono)
                 for x in (0.05, 0.195)]                                # tier 3, two columns
    run_text = fig.text(0.395, 0.768, "", fontsize=8.3, **mono)
    hist_head = fig.text(0.395, 0.678, "", fontsize=8.3, **mono)
    hist_texts = [fig.text(0.395, 0.638 - 0.0195 * i, "", fontsize=7.4, **mono)
                  for i in range(5)]
    footnote = fig.text(0.395, 0.510, "", fontsize=7.2, color=MUTED, linespacing=1.6, **mono)

    # ------------------------------------------------------------ control panel
    session_text = fig.text(0.69, 0.905, "", fontsize=7.8, **mono)
    fig.text(0.69, 0.785, "SETTINGS  (red tick = this epoch's start · def = project default)",
             fontsize=7.6, color=MUTED, va="center")
    sliders, captions, meanings = [], {}, {}
    y = 0.745
    for key, sym, meaning, lo, hi, default, model in SLIDER_SPECS:
        if key == "depth_base":
            fig.text(0.69, y + 0.022, "QUEUE MODEL (order book)", fontsize=7.6,
                     color=MUTED, weight="bold", va="center")
            y -= 0.020
        meanings[key] = meaning
        captions[key] = fig.text(0.69, y + 0.019, meaning, fontsize=7.4, color=INK,
                                 va="center")
        sl_ax = fig.add_axes([0.735, y, 0.150, 0.013])
        init = mm.config()[key]
        s = Slider(sl_ax, sym, lo, hi, valinit=init, color="#c9d6e3", track_color="#eeeeee")
        s.label.set_fontsize(7.8)
        s.valtext.set_fontsize(7.6)
        s.valtext.set_text(f"{init:.3g}  · def {default:.3g}")
        s.vline.set_linewidth(1.6)

        def changed(v, k=key, d=default, sl=s):
            mm.request(k, v)                          # queued, applied between ticks
            sl.valtext.set_text(f"{v:.3g}  · def {d:.3g}")
        s.on_changed(changed)
        sliders.append(s)
        y -= 0.045

    ax_chk = fig.add_axes([0.69, 0.165, 0.295, 0.105])
    ax_chk.set_title("SWITCHES", loc="left", fontsize=7.6, color=MUTED, pad=2)
    check_specs = [("auto_eta", "Auto-eta: raise skew as inventory grows"),
                   ("auto_widen", "Auto-widen: widen spread on detector alerts"),
                   ("queue_fills", "Queue fills (order book); off = exponential"),
                   ("truth_debug", "DEBUG: colour bubbles by TRUE hidden state")]
    chk = CheckButtons(ax_chk, [lab for _, lab in check_specs],
                       [mm.auto_eta, mm.auto_widen, mm.queue_fills, False])
    for t in chk.labels:
        t.set_fontsize(7.8)
    check_key = {lab: key for key, lab in check_specs}

    btn_pause = Button(fig.add_axes([0.69, 0.10, 0.092, 0.045]), "Pause")
    btn_reset = Button(fig.add_axes([0.791, 0.10, 0.092, 0.045]), "Reset")
    btn_freeze = Button(fig.add_axes([0.892, 0.10, 0.092, 0.045]), "Freeze")
    fig.text(0.69, 0.078, "keys: space or p = pause / resume", fontsize=7.4, color=MUTED,
             va="center")

    tab_buttons = {}
    for i, name in enumerate(TABS):
        tab_buttons[name] = Button(fig.add_axes([0.05 + i * 0.078, 0.888, 0.072, 0.03]), name)
        tab_buttons[name].label.set_fontsize(8.5)

    # Status line: every click leaves a visible trace here, so a handler that never
    # runs (or raises) is obvious in the UI instead of failing silently.
    status = fig.text(0.05, 0.018, "ready", family="monospace", fontsize=8,
                      va="bottom", color="#444444")
    counter = itertools.count(1)

    def announce(msg, error=False):
        status.set_text(f"[{time.strftime('%H:%M:%S')}] #{next(counter)}  {msg}")
        status.set_color("#b00020" if error else "#444444")
        fig.canvas.draw_idle()

    def action(name, fn):
        """Wrap a widget callback so it can neither die silently nor kill the loop."""
        def inner(*a, **k):
            try:
                announce(f"{name}: {fn(*a)}")
            except Exception as exc:                      # noqa: BLE001 - surface anything
                traceback.print_exc()
                announce(f"{name} FAILED: {type(exc).__name__}: {exc}", error=True)
        return inner

    def set_paused(flag):
        mm.paused = flag
        btn_pause.label.set_text("Resume" if flag else "Pause")

    def toggle_pause(*_):
        set_paused(not mm.paused)
        return "paused" if mm.paused else "running"

    def do_reset(*_):
        print(mm.epochs.table())
        print("-- reset: new run, parameters kept --")
        n = mm.epochs.n_completed
        mm.reset()
        return f"cleared {n} completed session(s); epoch table printed to terminal"

    def do_freeze(*_):
        """Pause and print the current epoch as a reproducible, self-checking script."""
        mm.apply_pending()
        set_paused(True)
        e = mm.epochs.current
        report = reproduction_report(e)
        print(report)
        return (f"epoch #{e.id} ({len(e.sessions)} sessions) -> "
                f"{len(report.splitlines())}-line script printed to TERMINAL")

    def do_check(label):
        key = check_key[label]
        if key == "truth_debug":                       # display only: never a config
            ui["truth_debug"] = not ui["truth_debug"]
            return ("bubble colour = TRUE hidden state (DEBUG)" if ui["truth_debug"]
                    else "bubble colour = detector score")
        cur = mm._pending.get(key, getattr(mm, key))   # double clicks within one frame
        mm.request(key, not cur)
        return f"{key} -> {not cur} (applies at the next tick; opens a new epoch)"

    tab_members = {
        "Market": [ax_price, ax_tox],
        "Book": [ax_ladder, ax_heat, ax_heat_cb, ax_queue, ax_bub, ax_bub_cb],
        "Strategies": strat_axes,
        # banner_text is deliberately NOT a member: its visibility depends on n too,
        # so update_stats_text owns it (see there).
        "Stats": [ax_diff, head_text, epoch_text, cfg_head, *cfg_texts, run_text,
                  hist_head, footnote, *hist_texts],
    }

    def show_tab(name):
        ui["tab"] = name
        for tab, members in tab_members.items():
            for art in members:
                art.set_visible(tab == name)
        for tab, btn in tab_buttons.items():
            active = tab == name
            btn.color = "#d9e4ef" if active else "#f4f4f4"
            btn.hovercolor = "#c9d6e3" if active else "#e6e6e6"
            btn.ax.set_facecolor(btn.color)
            btn.label.set_weight("bold" if active else "normal")
        fig.canvas.draw_idle()
        return f"showing {name}"

    on_pause, on_reset, on_freeze = (action("Pause", toggle_pause), action("Reset", do_reset),
                                     action("Freeze", do_freeze))
    btn_pause.on_clicked(on_pause)
    btn_reset.on_clicked(on_reset)
    btn_freeze.on_clicked(on_freeze)
    chk.on_clicked(action("Switch", do_check))
    for name, btn in tab_buttons.items():
        btn.on_clicked(action("Tab", lambda *_, t=name: show_tab(t)))

    def on_key(event):
        if event.key in (" ", "p"):
            on_pause()

    fig.canvas.mpl_connect("key_press_event", on_key)

    decor = []            # per-frame artists: truth bands, session boundaries
    diff_decor = []       # paired-difference band
    cache = {"diff_key": None}

    # ---------------------------------------------------------------- drawing
    def draw_decor(axes, x, truth, brk_idx, shade):
        for c in decor:
            c.remove()
        decor.clear()
        for ax in axes:
            if shade:
                decor.append(ax.fill_between(
                    x, 0, 1, where=truth > 0.5, transform=ax.get_xaxis_transform(),
                    color=TRUTH_C, alpha=0.35, lw=0))
            for i in brk_idx:
                decor.append(ax.axvline(x[i] - 0.5, color=MUTED, ls=":", lw=0.8))

    def draw_ladder(db):
        ax = ax_ladder
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        queue = db.source == "queue"
        L = LADDER_LEVELS if queue else len(db.bid_px)
        asks = list(zip(db.ask_px[:L], db.ask_sz[:L]))[::-1]     # far -> near, top down
        bids = list(zip(db.bid_px[:L], db.bid_sz[:L]))            # near -> far
        rows = [("ask", p, s) for p, s in asks] + [("mid", db.mid, None)] + \
               [("bid", p, s) for p, s in bids]
        top, bottom = 0.90, 0.16
        rh = (top - bottom) / len(rows)
        biggest = max([s for _, _, s in rows if s is not None] + [1e-9])
        head = dict(fontsize=7.8, color=MUTED, va="center")
        ax.text(0.12, top + 0.02, "bid size (lots)", ha="right", **head)
        ax.text(0.50, top + 0.02, "price", ha="center", **head)
        ax.text(0.88, top + 0.02, "ask size (lots)", ha="left", **head)
        tol = db.tick / 2
        shown = {"bid": False, "ask": False}
        for i, (side, price, size) in enumerate(rows):
            yc = top - (i + 0.5) * rh
            if side == "mid":
                ax.plot([0.02, 0.98], [yc, yc], color=MUTED, lw=0.6)
                ax.text(0.5, yc, f"mid {price:.2f}", ha="center", va="center", fontsize=7.6,
                        color=MUTED, backgroundcolor="white")
                continue
            ours = queue and _finite(getattr(db, f"our_{side}")) and \
                abs(price - getattr(db, f"our_{side}")) < tol
            if ours:
                shown[side] = True
                ax.add_patch(Rectangle((0.0, yc - rh / 2), 1.0, rh, color=AWARE_C,
                                       alpha=0.14, lw=0, zorder=0))
            w = 0.29 * size / biggest
            colour = BID_C if side == "bid" else ASK_C
            if side == "bid":
                ax.add_patch(Rectangle((0.43 - w, yc - rh * 0.32), w, rh * 0.64, color=colour,
                                       alpha=0.55, lw=0))
                ax.text(0.12, yc, f"{size:.1f}", ha="right", va="center", fontsize=7.6)
            else:
                ax.add_patch(Rectangle((0.57, yc - rh * 0.32), w, rh * 0.64, color=colour,
                                       alpha=0.55, lw=0))
                ax.text(0.88, yc, f"{size:.1f}", ha="left", va="center", fontsize=7.6)
            ax.text(0.50, yc, f"{price:.2f}", ha="center", va="center", fontsize=7.8,
                    weight="bold" if ours else "normal", color=colour)
            if ours:
                ahead, at = getattr(db, f"ahead_{side}"), getattr(db, f"depth_at_{side}")
                msg = f"ours: {ahead:.1f} of {at:.1f} ahead"
                if side == "bid":           # bid rows have no ask bar: use that column
                    ax.text(0.57, yc, "◀ " + msg, ha="left", va="center", fontsize=7.2,
                            color=AWARE_C, weight="bold")
                else:
                    ax.text(0.43, yc, msg + " ▶", ha="right", va="center", fontsize=7.2,
                            color=AWARE_C, weight="bold")
        foot = []
        if queue:
            ax.set_title("ORDER BOOK LADDER  ·  fill-driving book (aware)", loc="left",
                         fontsize=8.5, weight="bold")
            for side in ("bid", "ask"):
                p = getattr(db, f"our_{side}")
                if not _finite(p):
                    foot.append(f"our {side}: none resting (just filled)")
                elif not shown[side]:
                    foot.append(f"our {side} @ {p:.2f}: beyond the {L} levels shown")
                else:
                    foot.append(f"our {side} @ {p:.2f}: {getattr(db, f'size_{side}'):.2f} lots "
                                f"resting, {getattr(db, f'ahead_{side}'):.1f} lots ahead")
            foot.append("'ahead' = size in front of us at our price; we fill only once a")
            foot.append("market order consumes all of it (moving the quote = back of queue)")
        else:
            ax.set_title("SYNTHETIC BOOK  ·  display only", loc="left", fontsize=8.5,
                         weight="bold")
            foot += ["exponential fills have no order book and no queue position.",
                     "shown: the detector's synthetic book, generated from the hidden",
                     "informed direction (see Limitations). Tick 'Queue fills' for the",
                     f"real book. our quotes: bid {fmt(db.our_bid, '.2f')} / "
                     f"ask {fmt(db.our_ask, '.2f')}"]
        for i, line in enumerate(foot):
            ax.text(0.0, 0.125 - i * 0.022, line, fontsize=7.2, color=INK if i < 2 else MUTED,
                    va="top", family="monospace")

    def draw_book_tab():
        entries = list(mm.book_hist)
        if not entries:
            return
        db = entries[-1]
        draw_ladder(db)
        # the window can straddle a fill-model switch; draw only the current source
        entries = [e for e in entries if e.source == db.source]
        xs = np.array([e.n for e in entries], dtype=float)
        tick = db.tick
        Lh = HEAT_LEVELS if db.source == "queue" else len(db.bid_px)
        idx_all = np.concatenate([np.rint(np.r_[e.bid_px[:Lh], e.ask_px[:Lh]] / tick)
                                  for e in entries]).astype(int)
        lo, hi = idx_all.min(), idx_all.max()
        grid = np.full((hi - lo + 1, len(entries)), np.nan)
        for c, e in enumerate(entries):
            for px, sz in ((e.bid_px[:Lh], e.bid_sz[:Lh]), (e.ask_px[:Lh], e.ask_sz[:Lh])):
                grid[np.rint(px / tick).astype(int) - lo, c] = sz
        heat_im.set_data(grid)
        heat_im.set_extent((xs[0] - 0.5, xs[-1] + 0.5, (lo - 0.5) * tick, (hi + 0.5) * tick))
        vmax = np.nanpercentile(grid, 98) if np.isfinite(grid).any() else 1.0
        heat_im.set_clim(0, vmax)
        heat_cb.update_normal(heat_im)
        ob = np.array([e.our_bid for e in entries])
        oa = np.array([e.our_ask for e in entries])
        mids = np.array([e.mid for e in entries])
        for ln, yv in ((heat_bid, ob), (heat_bid2, ob), (heat_ask, oa), (heat_ask2, oa),
                       (heat_mid, mids)):
            ln.set_data(xs, yv)
        ax_heat.set_xlim(xs[0], max(xs[-1], xs[0] + 50))
        quotes = np.r_[ob, oa]
        quotes = quotes[np.isfinite(quotes)]
        y_lo = min((lo - 0.5) * tick, quotes.min() - tick) if quotes.size else (lo - 0.5) * tick
        y_hi = max((hi + 0.5) * tick, quotes.max() + tick) if quotes.size else (hi + 0.5) * tick
        ax_heat.set_ylim(y_lo, y_hi)
        _title(ax_heat, "depth over time" + ("" if db.source == "queue" else " (synthetic)"))

        # our queue position over time: share of the level's size that is ahead of us
        if db.source == "queue":
            fb = np.array([e.ahead_bid / e.depth_at_bid if _finite(e.depth_at_bid)
                           and e.depth_at_bid > 0 else np.nan for e in entries])
            fa = np.array([e.ahead_ask / e.depth_at_ask if _finite(e.depth_at_ask)
                           and e.depth_at_ask > 0 else np.nan for e in entries])
            q_bid.set_data(xs, fb)
            q_ask.set_data(xs, fa)
            queue_na.set_text("")
        else:
            q_bid.set_data([], [])
            q_ask.set_data([], [])
            queue_na.set_text("n/a under exponential fills (no queue)")
        _title(ax_queue, "our queue position: 0 = front of queue, 1 = back")

        # order-cluster bubbles
        cols = entries[::-1][::BUBBLE_EVERY][::-1]
        Lb = BUBBLE_LEVELS if db.source == "queue" else len(db.bid_px)
        pts, sizes, vals = [], [], []
        for e in cols:
            colour_val = e.truth if ui["truth_debug"] else e.score
            for px, sz in ((e.bid_px[:Lb], e.bid_sz[:Lb]), (e.ask_px[:Lb], e.ask_sz[:Lb])):
                pts.append(np.column_stack([np.full(len(px), e.n), px]))
                sizes.append(sz)
                vals.append(np.full(len(px), colour_val))
        pts, sizes, vals = np.vstack(pts), np.concatenate(sizes), np.concatenate(vals)
        scale = np.percentile(sizes, 95) if sizes.size else 1.0
        bub.set_offsets(pts)
        bub.set_sizes(2 + 38 * np.clip(sizes / max(scale, 1e-9), 0, 1.5))
        bub.set_array(vals)
        if ui["truth_debug"]:
            bub.set_cmap(cmap_truth)
            bub_cb.set_label("TRUE state: 0 quiet / 1 informed", fontsize=7.2, color=ASK_C)
            _title(ax_bub, "DEBUG: colour = TRUE hidden state · validation only, "
                           "never used by a strategy or statistic", color=ASK_C)
        else:
            bub.set_cmap(cmap_score)
            bub_cb.set_label("detector toxicity score", fontsize=7.2, color=INK)
            _title(ax_bub, "order clusters · size = resting depth · colour = detector score")
        bub.set_clim(0, 1)
        bub_cb.update_normal(bub)
        bub_mid.set_data(xs, mids)
        ax_bub.set_ylim(y_lo, y_hi)

    def draw_paired_chart():
        e = mm.epochs.current
        key = (e.id, len(e.sessions))
        if key == cache["diff_key"]:
            return
        cache["diff_key"] = key
        for c in diff_decor:
            c.remove()
        diff_decor.clear()
        k, mean, se = e.running_paired()
        d = e.paired_diffs()
        thin = len(d) < MIN_N
        _title(ax_diff, f"paired difference, epoch #{e.id}  ·  n = {len(d)} sessions"
                        + ("  ·  under-powered: read as noise, not a result" if thin else ""),
               color=MUTED if thin else INK)
        if not len(d):
            line_dmean.set_data([], [])
            scat_d.set_offsets(np.empty((0, 2)))
            return
        line_dmean.set_data(k, mean)
        line_dmean.set_color(MUTED if thin else INK)
        scat_d.set_offsets(np.column_stack([k, d]))
        ok = ~np.isnan(se)
        if ok.any():
            diff_decor.append(ax_diff.fill_between(k[ok], (mean - se)[ok], (mean + se)[ok],
                                                   color=INK, alpha=0.15, lw=0))
        ax_diff.set_xlim(0.5, max(len(d), 5) + 0.5)
        _set_ylim(ax_diff, d, mean - np.where(ok, se, 0), mean + np.where(ok, se, 0),
                  np.zeros(1))

    def update_stats_text():
        e = mm.epochs.current
        s = e.stats()
        n = s["n"]
        thin = n < MIN_N
        n_inf = sum(r.n_active for r in e.sessions)
        n_quiet = sum(r.n_quiet for r in e.sessions)

        # TIER 1 - the research question itself, given the top of the panel.
        head_text.set_text(f"aware − naive   {fmt(s['diff_mean']):>8} ± "
                           f"{fmt(s['diff_se'], '.2f'):>6}    t = {fmt(s['diff_t']):>6}"
                           f"    n = {n}")
        head_text.set_color(WARN if thin else INK)
        banner_text.set_text(f"UNDER-POWERED — n = {n}, need ~{MIN_N}+ before reading "
                             f"anything into this")
        # visibility is owned here, not by show_tab: it depends on n as well as the tab
        banner_text.set_visible(thin and ui["tab"] == "Stats")

        # TIER 2 - the supporting table, numbers aligned on the decimal point.
        epoch_text.set_text(
            f"CURRENT EPOCH #{e.id}  ·  n = {n} completed  ·  "
            f"{s['incomplete']} interrupted (excluded)\n"
            f"{'':7}{'mean ± se':>17}{'Sharpe':>9}{'max |q|':>10}{'worst dd':>11}{'n':>5}\n"
            f"{'aware':7}{fmt(s['mean']):>7} ± {fmt(s['se'], '.2f'):>6}"
            f"{fmt(s['sharpe']):>9}{fmt(s['max_abs_q'], '.2f'):>10}"
            f"{fmt(s['worst_dd'], '.2f'):>11}{n:>5}\n"
            f"{'naive':7}{fmt(s['naive_mean']):>7} ± {fmt(s['naive_se'], '.2f'):>6}"
            f"{fmt(s['naive_sharpe']):>9}{fmt(s['naive_max_abs_q'], '.2f'):>10}"
            f"{fmt(s['naive_worst_dd'], '.2f'):>11}{n:>5}\n\n"
            f"detector   detect {pct(s['detect'])} of {n_inf} informed ticks\n"
            f"{'':11}false alarms {pct(s['false_alarm'])} of {n_quiet} quiet ticks")
        epoch_text.set_color(INK)

        # TIER 3 - the config, smallest text on the panel.
        for t, col in zip(cfg_texts, config_columns(e.config)):
            t.set_text(col)

        footnote.set_text(
            ("fewer than 30 completed sessions: treat as noise, not a result.\n" if thin else "")
            + "Sharpe = per session, not annualised, no risk-free rate.\n"
            "max |q| and worst drawdown cover every tick, interrupted sessions included.")
        run_text.set_text(
            "RUN-WIDE  (all epochs since start or Reset)\n"
            f"sessions completed {mm.epochs.n_completed}  ·  incomplete "
            f"{mm.epochs.n_incomplete}  ·  epochs opened {len(mm.epochs.epochs)}\n"
            f"max |q| so far   aware {mm.run_max_abs_q:7.2f}   naive {mm.run_max_abs_q_n:7.2f}\n"
            f"worst drawdown   aware {mm.run_worst_dd:7.2f}   naive {mm.run_worst_dd_n:7.2f}")
        all_closed = [ep for ep in mm.epochs.epochs[:-1] if ep.sessions]
        closed = all_closed[-len(hist_texts):]
        hist_head.set_text("CLOSED EPOCHS  (each its own config; never pooled)\n"
                           f"{'#':<4}{'n':>4}{'paired ± se':>18}   changed from previous"
                           if closed else "CLOSED EPOCHS\n(none yet)")
        hist_head.set_color(INK if closed else MUTED)
        for i, t in enumerate(hist_texts):
            if i >= len(closed):
                t.set_text("")
                continue
            ep = closed[i]
            prev = all_closed[all_closed.index(ep) - 1].config if all_closed.index(ep) else None
            st = ep.stats()
            change = config_str(ep.config, prev) if prev else "first epoch"
            t.set_text(f"{('#' + str(ep.id)):<4}{st['n']:>4}"
                       f"{fmt(st['diff_mean']):>9} ± {fmt(st['diff_se'], '.2f'):>6}   "
                       f"{change[:30]}")
            t.set_color(MUTED if st["n"] < MIN_N else INK)

    def update_controls():
        e = mm.epochs.current
        if ui["epoch_id"] != e.id:                     # red tick = this epoch's start value
            ui["epoch_id"] = e.id
            for s, spec in zip(sliders, SLIDER_SPECS):
                v = e.config.get(spec[0], spec[5])
                s.vline.set_xdata([v, v])
        for key, *_, model in SLIDER_SPECS:            # grey the sliders the model ignores
            inactive = (model == "queue" and not mm.queue_fills) or \
                       (model == "exp" and mm.queue_fills)
            captions[key].set_color(MUTED if inactive else INK)
            captions[key].set_text(meanings[key] + (_UNUSED if inactive else ""))
        resid = mm.pnl - (mm.spread_pnl + mm.inventory_pnl)
        session_text.set_text(
            f"SESSION #{mm.session}  tick {mm.session_step}/{mm.session_steps}   fills: "
            f"{'QUEUE (order book)' if mm.queue_fills else 'exponential'}\n"
            f"mid {mm.m:8.2f}          P&L identity error {resid:.1e}\n"
            f"inventory    aware {mm.q:+7.2f}   naive {mm.q_n:+7.2f}\n"
            f"session P&L  aware {mm.pnl:+7.2f}   naive {mm.pnl_n:+7.2f}\n"
            f"eta now {mm.eta_eff:.3f}{' (auto)' if mm.auto_eta else ''}   half-spread "
            f"{mm.delta_eff:.3f}{' (auto)' if mm.auto_widen else ''}\n"
            f"detector score {mm.score:.2f}{' ALERT' if mm.score >= ls.ALERT else ''}   "
            f"imbalance {mm.imb:+.2f}")

    # ---------------------------------------------------------------- the loop
    def update(_frame=None):
        mm.apply_pending()                 # between ticks, also while paused
        if not mm.paused:
            for _ in range(STEPS_PER_FRAME):
                mm.step()

        h = mm.hist
        if not h["n"]:
            return
        a = {k: np.fromiter(v, dtype=float) for k, v in h.items()}
        x = a["n"]
        brk = np.flatnonzero(np.diff(a["sess"]) != 0) + 1

        def b(arr):                        # NaN at session boundaries -> broken lines
            return np.insert(arr, brk, np.nan)

        xb = b(x)
        # cheap line data for EVERY tab, every frame (tabs never change what's computed)
        for ln, key in ((line_mid, "mid"), (line_bid, "bid"), (line_ask, "ask"),
                        (line_bid_n, "bid_n"), (line_ask_n, "ask_n"),
                        (inv_a, "q"), (inv_n, "q_n"), (mtm_a, "pnl"), (mtm_n, "pnl_n"),
                        (spr_a, "spread"), (spr_n, "spread_n"), (ipl_a, "inv"),
                        (ipl_n, "inv_n"), (line_score, "score"), (line_imb, "imb")):
            ln.set_data(xb, b(a[key]))
        x0 = x[0]
        for sc, pts in ((scat_bid, mm.bid_fill_pts), (scat_ask, mm.ask_fill_pts),
                        (scat_bid_n, mm.bid_fill_pts_n), (scat_ask_n, mm.ask_fill_pts_n)):
            inside = [pt for pt in pts if pt[0] >= x0]
            sc.set_offsets(np.array(inside) if inside else np.empty((0, 2)))
        xlim = (x0, max(x[-1], x0 + 50))
        ax_price.set_xlim(*xlim)
        _set_ylim(ax_price, a["bid"], a["ask"], a["bid_n"], a["ask_n"], a["mid"])
        ax_inv.set_xlim(*xlim)
        qmax = max(5, np.abs(np.concatenate([a["q"], a["q_n"]])).max() + 1)
        ax_inv.set_ylim(-qmax, qmax)
        _set_ylim(ax_mtm, a["pnl"], a["pnl_n"])          # each P&L panel: its own scale
        _set_ylim(ax_spr, a["spread"], a["spread_n"])
        _set_ylim(ax_ipnl, a["inv"], a["inv_n"])
        draw_paired_chart()
        update_stats_text()
        update_controls()

        # expensive rebuilds: visible tab only
        tab = ui["tab"]
        if tab == "Market":
            draw_decor((ax_price, ax_tox), x, a["truth"], brk, shade=True)
        elif tab == "Strategies":
            draw_decor(strat_axes, x, a["truth"], brk, shade=False)
        elif tab == "Book":
            draw_book_tab()

    show_tab("Market")
    status.set_text("ready  ·  queue fills ON by default (the model the headline findings "
                    "use)" if mm.queue_fills else "ready  ·  exponential fills")

    # NOTE: matplotlib widgets are kept alive ONLY by the caller's reference. Anything
    # dropped here is garbage collected and its callbacks go dead while the widget is
    # still drawn (this is exactly how Reset and Freeze once broke). Keep every widget.
    return SimpleNamespace(
        fig=fig, mm=mm, update=update, ui=ui, show_tab=show_tab,
        do_freeze=on_freeze, do_reset=on_reset, toggle_pause=on_pause,
        sliders=sliders, chk=chk, status=status, captions=captions,
        buttons=dict(pause=btn_pause, reset=btn_reset, freeze=btn_freeze),
        tabs=tab_buttons, tab_members=tab_members,
        axes=dict(price=ax_price, tox=ax_tox, inv=ax_inv, mtm=ax_mtm, spread=ax_spr,
                  inv_pnl=ax_ipnl, diff=ax_diff, ladder=ax_ladder, heat=ax_heat,
                  queue=ax_queue, bubble=ax_bub),
        texts=dict(session=session_text, headline=head_text, banner=banner_text,
                   epoch=epoch_text, config=cfg_texts, run=run_text,
                   history=hist_texts, footnote=footnote),
        artists=dict(heat=heat_im, bubble=bub, bubble_cb=bub_cb),
        lines=dict(inv_n=inv_n, inv_a=inv_a, mtm_n=mtm_n, mtm_a=mtm_a, spr_n=spr_n, spr_a=spr_a,
                   ipl_n=ipl_n, ipl_a=ipl_a, dmean=line_dmean, bid_n=line_bid_n,
                   ask_n=line_ask_n, q_bid=q_bid, q_ask=q_ask))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=ls.SEED)
    ap.add_argument("--exponential", action="store_true",
                    help="start with exponential fills instead of the queue model")
    args = ap.parse_args()
    d = build_dashboard(args.seed, queue_fills=not args.exponential)
    anim = FuncAnimation(d.fig, d.update, interval=40, cache_frame_data=False)  # noqa: F841
    try:
        plt.show()
    finally:
        print(d.mm.epochs.table())


if __name__ == "__main__":
    main()
