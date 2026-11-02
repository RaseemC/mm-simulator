"""Phase B tests: paired strategies (B1), run-wide extremes (B3), freeze/reproduce (B4)."""
import gc
import importlib.util
import weakref
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig
from mmsim.core import simulate_paths
from mmsim.epochs import Epoch, SessionRecord
from mmsim.live_sim import LiveMarketMaker
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.repro import batch_possible, batch_script, live_script, reproduction_report
from mmsim.strategies import InventorySkewQuoter

N = 200


def run_ticks(mm, n):
    return [mm.step() for _ in range(n)]


def batch(seed, sessions, eta, strength, delta=0.75, sigma=2.0, A=140.0, k=1.5):
    market, adv = MarketConfig(sigma=sigma), AdverseConfig(strength=strength)
    prep = prepare_batch(ArithmeticRandomWalk(market), seed, sessions, adv)
    return simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(delta, eta),
                          FillConfig(A=A, k=k), market.dt, informed=prep.informed,
                          fill_boost=adv.fill_boost)


# ------------------------------------------------------------------ B1: pairing
@pytest.mark.filterwarnings("ignore:fill-probability cap")
@pytest.mark.parametrize("eta", [0.1, 0.5])
def test_both_strategies_match_the_batch_engine(eta):
    sessions = 12
    mm = LiveMarketMaker(seed=17, params=dict(eta=eta, tox=0.5))
    run_ticks(mm, sessions * N)
    recs = mm.epochs.current.sessions
    aware, naive = batch(17, sessions, eta, 0.5), batch(17, sessions, 0.0, 0.5)
    np.testing.assert_allclose([r.terminal_pnl for r in recs], aware["terminal_pnl"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.naive_terminal_pnl for r in recs], naive["terminal_pnl"],
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.naive_spread_pnl for r in recs], naive["state"].spread_pnl,
                               rtol=1e-12, atol=1e-9)
    assert [r.naive_max_abs_q for r in recs] == np.abs(naive["q"]).max(axis=1).astype(int).tolist()


def test_both_strategies_see_identical_mids_and_the_same_draws():
    mm = LiveMarketMaker(seed=3, params=dict(eta=0.3, tox=0.5))
    for _ in range(3 * N):
        mm.step()
        assert mm.state.m[0] == mm.state_n.m[0]      # same mid path, tick by tick
    # same draws => with eta 0 and no auto flags the two strategies are identical
    mm0 = LiveMarketMaker(seed=3, params=dict(eta=0.0, tox=0.5))
    for _ in range(3 * N):
        t = mm0.step()
        assert t.q == t.q_naive and t.pnl == t.pnl_naive
    assert all(d == 0 for d in mm0.epochs.current.paired_diffs())


def test_strategies_really_differ_when_eta_differs():
    mm = LiveMarketMaker(seed=3, params=dict(eta=0.3, tox=0.5))
    recs = run_ticks(mm, 20 * N) and mm.epochs.current.sessions
    assert max(r.naive_max_abs_q for r in recs) > 2 * max(r.max_abs_q for r in recs)
    assert any(r.terminal_pnl != r.naive_terminal_pnl for r in recs)


def test_paired_stats_match_manual():
    mm = LiveMarketMaker(seed=8, params=dict(eta=0.2, tox=0.5))
    run_ticks(mm, 15 * N)
    e = mm.epochs.current
    d = np.array([r.terminal_pnl - r.naive_terminal_pnl for r in e.sessions])
    st = e.stats()
    assert st["n"] == 15
    assert st["diff_mean"] == pytest.approx(d.mean())
    assert st["diff_se"] == pytest.approx(d.std(ddof=1) / np.sqrt(15))
    assert st["diff_t"] == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(15)))
    # pairing pays off: sd of the difference is below the combined spread of the two strategies
    assert st["diff_sd"] < np.hypot(st["sd"], st["naive_sd"])


def test_running_paired_matches_brute_force():
    recs = [SessionRecord(i, 10.0 + v, 0, 0, 0, 0, 0, 0, 0, 0, 0, naive_terminal_pnl=10.0)
            for i, v in enumerate([3.0, -1.0, 4.0, 1.5, -2.5, 6.0])]
    e = Epoch(0, {}, sessions=recs)
    k, mean, se = e.running_paired()
    d = np.array([3.0, -1.0, 4.0, 1.5, -2.5, 6.0])
    assert np.isnan(se[0])
    for j in range(1, len(d) + 1):
        assert mean[j - 1] == pytest.approx(d[:j].mean())
        if j > 1:
            assert se[j - 1] == pytest.approx(d[:j].std(ddof=1) / np.sqrt(j))
    assert k.tolist() == [1, 2, 3, 4, 5, 6]


def test_paired_stats_are_per_epoch_not_pooled():
    mm = LiveMarketMaker(seed=8, params=dict(eta=0.2, tox=0.5))
    run_ticks(mm, 6 * N)
    mm.request("eta", 0.0)                           # aware == naive in the second epoch
    run_ticks(mm, 6 * N)
    e0, e1 = mm.epochs.epochs
    assert e0.stats()["n"] == 6 and e1.stats()["n"] == 6
    assert e0.stats()["diff_mean"] != 0.0
    assert e1.stats()["diff_mean"] == 0.0 and e1.stats()["diff_se"] == 0.0


# ------------------------------------------------------------------ B3: extremes
def test_run_wide_extremes_cover_all_ticks_all_epochs_and_interrupted_sessions():
    mm = LiveMarketMaker(seed=11, params=dict(eta=0.1, tox=0.5))
    ticks = run_ticks(mm, 3 * N + 90)
    mm.request("sigma", 3.0)                         # interrupts session 3 at tick 90
    ticks += run_ticks(mm, 2 * N + 10)
    assert mm.epochs.n_incomplete == 1 and mm.epochs.n_completed == 5
    assert mm.run_max_abs_q == max(abs(t.q) for t in ticks)
    assert mm.run_max_abs_q_n == max(abs(t.q_naive) for t in ticks)

    def worst_dd(ticks, field):
        worst, sess, peak = 0.0, None, 0.0
        for t in ticks:
            if t.session != sess:
                sess, peak = t.session, 0.0
            pnl = getattr(t, field)
            peak = max(peak, pnl)
            worst = max(worst, peak - pnl)
        return worst
    assert mm.run_worst_dd == pytest.approx(worst_dd(ticks, "pnl"))
    assert mm.run_worst_dd_n == pytest.approx(worst_dd(ticks, "pnl_naive"))


def test_run_wide_extremes_reset_with_reset():
    mm = LiveMarketMaker(seed=11, params=dict(eta=0.0, tox=0.5))
    run_ticks(mm, 2 * N)
    assert mm.run_max_abs_q > 0
    mm.reset()
    assert (mm.run_max_abs_q, mm.run_worst_dd, mm.epochs.n_completed) == (0, 0.0, 0)


# ------------------------------------------------------------------ B4: freeze/repro
def after_a_config_change(**flags):
    mm = LiveMarketMaker(seed=5, **flags)
    run_ticks(mm, 3 * N)
    mm.request("sigma", 3.0)
    run_ticks(mm, 4 * N)
    return mm


def test_epoch_remembers_its_first_session():
    mm = after_a_config_change()
    assert [e.start_session for e in mm.epochs.epochs] == [0, 3]
    assert mm.session == 7 and mm.session_step == 0     # sessions 3..6 done, at a boundary
    run_ticks(mm, 50)                                    # session 7 now in flight
    mm.request("tox", 0.2)                               # interrupts it
    run_ticks(mm, N)
    e1, e2 = mm.epochs.epochs[1], mm.epochs.epochs[2]
    assert e1.interrupted == [7] and e2.start_session == 8
    assert e2.sessions[0].index == 8                      # the interrupted index is skipped


def test_live_script_reproduces_the_epoch_exactly():
    mm = after_a_config_change()
    exec(live_script(mm.epochs.current), {})          # asserts internally


def test_batch_script_reproduces_the_same_epoch():
    mm = after_a_config_change()
    assert batch_possible(mm.epochs.current)
    exec(batch_script(mm.epochs.current), {})


@pytest.mark.parametrize("flags", [dict(auto_widen=True), dict(auto_eta=True),
                                   dict(auto_eta=True, auto_widen=True)])
def test_live_script_reproduces_live_only_strategies(flags):
    mm = after_a_config_change(**flags)
    e = mm.epochs.current
    assert not batch_possible(e)
    with pytest.raises(ValueError):
        batch_script(e)
    exec(live_script(e), {})


def test_script_fails_loudly_if_the_numbers_do_not_reproduce():
    mm = after_a_config_change()
    src = live_script(mm.epochs.current)
    key = "'diff_mean': "
    i = src.index(key) + len(key)
    tampered = src[:i] + "123.0 + " + src[i:]
    with pytest.raises(AssertionError):
        exec(tampered, {})


def test_report_for_an_epoch_without_sessions_and_for_a_normal_one():
    mm = LiveMarketMaker(seed=5)
    assert "FREEZE" in reproduction_report(mm.epochs.current)       # nothing completed yet
    run_ticks(mm, N)
    rep = reproduction_report(mm.epochs.current)
    assert "live form" in rep and "batch form" in rep and "expected" in rep


# ------------------------------------------------------------------ dashboard smoke
def click(fig, ax):
    """Click a widget the way a user does: through the canvas event path.

    Calling the handler directly tests the function but NOT the wiring, which is
    how the Reset/Freeze widgets could be garbage collected (callbacks dead, widget
    still drawn) while the old test passed.
    """
    x, y = ax.transAxes.transform((0.5, 0.5))
    for kind in ("button_press_event", "button_release_event"):
        MouseEvent(kind, fig.canvas, x, y, 1)._process()


def load_dashboard():
    matplotlib.use("Agg", force=True)
    path = Path(__file__).resolve().parent.parent / "scripts" / "live_dashboard.py"
    spec = importlib.util.spec_from_file_location("live_dashboard_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_widgets_survive_after_build_returns():
    """Regression: matplotlib widgets stay alive only via the caller's reference."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider
    dash = load_dashboard()
    refs = {}
    originals = {cls: cls.__init__ for cls in (Button, Slider, CheckButtons)}

    def patch(cls, name):
        orig = originals[cls]

        def init(self, *a, **k):
            orig(self, *a, **k)
            label = getattr(getattr(self, "label", None), "get_text", lambda: name)()
            refs[f"{name}:{label}"] = weakref.ref(self)
        return init
    for cls, name in ((Button, "Button"), (Slider, "Slider"), (CheckButtons, "Check")):
        cls.__init__ = patch(cls, name)
    try:
        d = dash.build_dashboard(seed=42)
    finally:
        for cls, orig in originals.items():
            cls.__init__ = orig
    gc.collect()
    dead = [k for k, r in refs.items() if r() is None]
    assert refs and not dead, f"garbage-collected widgets (callbacks are dead): {dead}"
    plt.close(d.fig)


@pytest.mark.parametrize("name", ["pause", "reset", "freeze"])
def test_every_button_fires_through_a_real_click(name, capsys):
    import matplotlib.pyplot as plt
    dash = load_dashboard()
    d = dash.build_dashboard(seed=42)
    for _ in range(120):
        d.update()
    before = d.status.get_text()
    click(d.fig, d.buttons[name].ax)
    after = d.status.get_text()
    assert after != before, f"{name} button did not fire"
    assert name.capitalize() in after and "FAILED" not in after
    plt.close(d.fig)


def test_click_pause_toggles_and_space_bar_works():
    import matplotlib.pyplot as plt
    dash = load_dashboard()
    d = dash.build_dashboard(seed=42)
    for _ in range(50):
        d.update()
    assert not d.mm.paused
    click(d.fig, d.buttons["pause"].ax)
    assert d.mm.paused and d.buttons["pause"].label.get_text() == "Resume"
    tick = d.mm.n
    d.update()
    assert d.mm.n == tick                       # paused really stops the simulation
    KeyEvent("key_press_event", d.fig.canvas, " ")._process()   # space bar resumes
    assert not d.mm.paused and d.buttons["pause"].label.get_text() == "Pause"
    d.update()
    assert d.mm.n > tick
    plt.close(d.fig)


def test_click_reset_clears_the_run():
    import matplotlib.pyplot as plt
    dash = load_dashboard()
    d = dash.build_dashboard(seed=42)
    for _ in range(300):                         # 600 ticks = 3 sessions
        d.update()
    assert d.mm.epochs.n_completed == 3
    click(d.fig, d.buttons["reset"].ax)
    assert d.mm.epochs.n_completed == 0 and d.mm.run_max_abs_q == 0
    assert "Reset" in d.status.get_text()
    plt.close(d.fig)


def test_a_failing_handler_is_reported_not_swallowed():
    import matplotlib.pyplot as plt
    dash = load_dashboard()
    d = dash.build_dashboard(seed=42)
    d.mm.epochs = None                           # make Freeze raise
    click(d.fig, d.buttons["freeze"].ax)
    assert "FAILED" in d.status.get_text()
    assert d.status.get_color() == "#b00020"
    plt.close(d.fig)


def test_dashboard_panels_scales_and_freeze(capsys):
    import matplotlib.pyplot as plt
    dash = load_dashboard()
    d = dash.build_dashboard(seed=42)
    for _ in range(450):                              # 900 ticks = 4.5 sessions
        d.update()
    ax = d.axes
    # B2: the three P&L panels have independent y-scales, each covering its own data
    panels = [ax["mtm"], ax["spread"], ax["inv_pnl"]]
    for a in panels:
        assert len(a.get_shared_y_axes().get_siblings(a)) == 1
    for a, (ln_n, ln_a) in zip(panels, [(d.lines["mtm_n"], d.lines["mtm_a"]),
                                        (d.lines["spr_n"], d.lines["spr_a"]),
                                        (d.lines["ipl_n"], d.lines["ipl_a"])]):
        y = np.concatenate([ln_n.get_ydata(), ln_a.get_ydata()])
        y = y[~np.isnan(y)]
        lo, hi = a.get_ylim()
        assert lo <= y.min() and y.max() <= hi
        assert (hi - lo) < 3 * (y.max() - y.min()) + 2        # not blown out by another series
    assert len({tuple(np.round(a.get_ylim(), 6)) for a in panels}) == 3
    # B1: both strategies drawn, paired chart tracks the epoch's completed sessions
    n_done = len(d.mm.epochs.current.sessions)
    assert n_done == 4 and len(d.lines["dmean"].get_xdata()) == n_done
    assert "aware − naive" in d.texts["headline"].get_text()   # the paired result, tier 1
    # B3: run-wide panel
    run = d.texts["run"].get_text()
    assert "RUN-WIDE" in run and "sessions completed 4" in run and "max |q| so far" in run
    # B4: a real click on Freeze pauses and prints a script that reproduces the epoch
    click(d.fig, d.buttons["freeze"].ax)
    out = capsys.readouterr().out
    assert d.mm.paused and "FREEZE: epoch #0" in out and "assert" in out
    live = out.split("# ---- live form ----")[1].split("# ---- batch form")[0]
    exec(live, {})
    plt.close(d.fig)


def test_dashboard_slider_change_opens_epoch_at_a_tick_boundary():
    import matplotlib.pyplot as plt
    dash = load_dashboard()
    d = dash.build_dashboard(seed=42)
    for _ in range(120):
        d.update()
    d.sliders[2].set_val(4.0)                         # sigma slider: only queues the change
    assert d.mm.params["sigma"] == 2.0 and d.mm._pending == {"sigma": 4.0}
    d.update()
    assert d.mm.params["sigma"] == 4.0 and len(d.mm.epochs.epochs) == 2
    plt.close(d.fig)
