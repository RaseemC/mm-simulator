"""B+3..B+8 UI work: queue model by default, tabs, ladder/heatmap/bubbles, slider
labels, n-labelling, the nan fix, keyboard shortcuts, and queue-model epochs.

Widgets are driven through the canvas event path (real clicks/keys), not by calling
handlers directly - a handler-level test once passed while the buttons were dead.
"""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

from mmsim import live_sim as ls
from mmsim.epochs import SessionRecord
from mmsim.repro import batch_script, live_script

matplotlib.use("Agg", force=True)


def load_dashboard():
    path = Path(__file__).resolve().parent.parent / "scripts" / "live_dashboard.py"
    spec = importlib.util.spec_from_file_location("live_dashboard_ui_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def click(fig, ax, xy=(0.5, 0.5)):
    x, y = ax.transAxes.transform(xy)
    for kind in ("button_press_event", "button_release_event"):
        MouseEvent(kind, fig.canvas, x, y, 1)._process()


def click_check(d, label_start):
    """Click a CheckButtons row by its label, through the canvas."""
    renderer = d.fig.canvas.get_renderer()
    for i, t in enumerate(d.chk.labels):
        if t.get_text().startswith(label_start):
            bb = t.get_window_extent(renderer)          # click the label, as a user would
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            for kind in ("button_press_event", "button_release_event"):
                MouseEvent(kind, d.fig.canvas, cx, cy, 1)._process()
            return i
    raise KeyError(label_start)


@pytest.fixture
def dash():
    import matplotlib.pyplot as plt
    mod = load_dashboard()
    yield mod
    plt.close("all")


# ------------------------------------------------------------ 1. queue by default
def test_queue_model_is_on_by_default_and_the_checkbox_switches_it(dash):
    d = dash.build_dashboard(seed=42)
    assert d.mm.queue_fills and d.chk.get_status()[2]
    for _ in range(20):
        d.update()
    assert d.mm.book_hist[-1].source == "queue"
    click_check(d, "Queue fills")
    assert d.mm.queue_fills                      # queued, not applied mid-frame
    d.update()
    assert not d.mm.queue_fills and len(d.mm.epochs.epochs) == 2
    assert d.mm.epochs.current.config["queue_fills"] is False
    assert d.mm.book_hist[-1].source == "synthetic"
    assert "Switch" in d.status.get_text()


def test_exponential_start_flag(dash):
    d = dash.build_dashboard(seed=42, queue_fills=False)
    d.update()
    assert not d.mm.queue_fills and d.mm.book_hist[-1].source == "synthetic"


# ------------------------------------------------------------ 2. the book views
def test_display_book_matches_the_fill_driving_book():
    mm = ls.LiveMarketMaker(seed=3, queue_fills=True)
    seen_ours = 0
    for _ in range(400):
        mm.step()
        db = mm.book_hist[-1]
        assert db.source == "queue" and db.tick == mm.queue_cfg.tick
        assert np.all(np.diff(db.bid_px) < 0) and np.all(np.diff(db.ask_px) > 0)
        assert db.bid_px[0] < db.mid < db.ask_px[0]
        for side in ("bid", "ask"):
            ahead, at = getattr(db, f"ahead_{side}"), getattr(db, f"depth_at_{side}")
            if np.isfinite(ahead) and np.isfinite(at):
                assert -1e-9 <= ahead <= at + 1e-9        # can't have more ahead than exists
                seen_ours += 1
    assert seen_ours > 100


def test_display_book_is_read_only_and_does_not_change_the_simulation():
    a = ls.LiveMarketMaker(seed=9, queue_fills=True)
    b = ls.LiveMarketMaker(seed=9, queue_fills=True)
    for _ in range(300):
        a.step()
        b.step()
        a.display_book()                                  # extra reads must change nothing
        a.display_book()
    assert a.pnl == b.pnl and a.q == b.q and a.pnl_n == b.pnl_n


def test_ladder_shows_our_queue_position_text(dash):
    d = dash.build_dashboard(seed=42)
    click(d.fig, d.tabs["Book"].ax)
    for _ in range(80):
        d.update()
    texts = [t.get_text() for t in d.axes["ladder"].texts]
    assert any(t.startswith("◀ ours:") or t.endswith("ahead ▶") for t in texts)
    assert any("ours:" in t and " of " in t and "ahead" in t for t in texts)
    assert d.artists["heat"].get_array().shape[1] > 1      # heatmap has time columns


def test_bubble_debug_toggle_is_display_only(dash):
    d = dash.build_dashboard(seed=42)
    click(d.fig, d.tabs["Book"].ax)
    for _ in range(60):
        d.update()
    epochs_before, pnl_before = len(d.mm.epochs.epochs), d.mm.pnl
    cfg_before = dict(d.mm.config())
    click_check(d, "DEBUG")
    assert d.ui["truth_debug"]
    d.update()
    assert len(d.mm.epochs.epochs) == epochs_before        # never a config change
    assert d.mm.config() == cfg_before
    assert "DEBUG" in d.axes["bubble"].get_title(loc="right")
    vals = np.asarray(d.artists["bubble"].get_array())
    assert set(np.unique(vals)) <= {0.0, 1.0}               # truth is binary
    del pnl_before


# ------------------------------------------------------------ 3. tabs
def test_tab_clicks_change_visibility(dash):
    d = dash.build_dashboard(seed=42)
    for name in dash.TABS:
        click(d.fig, d.tabs[name].ax)
        assert d.ui["tab"] == name
        for tab, members in d.tab_members.items():
            assert all(m.get_visible() == (tab == name) for m in members)
        # controls are visible on every tab
        assert all(s.ax.get_visible() for s in d.sliders)
        assert d.buttons["freeze"].ax.get_visible() and d.chk.ax.get_visible()


def test_tabs_change_what_is_drawn_never_what_is_computed(dash):
    a = dash.build_dashboard(seed=7)
    b = dash.build_dashboard(seed=7)
    tabs = list(dash.TABS)
    for i in range(240):
        if i % 30 == 0:
            b.show_tab(tabs[(i // 30) % len(tabs)])
        a.update()
        b.update()
    assert a.mm.n == b.mm.n and a.mm.pnl == b.mm.pnl and a.mm.pnl_n == b.mm.pnl_n
    assert list(a.mm.hist["q"]) == list(b.mm.hist["q"])
    assert a.mm.epochs.table() == b.mm.epochs.table()


# ------------------------------------------------------------ 5. slider labels
def test_every_slider_has_a_plain_language_caption_and_shows_its_default(dash):
    d = dash.build_dashboard(seed=42)
    d.update()
    assert len(d.sliders) == len(dash.SLIDER_SPECS)
    for s, (key, sym, meaning, *_rest) in zip(d.sliders, dash.SLIDER_SPECS):
        default = _rest[2]
        assert s.label.get_text() == sym
        assert d.captions[key].get_text().startswith(meaning)
        assert f"def {default:.3g}" in s.valtext.get_text()
    keys = [spec[0] for spec in dash.SLIDER_SPECS]
    for q in ls.QUEUE_KEYS:
        assert q in keys                                     # queue params are exposed too


def test_red_tick_marks_the_epoch_start_value(dash):
    d = dash.build_dashboard(seed=42)
    d.update()
    sigma = d.sliders[2]
    assert sigma.vline.get_xdata()[0] == pytest.approx(2.0)
    sigma.set_val(4.0)
    assert sigma.vline.get_xdata()[0] == pytest.approx(2.0)   # still the OLD epoch
    d.update()
    assert d.mm.epochs.current.config["sigma"] == 4.0
    assert sigma.vline.get_xdata()[0] == pytest.approx(4.0)   # moved with the new epoch


def test_queue_sliders_are_greyed_when_the_queue_model_is_off(dash):
    d = dash.build_dashboard(seed=42, queue_fills=False)
    d.update()
    assert "unused" in d.captions["depth_base"].get_text()
    assert "unused" not in d.captions["eta"].get_text()
    assert d.captions["depth_base"].get_color() == dash.MUTED


def test_queue_slider_change_opens_an_epoch_and_reaches_the_book(dash):
    d = dash.build_dashboard(seed=42)
    for _ in range(10):
        d.update()
    keys = [spec[0] for spec in dash.SLIDER_SPECS]
    d.sliders[keys.index("depth_base")].set_val(15.0)
    d.update()
    assert d.mm.queue_cfg.depth_base == 15.0
    assert d.mm.epochs.current.config["depth_base"] == 15.0
    assert len(d.mm.epochs.epochs) == 2


# ------------------------------------------------------------ 6. n, greying, keys
def fake_sessions(mm, n):
    for i in range(n):
        mm.epochs.current.sessions.append(SessionRecord(
            i, 10.0 + i % 3, 9.0, 1.0, 2.0, 1.0, 50.0, 10, 7, 190, 3,
            naive_terminal_pnl=9.0 + i % 2))


def test_thin_epochs_raise_the_banner_and_thick_ones_do_not(dash):
    """One under-powered banner carries the warning; no per-number asterisks anywhere."""
    d = dash.build_dashboard(seed=42)
    d.show_tab("Stats")
    fake_sessions(d.mm, 12)
    d.update()
    assert d.texts["banner"].get_visible()
    assert "n = 12" in d.texts["banner"].get_text()
    assert d.texts["headline"].get_color() == dash.WARN
    assert "n = 12" in d.texts["epoch"].get_text()
    assert "fewer than 30" in d.texts["footnote"].get_text()
    fake_sessions(d.mm, 25)                                     # now 37
    d.update()
    assert not d.texts["banner"].get_visible()
    assert d.texts["headline"].get_color() == dash.INK
    assert "n = 37" in d.texts["epoch"].get_text()
    for t in (d.texts["headline"], d.texts["epoch"], *d.texts["history"]):
        assert "*" not in t.get_text()                          # asterisks dropped entirely


def test_the_banner_is_hidden_on_other_tabs_even_when_under_powered(dash):
    d = dash.build_dashboard(seed=42)
    fake_sessions(d.mm, 5)
    d.show_tab("Stats")
    d.update()
    assert d.texts["banner"].get_visible()
    d.show_tab("Market")
    d.update()
    assert not d.texts["banner"].get_visible()


def test_config_block_is_grouped_rounded_and_marks_changes(dash):
    d = dash.build_dashboard(seed=42)
    d.show_tab("Stats")
    d.update()
    text = "\n".join(t.get_text() for t in d.texts["config"])
    for group in ("MARKET", "STRATEGY", "QUEUE MODEL", "RUN"):
        assert group in text
    assert "•" not in text                                      # nothing changed yet
    keys = [spec[0] for spec in dash.SLIDER_SPECS]
    d.sliders[keys.index("eta")].set_val(0.124074)
    d.update()
    text = "\n".join(t.get_text() for t in d.texts["config"])
    assert "0.124" in text and "0.124074" not in text           # rounded to 3 dp
    assert "•" in text                                          # and marked as changed


def test_no_nan_ever_reaches_the_screen(dash):
    """Zero sessions, zero informed ticks: every undefined statistic is a dash."""
    d = dash.build_dashboard(seed=42)
    d.update()
    for t in [d.texts["epoch"], d.texts["run"], d.texts["session"], *d.texts["history"]]:
        assert "nan" not in t.get_text().lower()
    assert "—" in d.texts["epoch"].get_text()
    fake_sessions(d.mm, 1)                                      # n=1: se/Sharpe undefined
    for r in list(d.mm.epochs.current.sessions):
        object.__setattr__(r, "n_active", 0)                    # no informed ticks at all
    d.update()
    assert "nan" not in d.texts["epoch"].get_text().lower()
    assert "detect —" in d.texts["epoch"].get_text()


def test_p_and_space_pause_and_p_is_not_pan(dash):
    import matplotlib.pyplot as plt
    d = dash.build_dashboard(seed=42)
    assert "p" not in plt.rcParams["keymap.pan"]
    d.update()
    KeyEvent("key_press_event", d.fig.canvas, "p")._process()
    assert d.mm.paused
    KeyEvent("key_press_event", d.fig.canvas, " ")._process()
    assert not d.mm.paused


# ------------------------------------------------------------ Freeze under the queue model
@pytest.mark.parametrize("change", [("sigma", 3.0), ("depth_base", 6.0), ("mo_rate", 0.3)])
def test_freeze_scripts_reproduce_queue_model_epochs(change):
    import contextlib
    import io
    mm = ls.LiveMarketMaker(seed=5, queue_fills=True, params=dict(tox=0.3))
    for _ in range(2 * 200):
        mm.step()
    mm.request(*change)
    for _ in range(3 * 200):
        mm.step()
    e = mm.epochs.current
    assert e.config[change[0]] == change[1]
    for script in (live_script(e), batch_script(e)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(script, {})
        assert "OK: epoch reproduced" in out.getvalue()
