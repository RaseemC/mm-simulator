"""A1 tests: parameter epochs and session structure of the live simulator."""
import numpy as np
import pytest

from mmsim.epochs import EpochLog, SessionRecord
from mmsim.live_sim import LiveMarketMaker

N = 200


def run_ticks(mm, n):
    return [mm.step() for _ in range(n)]


def test_no_change_single_epoch_pools_all_sessions():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, 3 * N)
    assert len(mm.epochs.epochs) == 1
    assert mm.epochs.current.stats()["n"] == 3
    assert mm.epochs.n_incomplete == 0


def test_change_at_session_boundary_opens_epoch_without_incomplete():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, 2 * N)                 # exactly two completed sessions
    mm.request("sigma", 4.0)
    run_ticks(mm, 2 * N)
    e0, e1 = mm.epochs.epochs
    assert (len(e0.sessions), len(e1.sessions)) == (2, 2)
    assert e0.interrupted == [] and e1.interrupted == []
    assert e0.config["sigma"] == 2.0 and e1.config["sigma"] == 4.0


def test_mid_session_change_discards_session_from_both_epochs():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, N + 50)                # session 0 done, session 1 at tick 50
    mm.request("delta", 1.0)
    run_ticks(mm, 1)                     # applied at the top of the next tick
    e0, e1 = mm.epochs.epochs
    assert [s.index for s in e0.sessions] == [0]
    assert e0.interrupted == [1]         # recorded against the epoch it ran under
    assert e1.sessions == [] and e1.interrupted == []
    assert mm.session == 2               # next session index, fresh state
    assert mm.session_step == 1
    run_ticks(mm, N - 1)
    assert [s.index for s in mm.epochs.current.sessions] == [2]
    assert 1 not in [s.index for e in mm.epochs.epochs for s in e.sessions]


def test_request_does_not_touch_params_until_a_tick_boundary():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, 10)
    mm.request("eta", 0.3)
    assert mm.params["eta"] == 0.10      # a GUI callback cannot change quotes mid-tick
    mm.step()
    assert mm.params["eta"] == 0.3


def test_changes_between_ticks_coalesce_and_revert_is_no_change():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, 20)
    mm.request("sigma", 5.0)
    mm.request("sigma", 2.0)             # dragged and put back before the next tick
    mm.step()
    assert len(mm.epochs.epochs) == 1 and mm.epochs.n_incomplete == 0
    assert mm.session == 0 and mm.session_step == 21


def test_two_changes_in_one_tick_open_one_epoch():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, 20)
    mm.request("sigma", 5.0)
    mm.request("A", 100.0)
    mm.step()
    assert len(mm.epochs.epochs) == 2
    assert mm.epochs.epochs[0].interrupted == [0]


def test_toggle_flags_are_config():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, N)
    mm.request("auto_widen", True)
    run_ticks(mm, N)
    assert [e.config["auto_widen"] for e in mm.epochs.epochs] == [False, True]


def test_auto_eta_adjusts_state_without_opening_epochs():
    mm = LiveMarketMaker(seed=3)
    mm.request("auto_eta", True)
    mm.step()
    n_epochs = len(mm.epochs.epochs)
    etas = set()
    for _ in range(4 * N):
        mm.step()
        etas.add(round(mm.eta_eff, 6))
    assert len(mm.epochs.epochs) == n_epochs      # eta_eff moving is not a config change
    assert len(etas) > 1                          # but it really did move
    assert mm.params["eta"] == 0.10               # user setting untouched


def test_epoch_stats_match_manual_and_are_not_pooled():
    mm = LiveMarketMaker(seed=5)
    run_ticks(mm, 6 * N)
    mm.request("sigma", 6.0)
    run_ticks(mm, 6 * N)
    e0, e1 = mm.epochs.epochs
    for e in (e0, e1):
        pnl = np.array([s.terminal_pnl for s in e.sessions])
        st = e.stats()
        assert st["n"] == 6
        assert st["mean"] == pytest.approx(pnl.mean())
        assert st["sd"] == pytest.approx(pnl.std(ddof=1))
        assert st["se"] == pytest.approx(pnl.std(ddof=1) / np.sqrt(6))
        assert st["sharpe"] == pytest.approx(pnl.mean() / pnl.std(ddof=1))
    pooled = np.mean([s.terminal_pnl for e in (e0, e1) for s in e.sessions])
    assert pooled != pytest.approx(e0.stats()["mean"])   # a pooled number would differ


def test_session_boundary_resets_inventory_and_cash():
    mm = LiveMarketMaker(seed=2)
    ticks = run_ticks(mm, 5 * N)
    completed = [t for t in ticks if t.completed]
    assert len(completed) == 5
    assert (mm.q, mm.X, mm.spread_pnl, mm.inventory_pnl) == (0, 0.0, 0.0, 0.0)
    assert mm.session_step == 0 and mm.m == 100.0


def test_session_i_depends_only_on_seed_and_index():
    """Per-session seeding contract: session 3 is the same whether or not 0-2 ran."""
    a = LiveMarketMaker(seed=7)
    run_ticks(a, 4 * N)
    b = LiveMarketMaker(seed=7)
    b.session = 3
    b._start_session()
    run_ticks(b, N)
    ra = [s for s in a.epochs.current.sessions if s.index == 3][0]
    rb = b.epochs.current.sessions[0]
    assert rb.index == 3 and rb.terminal_pnl == ra.terminal_pnl
    assert rb.spread_pnl == ra.spread_pnl


def test_determinism_with_identical_schedule():
    def go():
        mm = LiveMarketMaker(seed=9)
        run_ticks(mm, 2 * N + 37)
        mm.request("tox", 1.5)
        run_ticks(mm, 3 * N)
        return mm.epochs.table()
    assert go() == go()


def test_pnl_identity_holds_per_session():
    mm = LiveMarketMaker(seed=4)
    run_ticks(mm, 8 * N)
    for s in mm.epochs.current.sessions:
        assert s.terminal_pnl == pytest.approx(s.spread_pnl + s.inventory_pnl, abs=1e-9)


def test_table_hides_empty_epochs_and_shows_diffs():
    mm = LiveMarketMaker(seed=1)
    run_ticks(mm, N)
    for v in (2.5, 3.0, 3.5):            # slider drag at a session boundary -> empty epochs
        mm.request("sigma", v)
        assert mm.apply_pending()
    assert len(mm.epochs.epochs) == 4
    run_ticks(mm, N)
    t = mm.epochs.table()
    assert "sigma 2->3.5" in t and "aware mean" in t
    assert "(2 epoch(s) with no sessions at all omitted)" in t
    assert t.count("\n") == 4                                     # header, rule, 2 epochs, omitted note


def test_epochlog_unit():
    log = EpochLog(dict(a=1.0))
    rec = lambda i, pnl: SessionRecord(i, pnl, pnl, 0.0, 1, 0.0, 2, 0, 0, 0, 0)
    log.add_session(rec(0, 1.0))
    assert log.update_config(dict(a=1.0)) is False
    assert log.update_config(dict(a=2.0), interrupted=1) is True
    log.add_session(rec(2, 5.0))
    assert log.epochs[0].interrupted == [1] and log.epochs[1].interrupted == []
    assert log.n_completed == 2 and log.n_incomplete == 1
    assert log.epochs[0].stats()["mean"] == 1.0 and log.epochs[1].stats()["mean"] == 5.0
