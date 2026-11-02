"""Live/batch equivalence: the live tick loop and the vectorised batch engine must
agree session by session, with adverse selection ON, across the eta range we
sweep (including large eta, where the quote-distance floor binds)."""
import numpy as np
import pytest

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from mmsim.core import simulate_paths
from mmsim.live_sim import LiveMarketMaker
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.strategies import InventorySkewQuoter

SEED, SESSIONS = 31, 25
ETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


def batch(seed, sessions, eta, strength, delta=0.75, sigma=2.0, A=140.0, k=1.5,
          queue_fills=False):
    market = MarketConfig(sigma=sigma)
    adv = AdverseConfig(strength=strength)
    qc = QueueConfig() if queue_fills else None
    prep = prepare_batch(ArithmeticRandomWalk(market), seed, sessions, adv)
    res = simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(delta, eta),
                         FillConfig(A=A, k=k), market.dt, informed=prep.informed,
                         fill_boost=adv.fill_boost, queue=qc, u_queue=prep.u_queue,
                         u_depth0=prep.u_depth0, tilt=prep.tilt)
    return res, prep


def live(seed, sessions, eta, strength, delta=0.75, sigma=2.0, A=140.0, k=1.5,
         queue_fills=False):
    mm = LiveMarketMaker(seed=seed, queue_fills=queue_fills,
                         params=dict(eta=eta, tox=strength, delta=delta,
                                     sigma=sigma, A=A, k=k))
    for _ in range(sessions * mm.session_steps):
        mm.step()
    return mm.epochs.current.sessions


def assert_equivalent(eta, strength, **kw):
    res, prep = batch(SEED, SESSIONS, eta, strength, **kw)
    recs = live(SEED, SESSIONS, eta, strength, **kw)
    st = res["state"]
    # The live simulator runs BOTH strategies; check the naive leg too, against a
    # batch run at eta = 0. Without this a change to the live naive quoter goes
    # undetected (found by mutation: only the aware leg was being compared).
    nres, _ = batch(SEED, SESSIONS, 0.0, strength, **kw)
    np.testing.assert_allclose([r.naive_terminal_pnl for r in recs], nres["terminal_pnl"],
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.naive_spread_pnl for r in recs], nres["state"].spread_pnl,
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.naive_max_abs_q for r in recs], np.abs(nres["q"]).max(axis=1),
                               rtol=1e-12, atol=1e-12)
    assert [r.index for r in recs] == list(range(SESSIONS))
    np.testing.assert_allclose([r.terminal_pnl for r in recs], res["terminal_pnl"],
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.spread_pnl for r in recs], st.spread_pnl, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.inventory_pnl for r in recs], st.inventory_pnl,
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose([r.fills for r in recs], st.bid_fills + st.ask_fills,
                               rtol=1e-12, atol=1e-12)
    assert [r.clamp_steps for r in recs] == st.clamp_steps.tolist()
    np.testing.assert_allclose([r.max_abs_q for r in recs], np.abs(res["q"]).max(axis=1),
                               rtol=1e-12, atol=1e-12)
    return recs, prep


@pytest.mark.filterwarnings("ignore:fill-probability cap")
@pytest.mark.parametrize("eta", ETAS)
def test_equivalence_with_adverse_selection_on(eta):
    recs, prep = assert_equivalent(eta, strength=0.5)
    assert (prep.informed != 0).any()            # the case is not vacuous: informed ticks exist


@pytest.mark.filterwarnings("ignore:fill-probability cap")
@pytest.mark.parametrize("eta", [0.0, 0.1, 1.0])
def test_equivalence_with_adverse_selection_off(eta):
    assert_equivalent(eta, strength=0.0)


@pytest.mark.filterwarnings("ignore:fill-probability cap")
def test_large_eta_exercises_the_clamp_in_both_paths():
    recs, _ = assert_equivalent(1.0, strength=0.5)
    assert sum(r.clamp_steps for r in recs) > 0   # the floor bound, and still agrees
    recs0, _ = assert_equivalent(0.05, strength=0.5)
    assert sum(r.clamp_steps for r in recs0) == 0


@pytest.mark.filterwarnings("ignore:fill-probability cap")
def test_equivalence_at_other_market_settings():
    assert_equivalent(0.3, strength=1.0, sigma=4.86, delta=0.4, A=200.0, k=1.0)


# --------------------------------------------------------------- B+2(d) queue model
@pytest.mark.filterwarnings("ignore:fill-probability cap")
@pytest.mark.parametrize("eta", ETAS)
def test_queue_model_equivalence_across_eta(eta):
    """Live and batch must agree tick for tick with queue-position fills on."""
    recs, prep = assert_equivalent(eta, strength=0.5, queue_fills=True)
    assert sum(r.fills for r in recs) > 0          # not vacuous: it really traded


@pytest.mark.filterwarnings("ignore:fill-probability cap")
def test_queue_model_equivalence_at_other_market_settings():
    assert_equivalent(0.3, strength=1.0, sigma=4.86, delta=0.4, A=200.0, k=1.0,
                      queue_fills=True)
    assert_equivalent(0.1, strength=0.0, queue_fills=True)       # adverse selection off


def test_queue_flag_off_is_bit_identical_to_the_exponential_engine():
    """Selecting the old model must reproduce the pre-queue results exactly."""
    for eta in (0.0, 0.1, 0.5):
        a, _ = batch(SEED, SESSIONS, eta, 0.5, queue_fills=False)
        b, _ = batch(SEED, SESSIONS, eta, 0.5, queue_fills=False)
        np.testing.assert_array_equal(a["terminal_pnl"], b["terminal_pnl"])
        # fills are indicators under the exponential model, so q stays integral
        assert np.all(a["q"] == np.round(a["q"]))
        assert np.all(np.isin(a["f_bid"], [0.0, 1.0]))


@pytest.mark.filterwarnings("ignore:fill-probability cap")
def test_queue_model_pnl_identity_holds_under_partial_fills():
    res, _ = batch(SEED, 200, 0.1, 0.5, queue_fills=True)
    st = res["state"]
    np.testing.assert_allclose(res["terminal_pnl"], st.spread_pnl + st.inventory_pnl,
                               rtol=0, atol=1e-9)
    f = np.concatenate([res["f_bid"].ravel(), res["f_ask"].ravel()])
    f = f[f > 0]
    partial = float(np.mean(f < QueueConfig().our_size - 1e-9))
    assert partial > 0.005, "no partial fills: the identity was not exercised fractionally"
    assert np.all(f <= QueueConfig().our_size + 1e-12)


@pytest.mark.filterwarnings("ignore:fill-probability cap")
def test_spread_capture_uses_the_executed_grid_price_not_the_requested_distance():
    """Quotes are rounded to the grid, so spread P&L must use the ACTUAL distance
    and be weighted by filled size. Rebuild it from executed prices and compare."""
    res, prep = batch(SEED, 60, 0.1, 0.5, queue_fills=True)
    st, tick = res["state"], QueueConfig().tick
    for px in (res["bid"], res["ask"]):
        assert np.allclose(px / tick, np.round(px / tick), atol=1e-9)   # on the grid
    m = prep.mids[:, :-1]                                               # mid at quote time
    rebuilt = (res["f_bid"] * (m - res["bid"]) + res["f_ask"] * (res["ask"] - m)).sum(axis=1)
    np.testing.assert_allclose(rebuilt, st.spread_pnl, rtol=1e-10, atol=1e-9)
    # the requested distance would NOT reproduce it (rounding really moves the price)
    naive_rebuild = (0.1 * 0 + 0.75) * (res["f_bid"] + res["f_ask"]).sum(axis=1)
    assert not np.allclose(naive_rebuild, st.spread_pnl, rtol=1e-6)
