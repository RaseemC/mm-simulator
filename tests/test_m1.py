"""M1 tests: core accounting, seeding, midprice, fills, strategies."""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from mmsim.config import FillConfig, MarketConfig, SimConfig, StrategyConfig
from mmsim.core import initial_state, simulate_paths, step
from mmsim.draws import draw_batch, draw_session
from mmsim.fills import fill_prob
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.strategies import InventorySkewQuoter

CFG = SimConfig()
MKT, FILLS = CFG.market, CFG.fills


def run(eta, n=200, seed=1, market=MKT, fills=FILLS, delta=0.75, penalty=0.0):
    proc = ArithmeticRandomWalk(market)
    d = draw_batch(proc, seed, n)
    mids = proc.path_from_shocks(d.shocks)
    return simulate_paths(mids, d.u_bid, d.u_ask, InventorySkewQuoter(delta, eta),
                          fills, market.dt, penalty), mids


@pytest.mark.filterwarnings("ignore:fill-probability cap")
@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5])
def test_decomposition_identity(eta):
    out, _ = run(eta)
    s = out["state"]
    np.testing.assert_allclose(out["terminal_pnl"], s.spread_pnl + s.inventory_pnl, atol=1e-9)


UNCLAMPED = FillConfig(min_dist=-np.inf)


@pytest.mark.filterwarnings("ignore:fill-probability cap")
def test_decomposition_with_negative_distances():
    out, mids = run(eta=1.5, fills=UNCLAMPED)  # aggressive skew, floor disabled
    s = out["state"]
    assert np.any(out["ask"] < mids[:, :-1])   # ask really did sit below the mid
    assert s.clamp_steps.sum() == 0
    np.testing.assert_allclose(s.mtm, s.spread_pnl + s.inventory_pnl, atol=1e-9)


def test_min_dist_floor_binds_and_identity_holds():
    out, mids = run(eta=1.5)                   # default floor 0.02
    s = out["state"]
    assert s.clamp_steps.sum() > 0
    assert np.all(out["ask"] - mids[:, :-1] >= FILLS.min_dist - 1e-12)
    assert np.all(mids[:, :-1] - out["bid"] >= FILLS.min_dist - 1e-12)
    np.testing.assert_allclose(s.mtm, s.spread_pnl + s.inventory_pnl, atol=1e-9)


def test_liquidation_penalty():
    out, _ = run(0.1, penalty=0.5)
    s = out["state"]
    np.testing.assert_allclose(out["terminal_pnl"], s.mtm - 0.5 * np.abs(s.q))


def test_eta_zero_symmetric_and_flat_inventory():
    q = InventorySkewQuoter(0.75, 0.0)
    db, da = q.distances(np.array([100.0, 99.0]), np.array([5.0, -3.0]), 0.0)
    assert np.all(db == da) and np.all(db == 0.75)
    out, _ = run(0.0, n=2000)
    q = out["state"].q
    assert abs(q.mean()) < 4 * q.std() / np.sqrt(q.size)  # mean inventory is 0 up to noise
    assert 7 < q.std() < 12  # var(q_N) ~ 2*N*p, p = A e^{-k delta} dt  ->  sd ~ 9.5


def test_aware_quote_formula():
    q = InventorySkewQuoter(0.75, 0.1)
    db, da = q.distances(np.array([100.0]), np.array([4.0]), 0.0)
    assert db[0] == pytest.approx(0.75 + 0.4) and da[0] == pytest.approx(0.75 - 0.4)


def test_zero_volatility_no_inventory_pnl():
    out, _ = run(0.1, market=replace(MKT, sigma=0.0))
    assert np.all(out["state"].inventory_pnl == 0.0)


def test_zero_A_nothing_happens():
    out, _ = run(0.1, fills=FillConfig(A=0.0, k=1.5))
    assert np.all(out["terminal_pnl"] == 0) and np.all(out["q"] == 0)


def test_idempotence():
    a, _ = run(0.1, seed=7)
    b, _ = run(0.1, seed=7)
    np.testing.assert_array_equal(a["terminal_pnl"], b["terminal_pnl"])
    np.testing.assert_array_equal(a["q"], b["q"])


def test_common_random_numbers():
    proc = ArithmeticRandomWalk(MKT)
    d = draw_batch(proc, 3, 20)
    mids = proc.path_from_shocks(d.shocks)
    before = (mids.copy(), d.u_bid.copy(), d.u_ask.copy())
    for eta in (0.0, 0.1):
        simulate_paths(mids, d.u_bid, d.u_ask, InventorySkewQuoter(0.75, eta), FILLS, MKT.dt)
    # engine never mutates the shared inputs, so both strategies saw identical data
    for x, y in zip(before, (mids, d.u_bid, d.u_ask)):
        np.testing.assert_array_equal(x, y)


def test_session_seed_independent_of_batch_size():
    proc = ArithmeticRandomWalk(MKT)
    small, big = draw_batch(proc, 9, 3), draw_batch(proc, 9, 10)
    np.testing.assert_array_equal(small.shocks, big.shocks[:3])
    np.testing.assert_array_equal(small.u_ask, big.u_ask[:3])


def test_spawn_key_matches_spawn():
    kids = np.random.SeedSequence(5).spawn(4)
    for i, k in enumerate(kids):
        a = np.random.Generator(np.random.PCG64(k)).random(5)
        b = np.random.Generator(np.random.PCG64(np.random.SeedSequence(5, spawn_key=(i,)))).random(5)
        np.testing.assert_array_equal(a, b)


def test_generator_matches_batch_path_exactly():
    proc = ArithmeticRandomWalk(MKT)
    d = draw_session(proc, 11, 4)
    np.testing.assert_array_equal(np.array(list(proc.stream(d.shocks))),
                                  proc.path_from_shocks(d.shocks))


def test_step_is_pure():
    st = initial_state(100.0, 3)
    snap = {k: v.copy() for k, v in st.__dict__.items() if v is not None}
    step(st, np.array([100.1] * 3), np.zeros(3), np.zeros(3),
         InventorySkewQuoter(0.75, 0.1), FILLS, MKT.dt, 0.0)
    for k, v in snap.items():
        np.testing.assert_array_equal(getattr(st, k), v)


def test_midprice_increments_moments():
    proc = ArithmeticRandomWalk(MKT)
    d = draw_batch(proc, 2, 2000)
    inc = np.diff(proc.path_from_shocks(d.shocks), axis=1).ravel()
    assert abs(inc.mean()) < 4 * inc.std() / np.sqrt(inc.size)
    assert inc.var() == pytest.approx(MKT.sigma**2 * MKT.dt, rel=0.02)


def test_empirical_fill_frequency():
    rng = np.random.default_rng(0)
    d = 0.75
    p = float(fill_prob(np.array(d), FILLS, MKT.dt))
    assert p == pytest.approx(140 * np.exp(-1.5 * d) * 0.005)
    n = 400_000
    freq = (rng.random(n) < p).mean()
    assert freq == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / n))


def test_fill_prob_capped_at_one():
    assert fill_prob(np.array([-5.0]), FILLS, MKT.dt)[0] == 1.0


def test_cap_warning():
    with pytest.warns(UserWarning, match="cap"):
        run(eta=3.0, n=20, fills=UNCLAMPED)


def test_no_warning_at_defaults():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(0.1, n=50)


@pytest.mark.parametrize("bad", [
    lambda: MarketConfig(sigma=-1), lambda: MarketConfig(T=0), lambda: MarketConfig(N=0),
    lambda: FillConfig(A=-1), lambda: FillConfig(k=-1), lambda: StrategyConfig(delta=-1),
    lambda: StrategyConfig(eta=-0.1),
    lambda: SimConfig(market=MarketConfig(N=10)),  # A*dt = 14 > 1
    lambda: SimConfig(n_sessions=0), lambda: SimConfig(liquidation_penalty=-1),
])
def test_validation(bad):
    with pytest.raises(ValueError):
        bad()
