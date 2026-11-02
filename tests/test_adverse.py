"""Adverse selection: episode state machine, drift scaling, fill boost, seeding contract."""
import numpy as np
import pytest

from mmsim.adverse import build_regime, informed_drift, prepare, prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig
from mmsim.core import initial_state, simulate_paths, step
from mmsim.draws import draw_batch, draw_session, session_rng
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.strategies import InventorySkewQuoter

MKT, FILLS = MarketConfig(), FillConfig()
CFG = AdverseConfig(strength=0.2, lead=10)
N = MKT.N


def regime_from(u_start, u_dir, u_dur, cfg=CFG):
    return build_regime(u_start[None, :], u_dir[None, :], u_dur[None, :], cfg)


def test_episode_lifecycle_exact():
    u_start = np.ones(N); u_start[3] = 0.0           # one episode starts at tick 3
    u_dir = np.full(N, 0.1)                           # < 0.5 -> direction +1
    u_dur = np.zeros(N)                               # duration 1 + floor(0) = 1 tick
    r = regime_from(u_start, u_dir, u_dur)
    tilt, informed = r.tilt[0], r.informed[0]
    assert np.flatnonzero(tilt).tolist() == list(range(3, 14))   # 10 lead ticks + 1 active
    assert np.flatnonzero(informed).tolist() == [13]              # active only after the lead
    assert set(tilt[tilt != 0]) == {1} and set(informed[informed != 0]) == {1}


def test_direction_down_and_long_duration():
    u_start = np.ones(N); u_start[0] = 0.0
    u_dir = np.full(N, 0.9)                           # >= 0.5 -> direction -1
    u_dur = np.full(N, 1 - np.exp(-1.0))              # duration 1 + floor(mean*1) = 61
    r = regime_from(u_start, u_dir, u_dur)
    act = np.flatnonzero(r.informed[0])
    assert act[0] == 10 and len(act) == 61 and set(r.informed[0][act]) == {-1}


def test_strength_zero_disables_everything():
    d = draw_batch(ArithmeticRandomWalk(MKT), 1, 50)
    r = build_regime(d.u_start, d.u_dir, d.u_dur, AdverseConfig(strength=0.0))
    assert not r.informed.any() and not r.tilt.any()


def test_informed_is_subset_of_tilt_and_episodes_do_not_overlap():
    d = draw_batch(ArithmeticRandomWalk(MKT), 2, 400)
    r = build_regime(d.u_start, d.u_dir, d.u_dur, CFG)
    assert r.informed.any()
    assert np.all((r.informed == 0) | (r.informed == r.tilt))   # same direction, subset
    # direction never flips without passing through quiet
    flips = (r.tilt[:, 1:] * r.tilt[:, :-1]) < 0
    assert not flips.any()


def test_episode_frequency_and_duration():
    d = draw_batch(ArithmeticRandomWalk(MarketConfig(N=100_000)), 3, 1)
    cfg = AdverseConfig(strength=0.1, p_episode=0.004, lead=10, mean_duration=60.0)
    r = build_regime(d.u_start, d.u_dir, d.u_dur, cfg)
    a = (r.informed[0] != 0).astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, a]) == 1)
    lens = np.diff(np.flatnonzero(np.diff(np.r_[0, a, 0]) != 0))[::2]
    assert lens.mean() == pytest.approx(60.5, rel=0.15)           # 1 + floor(Exp(60))
    # one episode per (1/p + lead + duration) ticks on average
    assert len(starts) == pytest.approx(100_000 / (250 + 10 + 60.5), rel=0.15)


@pytest.mark.parametrize("sigma", [0.5, 2.0, 7.0])
def test_drift_is_relative_to_diffusion_scale(sigma):
    mk = MarketConfig(sigma=sigma)
    inf = np.array([1, 0, -1])
    drift = informed_drift(inf, AdverseConfig(strength=0.3), mk)
    np.testing.assert_allclose(drift / (sigma * np.sqrt(mk.dt)), 0.3 * inf)


def test_path_with_drift_matches_stream_exactly_and_none_is_unchanged():
    proc = ArithmeticRandomWalk(MKT)
    d = draw_session(proc, 11, 4)
    drift = np.full(N, 0.01)
    np.testing.assert_array_equal(np.array(list(proc.stream(d.shocks, drift))),
                                  proc.path_from_shocks(d.shocks, drift))
    np.testing.assert_array_equal(proc.path_from_shocks(d.shocks),
                                  proc.path_from_shocks(d.shocks, np.zeros(N)))


def test_core_draws_unchanged_by_the_new_block():
    """Z and U must equal the pre-adverse stream: the episode block is drawn LAST."""
    proc = ArithmeticRandomWalk(MKT)
    d = draw_session(proc, 9, 3)
    rng = session_rng(9, 3)
    np.testing.assert_array_equal(d.shocks, rng.standard_normal(N))
    u = rng.random((N, 2))
    np.testing.assert_array_equal(d.u_bid, u[:, 0])
    np.testing.assert_array_equal(d.u_ask, u[:, 1])
    v = rng.random((N, 3))
    np.testing.assert_array_equal(d.u_start, v[:, 0])


def test_regime_independent_of_config_draws_and_batch_size():
    proc = ArithmeticRandomWalk(MKT)
    a = prepare_batch(proc, 5, 3, CFG)
    b = prepare_batch(proc, 5, 10, CFG)
    np.testing.assert_array_equal(a.informed, b.informed[:3])
    np.testing.assert_array_equal(a.mids, b.mids[:3])
    c = prepare_batch(proc, 5, 3, AdverseConfig(strength=0.6, lead=10))   # same draws, new strength
    np.testing.assert_array_equal(a.informed, c.informed)


def test_fill_boost_hits_only_the_picked_off_side():
    quoter, st = InventorySkewQuoter(0.75, 0.0), initial_state(100.0, 3)
    u = np.array([0.4, 0.4, 0.4])                    # base p = 0.2268, boosted 0.567
    informed = np.array([-1, 0, +1])
    new, info = step(st, np.full(3, 100.0), u, u, quoter, FILLS, MKT.dt, 0.0,
                     informed=informed, fill_boost=2.5)
    assert info.f_bid.tolist() == [1, 0, 0]          # informed down: bid is hit
    assert info.f_ask.tolist() == [0, 0, 1]          # informed up: ask is lifted
    new0, info0 = step(st, np.full(3, 100.0), u, u, quoter, FILLS, MKT.dt, 0.0)
    assert info0.f_bid.sum() == 0 and info0.f_ask.sum() == 0     # no informed -> unchanged


def test_step_with_informed_is_pure():
    st = initial_state(100.0, 2)
    snap = {k: v.copy() for k, v in st.__dict__.items() if v is not None}
    inf = np.array([1, -1], dtype=np.int8)
    step(st, np.array([100.1, 99.9]), np.zeros(2), np.zeros(2),
         InventorySkewQuoter(0.75, 0.1), FILLS, MKT.dt, 0.0, informed=inf, fill_boost=2.5)
    for k, v in snap.items():
        np.testing.assert_array_equal(getattr(st, k), v)
    assert inf.tolist() == [1, -1]


def test_strength_zero_reproduces_the_non_adverse_batch_exactly():
    proc = ArithmeticRandomWalk(MKT)
    prep = prepare_batch(proc, 4, 30, AdverseConfig(strength=0.0))
    quoter = InventorySkewQuoter(0.75, 0.1)
    with_adv = simulate_paths(prep.mids, prep.u_bid, prep.u_ask, quoter, FILLS, MKT.dt,
                              informed=prep.informed, fill_boost=2.5)
    d = draw_batch(proc, 4, 30)
    plain = simulate_paths(proc.path_from_shocks(d.shocks), d.u_bid, d.u_ask, quoter, FILLS, MKT.dt)
    np.testing.assert_array_equal(with_adv["terminal_pnl"], plain["terminal_pnl"])
    np.testing.assert_array_equal(with_adv["q"], plain["q"])


def test_adverse_selection_costs_the_market_maker():
    """Informed flow must lower inventory P&L (the economic point of the model)."""
    proc = ArithmeticRandomWalk(MKT)
    quoter = InventorySkewQuoter(0.75, 0.0)
    out = {}
    for s in (0.0, 0.5):
        cfg = AdverseConfig(strength=s)
        prep = prepare_batch(proc, 21, 3000, cfg)
        r = simulate_paths(prep.mids, prep.u_bid, prep.u_ask, quoter, FILLS, MKT.dt,
                           informed=prep.informed, fill_boost=cfg.fill_boost)
        out[s] = r["state"].inventory_pnl
    se = out[0.5].std() / np.sqrt(out[0.5].size)
    assert out[0.5].mean() < out[0.0].mean() - 5 * se


def test_config_validation():
    for bad in (lambda: AdverseConfig(strength=-1), lambda: AdverseConfig(p_episode=2),
                lambda: AdverseConfig(lead=0), lambda: AdverseConfig(mean_duration=0),
                lambda: AdverseConfig(fill_boost=0.5)):
        with pytest.raises(ValueError):
            bad()
