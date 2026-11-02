"""B+2(a): the order book / queue-position state model, before it drives fills."""
import numpy as np
import pytest

from mmsim import book as bk
from mmsim.config import QueueConfig
from mmsim.draws import MAX_LEVELS, N_QUEUE_COLS, draw_batch, draw_session
from mmsim.config import MarketConfig
from mmsim.midprice import ArithmeticRandomWalk

CFG = QueueConfig()
FLAT = QueueConfig(depth_cv=0.0)          # deterministic depth: exact arithmetic in tests
QUIET = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, replenish_rate=0.0,
                    mo_rate=0.0, tilt=0.0)


def fresh(cfg=FLAT, n=1, m0=100.0):
    u0 = np.full((n, 2, MAX_LEVELS), 0.5)
    return bk.initial_state(np.full(n, m0), u0, cfg)


def place_at(state, bid_g, ask_g, cfg=FLAT):
    return bk.place(state, np.array([[bid_g, ask_g]] * state.n, dtype=np.int64), cfg)


# ---------------------------------------------------------------- geometry
def test_levels_are_absolute_prices_around_the_anchor():
    st = fresh(m0=100.13)
    assert st.anchor[0] == 400                      # floor(100.13 / 0.25)
    pb, pa = bk.level_prices(st.anchor, FLAT)
    assert pb[0, 0] == pytest.approx(100.0) and pa[0, 0] == pytest.approx(100.25)
    assert pb[0, 1] == pytest.approx(99.75) and pa[0, 1] == pytest.approx(100.5)


def test_quote_grids_never_quote_closer_than_asked():
    m = np.array([100.13, 99.9])
    d = np.array([0.75, 0.30])
    g = bk.quote_grids(d, d, m, FLAT)
    bid_px, ask_px = g[:, 0] * FLAT.tick, g[:, 1] * FLAT.tick
    assert np.all(m - bid_px >= d - 1e-12) and np.all(ask_px - m >= d - 1e-12)
    assert np.all(m - bid_px < d + FLAT.tick) and np.all(ask_px - m < d + FLAT.tick)


def test_target_profile_slopes_and_tilts():
    t = bk.target_depth(np.array([0.0]), np.ones((1, 2, FLAT.levels)), FLAT)
    np.testing.assert_allclose(
        t[0, 0], FLAT.depth_base * (1 + FLAT.depth_slope * np.arange(FLAT.levels)))
    np.testing.assert_allclose(t[0, 0], t[0, 1])                      # symmetric when quiet
    up = bk.target_depth(np.array([1.0]), np.ones((1, 2, FLAT.levels)), FLAT)
    assert np.all(up[0, bk.BID] > t[0, bk.BID]) and np.all(up[0, bk.ASK] < t[0, bk.ASK])


# ---------------------------------------------------------------- queue priority
def test_joining_a_level_puts_us_at_the_back():
    st = place_at(fresh(), 397, 403)
    j = bk.our_level(st, FLAT)
    assert j.tolist() == [[3, 2]]
    assert st.ahead[0, bk.BID] == pytest.approx(st.depth[0, bk.BID, 3])   # everyone is ahead
    assert st.size[0].tolist() == [FLAT.our_size, FLAT.our_size]
    # a FIRST placement costs no priority (there was nothing resting to lose)
    assert st.requotes[0].tolist() == [0, 0] and st.ticks_resting[0].tolist() == [0, 0]


def test_reposting_after_a_complete_fill_is_not_counted_as_a_requote():
    """Priority lost by MOVING a live order is the cost; re-posting after a fill
    is not, because there was nothing left in the queue to lose."""
    cfg = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, replenish_rate=0.0)
    st = place_at(fresh(cfg), 400, 401, cfg)
    from dataclasses import replace
    st = replace(st, ahead=np.zeros((1, 2)))
    out, filled, _ = bk.trade(st, mo(buy=cfg.our_size + 1.0), np.zeros(1), cfg)
    assert filled[0, bk.ASK] == pytest.approx(cfg.our_size) and not out.live[0, bk.ASK]
    again = bk.place(out, out.rest.copy(), cfg)
    assert again.requotes[0, bk.ASK] == out.requotes[0, bk.ASK]      # not a requote
    assert again.size[0, bk.ASK] == pytest.approx(cfg.our_size)      # re-posted in full
    assert again.ahead[0, bk.ASK] == pytest.approx(again.depth[0, bk.ASK, 0])  # at the back


def test_requoting_the_same_price_keeps_priority_a_different_price_loses_it():
    st = place_at(fresh(), 397, 403)
    st = replace_ahead(st, 5.0)
    kept = place_at(st, 397, 403)
    assert kept.ahead[0, bk.BID] == pytest.approx(5.0)        # priority retained
    assert kept.requotes[0, bk.BID] == 0 and kept.ticks_resting[0, bk.BID] == 1
    moved = place_at(st, 396, 403)
    assert moved.ahead[0, bk.BID] == pytest.approx(moved.depth[0, bk.BID, 4])  # back of the queue
    assert moved.requotes[0, bk.BID] == 1 and moved.ticks_resting[0, bk.BID] == 0
    assert moved.ticks_resting[0, bk.ASK] == 1                # the untouched side kept resting


def replace_ahead(state, value):
    from dataclasses import replace
    return replace(state, ahead=np.full_like(state.ahead, value))


def test_cancels_advance_us_and_adds_do_not():
    st = replace_ahead(place_at(fresh(), 397, 403), 10.0)
    d0 = st.depth[0, bk.BID, 3]
    cancelled = bk.flow(st, np.zeros((1, N_QUEUE_COLS)), np.zeros(1),
                        np.ones((1, 2, FLAT.levels)), QUIET)
    assert cancelled.ahead[0, bk.BID] == pytest.approx(10.0)   # no cancels -> no change
    u = np.zeros((1, N_QUEUE_COLS))
    u[:, bk.CANCEL[0]] = 0.5                                    # frac = 0.5*2*cancel_rate
    only_cancel = QueueConfig(depth_cv=0.0, add_rate=0.0, replenish_rate=0.0, mo_rate=0.0)
    out = bk.flow(st, u, np.zeros(1), np.ones((1, 2, FLAT.levels)), only_cancel)
    f = only_cancel.cancel_rate
    assert out.ahead[0, bk.BID] == pytest.approx(10.0 * (1 - f))
    assert out.depth[0, bk.BID, 3] == pytest.approx(d0 * (1 - f))
    u2 = np.zeros((1, N_QUEUE_COLS))
    u2[:, bk.ADD[0]] = 1.0
    only_add = QueueConfig(depth_cv=0.0, cancel_rate=0.0, replenish_rate=0.0, mo_rate=0.0)
    out2 = bk.flow(st, u2, np.zeros(1), np.ones((1, 2, FLAT.levels)), only_add)
    assert out2.ahead[0, bk.BID] == pytest.approx(10.0)          # adds join BEHIND us
    assert out2.depth[0, bk.BID, 3] > d0


def test_replenishment_pulls_depth_toward_target():
    from dataclasses import replace
    st = fresh()
    thin = replace(st, depth=st.depth * 0.1)
    mult = np.ones((1, 2, FLAT.levels))
    target = bk.target_depth(np.zeros(1), mult, FLAT)
    cfg = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, mo_rate=0.0)
    out = bk.flow(thin, np.zeros((1, N_QUEUE_COLS)), np.zeros(1), mult, cfg)
    gap_before, gap_after = target - thin.depth, target - out.depth
    assert np.all(gap_after < gap_before) and np.all(gap_after > 0)
    np.testing.assert_allclose(out.depth, thin.depth + cfg.replenish_rate * gap_before)


# ---------------------------------------------------------------- trades
def mo(n=1, buy=0.0, sell=0.0):
    """Uniforms that fire a market order of a given SIZE on each side."""
    u = np.zeros((n, N_QUEUE_COLS))
    u[:, bk.MO_ARRIVE[0]] = 0.0 if sell else 1.0        # arrive iff u < mo_rate
    u[:, bk.MO_ARRIVE[1]] = 0.0 if buy else 1.0
    for col, want in ((bk.MO_SIZE[0], sell), (bk.MO_SIZE[1], buy)):
        u[:, col] = 1 - np.exp(-want / CFG.mo_size_mean) if want else 0.0
    return u


def test_market_order_eats_ahead_then_us_then_behind():
    cfg = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, replenish_rate=0.0)
    from dataclasses import replace
    # rest at the TOUCH (ask level 0 = anchor+1) so a small order reaches us
    st = place_at(fresh(cfg), 400, 401, cfg)
    assert bk.our_level(st, cfg)[0, bk.ASK] == 0
    st = replace(st, ahead=np.array([[0.0, 4.0]]))      # 4 lots ahead of us on the ask
    d_before = st.depth[0, bk.ASK, 0]
    # a buy of 4 clears exactly the queue ahead of us: no fill yet
    out, filled, traded = bk.trade(st, mo(buy=4.0), np.zeros(1), cfg)
    assert filled[0, bk.ASK] == pytest.approx(0.0)
    assert out.ahead[0, bk.ASK] == pytest.approx(0.0)
    assert out.depth[0, bk.ASK, 0] == pytest.approx(d_before - 4.0)
    # a buy of 4.6 clears the queue and takes 0.6 of our 1.0: a PARTIAL fill
    out, filled, traded = bk.trade(st, mo(buy=4.6), np.zeros(1), cfg)
    assert filled[0, bk.ASK] == pytest.approx(0.6)
    assert out.size[0, bk.ASK] == pytest.approx(0.4) and out.live[0, bk.ASK]
    # a buy of 6 fills us completely and eats into the size behind us
    out, filled, traded = bk.trade(st, mo(buy=6.0), np.zeros(1), cfg)
    assert filled[0, bk.ASK] == pytest.approx(1.0)
    assert out.size[0, bk.ASK] == pytest.approx(0.0) and not out.live[0, bk.ASK]
    assert out.depth[0, bk.ASK, 0] == pytest.approx(d_before - 5.0)   # 4 ahead + 1 behind


def test_a_small_market_order_never_reaches_us():
    cfg = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, replenish_rate=0.0)
    st = place_at(fresh(cfg), 397, 403, cfg)            # we join the back: whole level ahead
    _, filled, _ = bk.trade(st, mo(buy=1.0, sell=1.0), np.zeros(1), cfg)
    assert filled.sum() == pytest.approx(0.0)


def test_market_orders_walk_levels_from_the_touch_outward():
    cfg = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, replenish_rate=0.0)
    st = fresh(cfg)
    d = st.depth[0, bk.ASK].copy()
    size = d[0] + d[1] + 3.0
    out, _, traded = bk.trade(st, mo(buy=size), np.zeros(1), cfg)
    assert out.depth[0, bk.ASK, 0] == pytest.approx(0.0)
    assert out.depth[0, bk.ASK, 1] == pytest.approx(0.0)
    assert out.depth[0, bk.ASK, 2] == pytest.approx(d[2] - 3.0)
    assert traded[0, bk.ASK] == pytest.approx(size)
    assert out.depth[0, bk.BID].tolist() == st.depth[0, bk.BID].tolist()   # other side untouched


def test_a_market_order_larger_than_the_book_consumes_everything_available():
    cfg = QueueConfig(levels=4, depth_cv=0.0, cancel_rate=0.0, add_rate=0.0,
                      replenish_rate=0.0)
    st = fresh(cfg)
    total = float(st.depth[0, bk.ASK].sum())
    out, _, traded = bk.trade(st, mo(buy=total + 20.0), np.zeros(1), cfg)
    assert np.all(out.depth[0, bk.ASK] == pytest.approx(0.0))
    assert traded[0, bk.ASK] == pytest.approx(total)      # traded excludes what it could not fill


def test_informed_flow_boosts_the_side_about_to_be_picked_off():
    cfg = QueueConfig(depth_cv=0.0, mo_rate=0.2, informed_mo_mult=3.0)
    n = 20000
    rng = np.random.default_rng(4)
    u = np.zeros((n, N_QUEUE_COLS))
    u[:, [bk.MO_ARRIVE[0], bk.MO_ARRIVE[1]]] = rng.random((n, 2))
    u[:, [bk.MO_SIZE[0], bk.MO_SIZE[1]]] = rng.random((n, 2))
    st = fresh(cfg, n=n)
    for direction, hot in ((1, bk.ASK), (-1, bk.BID)):
        _, _, traded = bk.trade(st, u, np.full(n, direction), cfg)
        rate_hot = (traded[:, hot] > 0).mean()
        rate_cold = (traded[:, 1 - hot] > 0).mean()
        assert rate_hot == pytest.approx(cfg.mo_rate * cfg.informed_mo_mult, abs=0.02)
        assert rate_cold == pytest.approx(cfg.mo_rate, abs=0.02)


# ---------------------------------------------------------------- the mid moving
def test_our_order_keeps_its_price_when_the_mid_moves():
    st = place_at(fresh(), 397, 403)
    before = bk.our_level(st, FLAT)
    mult = np.ones((1, 2, FLAT.levels))
    out = bk.reanchor(st, np.array([100.5]), mult, np.zeros(1), FLAT)   # anchor 400 -> 402
    assert out.anchor[0] == 402
    assert out.rest[0].tolist() == st.rest[0].tolist()                  # the ORDER did not move
    after = bk.our_level(out, FLAT)
    assert after[0, bk.BID] == before[0, bk.BID] + 2                    # it is now deeper
    assert after[0, bk.ASK] == before[0, bk.ASK] - 2                    # and nearer on the ask


def test_reanchor_carries_depth_with_the_price_and_fills_new_levels_from_target():
    from dataclasses import replace
    st = fresh()
    marked = replace(st, depth=st.depth.copy())
    marked.depth[0, bk.BID, 2] = 123.0                                  # a landmark at that price
    mult = np.ones((1, 2, FLAT.levels))
    out = bk.reanchor(marked, np.array([100.25]), mult, np.zeros(1), FLAT)   # anchor +1
    assert out.depth[0, bk.BID, 3] == pytest.approx(123.0)              # same price, new index
    target = bk.target_depth(np.zeros(1), mult, FLAT)
    assert out.depth[0, bk.BID, 0] == pytest.approx(target[0, bk.BID, 0])   # newly exposed
    assert out.depth[0, bk.ASK, FLAT.levels - 1] == pytest.approx(target[0, bk.ASK, -1])


def test_an_order_outside_the_window_keeps_its_price_and_priority_but_cannot_fill():
    cfg = QueueConfig(depth_cv=0.0, cancel_rate=0.0, add_rate=0.0, replenish_rate=0.0)
    st = place_at(fresh(cfg), 400, 401, cfg)          # resting at the touch
    from dataclasses import replace
    st = replace(st, ahead=np.array([[2.0, 2.0]]))
    mult = np.ones((1, 2, cfg.levels))
    far = bk.reanchor(st, np.array([130.0]), mult, np.zeros(1), cfg)
    assert bk.our_level(far, cfg)[0, bk.ASK] == -1    # outside the modelled window
    assert far.live[0, bk.ASK] and far.rest[0, bk.ASK] == st.rest[0, bk.ASK]
    assert far.ahead[0, bk.ASK] == pytest.approx(2.0)  # priority is NOT reset
    _, filled, _ = bk.trade(far, mo(buy=1e3), np.zeros(1), cfg)
    assert filled.sum() == pytest.approx(0.0)          # and it cannot be reached
    back = bk.reanchor(far, np.array([100.0]), mult, np.zeros(1), cfg)
    assert bk.our_level(back, cfg)[0, bk.ASK] == 0     # fillable again when the mid returns
    assert back.ahead[0, bk.ASK] == pytest.approx(2.0)


def test_requoting_to_the_same_price_outside_the_window_keeps_priority():
    cfg = QueueConfig(depth_cv=0.0)
    st = place_at(fresh(cfg), 400, 401, cfg)
    from dataclasses import replace
    st = replace(st, ahead=np.array([[3.0, 3.0]]))
    far = bk.reanchor(st, np.array([130.0]), np.ones((1, 2, cfg.levels)), np.zeros(1), cfg)
    again = bk.place(far, far.rest.copy(), cfg)
    assert again.ahead[0, bk.ASK] == pytest.approx(3.0)
    assert again.requotes[0, bk.ASK] == far.requotes[0, bk.ASK]


# ---------------------------------------------------------------- invariants
def run_session(seed=0, n=4, cfg=CFG, ticks=200, eta=0.1, delta=0.75):
    """Drive the book the way core.step will, and collect diagnostics."""
    market = MarketConfig(sigma=2.0)
    d = draw_batch(ArithmeticRandomWalk(market), seed, n)
    mids = ArithmeticRandomWalk(market).path_from_shocks(d.shocks)
    mult = bk.session_multipliers(d.u_depth0, cfg)
    st = bk.initial_state(mids[:, 0], d.u_depth0, cfg)
    q = np.zeros(n)
    stats = dict(fills=np.zeros((n, 2)), ticks=0, outside=0, kept=0)
    for t in range(ticks):
        want = bk.quote_grids(delta + eta * q, delta - eta * q, mids[:, t], cfg)
        before = st.rest.copy()
        st = bk.place(st, want, cfg)
        stats["kept"] += int(np.sum(st.rest == before))
        st = bk.flow(st, d.u_queue[:, t, :], np.zeros(n), mult, cfg)
        st, filled, _ = bk.trade(st, d.u_queue[:, t, :], np.zeros(n), cfg)
        q = q + filled[:, bk.BID] - filled[:, bk.ASK]
        stats["fills"] += filled
        j = bk.our_level(st, cfg)
        stats["outside"] += int(np.sum(j < 0))
        stats["ticks"] += 1
        assert_invariants(st, cfg)
        st = bk.reanchor(st, mids[:, t + 1], mult, np.zeros(n), cfg)
    return st, stats


def assert_invariants(st, cfg):
    assert np.all(st.depth >= 0) and np.all(np.isfinite(st.depth))
    assert np.all(st.ahead >= -1e-9) and np.all(st.size >= -1e-9)
    j = bk.our_level(st, cfg)
    inside = j >= 0
    if inside.any():
        d_at = np.take_along_axis(st.depth, np.where(inside, j, 0)[:, :, None], axis=2)[:, :, 0]
        assert np.all(st.ahead[inside] <= d_at[inside] + 1e-9), "ahead exceeds the level's depth"
    assert np.all(st.size <= cfg.our_size + 1e-9)


def test_invariants_hold_over_a_full_session():
    st, stats = run_session()
    assert stats["fills"].sum() > 0                      # the model actually trades


def test_batch_of_one_matches_a_row_of_a_batch():
    many, _ = run_session(seed=9, n=5)
    one, _ = run_session(seed=9, n=1)
    for f in ("anchor", "depth", "rest", "ahead", "size", "requotes"):
        np.testing.assert_array_equal(getattr(one, f)[0], getattr(many, f)[0])


def test_book_draws_have_fixed_shapes_and_do_not_depend_on_config():
    d = draw_session(ArithmeticRandomWalk(MarketConfig()), 3, 0)
    assert d.u_queue.shape == (MarketConfig().N, N_QUEUE_COLS)
    assert d.u_depth0.shape == (2, MAX_LEVELS)


def test_normal_quantile_approximation_is_accurate():
    # reference values of Phi^{-1}(p)
    for p, want in [(0.5, 0.0), (0.975, 1.959963985), (0.025, -1.959963985),
                    (0.001, -3.090232306), (0.999, 3.090232306), (0.01, -2.326347874),
                    (0.2, -0.841621234), (1e-6, -4.753424309)]:
        assert bk._ppf(np.array([p]))[0] == pytest.approx(want, abs=2e-6)
    u = np.linspace(1e-6, 1 - 1e-6, 20001)
    z = bk._ppf(u)
    assert np.all(np.diff(z) > 0)                        # monotone
    assert bk._lognormal(np.full(5000, 0.5), 0.3).mean() == pytest.approx(np.exp(-0.045), rel=1e-6)


def test_config_validation():
    for bad in (lambda: QueueConfig(tick=0), lambda: QueueConfig(levels=0),
                lambda: QueueConfig(levels=25), lambda: QueueConfig(depth_base=0),
                lambda: QueueConfig(mo_rate=1.5), lambda: QueueConfig(cancel_rate=-0.1),
                lambda: QueueConfig(our_size=0), lambda: QueueConfig(informed_mo_mult=0.5),
                lambda: QueueConfig(depth_slope=-1)):
        with pytest.raises(ValueError):
            bad()
