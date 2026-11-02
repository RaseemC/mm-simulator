"""C1 (2-D): the flow/drift separation and the arrival-probability saturation.

These tests protect the two findings the 2-D sweep is built on: (1) strength
and informed_mo_mult are independent knobs that core.step actually applies
independently, and (2) the arrival probability min(1, mo_rate * mo_mult)
genuinely saturates, which is why the sweep's grid was redesigned around it.
"""
import numpy as np
import pytest

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, FillConfig, MarketConfig, QueueConfig
from mmsim.core import simulate_paths
from mmsim.midprice import ArithmeticRandomWalk
from mmsim.strategies import InventorySkewQuoter

MKT = MarketConfig(sigma=2.0)


def terminal_pnl(strength, mo_mult, eta=0.1, n=800, seed=2024):
    adv = AdverseConfig(strength=strength)
    qc = QueueConfig(informed_mo_mult=mo_mult)
    prep = prepare_batch(ArithmeticRandomWalk(MKT), seed, n, adv)
    res = simulate_paths(prep.mids, prep.u_bid, prep.u_ask, InventorySkewQuoter(0.75, eta),
                         FillConfig(), MKT.dt, informed=prep.informed, fill_boost=adv.fill_boost,
                         queue=qc, u_queue=prep.u_queue, u_depth0=prep.u_depth0, tilt=prep.tilt)
    return res["terminal_pnl"]


def test_strength_zero_ignores_informed_mo_mult_entirely():
    """With no episodes, the flow-boost parameter cannot matter: build_regime
    short-circuits on strength == 0 and emits zero `informed` everywhere."""
    a = terminal_pnl(0.0, 1.0)
    b = terminal_pnl(0.0, 5.0)
    np.testing.assert_array_equal(a, b)


def test_mo_mult_changes_outcomes_once_episodes_exist():
    """The opposite case: with episodes present, mo_mult must matter (below
    the saturation point), confirming the two knobs are genuinely independent."""
    a = terminal_pnl(0.2, 1.0)
    b = terminal_pnl(0.2, 1.5)
    assert not np.array_equal(a, b)
    # the effect is a REAL shift, not noise: paired mean should differ by more
    # than a few standard errors
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert abs(d.mean()) > 5 * se


def test_strength_changes_outcomes_with_flow_boost_off():
    """Drift alone (informed_mo_mult = 1.0, i.e. no flow boost) must still move
    the P&L: the mid genuinely drifts during an episode even without extra fills."""
    a = terminal_pnl(0.05, 1.0)
    b = terminal_pnl(0.5, 1.0)
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert abs(d.mean()) > 5 * se


def test_arrival_probability_saturates_at_the_documented_point():
    """Arrival is drawn as u < min(1, mo_rate * mo_mult) with u ~ Uniform[0, 1).
    Because u is STRICTLY less than 1, any threshold >= 1 makes arrival certain
    on every tick, so mo_mult at or past 1/mo_rate is indistinguishable from any
    larger value: this is an inherent property of threshold-vs-uniform sampling,
    not an explicit clamp that could be left out by mistake (confirmed: removing
    the np.minimum() wrapper changes nothing, since u < 1 already holds for every
    draw - there is no mutation of that line this test could usefully catch).
    What IS worth protecting is the empirical claim the 2-D sweep's grid design
    relies on: results at and past the threshold are identical, and results
    strictly below it are not."""
    cfg = QueueConfig()
    cap = 1.0 / cfg.mo_rate
    at_cap = terminal_pnl(0.2, cap)
    past_cap = terminal_pnl(0.2, cap + 1.0)
    np.testing.assert_array_equal(at_cap, past_cap)
    just_below = terminal_pnl(0.2, cap - 0.2)
    assert not np.array_equal(just_below, at_cap)   # below the threshold, it still moves


def test_c1_default_mo_mult_was_already_past_the_saturation_point():
    """Documents why the 1-D sweep's queue-model toxicity axis was, in effect,
    a regime switch (episodes on/off) rather than a continuously varying flow
    strength: its mo_mult (2.5) sits past the cap (2.0 at the default mo_rate)."""
    cfg = QueueConfig()  # informed_mo_mult default used throughout C1 (1-D)
    assert cfg.informed_mo_mult > 1.0 / cfg.mo_rate


def test_sharpe_and_pnl_gap_can_rank_fill_models_oppositely():
    """Regression for the C1 finding that the two statistics are not
    interchangeable: construct paired aware/naive outcomes where one ranking
    flips relative to the other, confirming the metric (not just the mechanism)
    needs to be reported explicitly rather than assumed to agree."""
    rng = np.random.default_rng(0)
    # model A: small mean edge, small spread (favours Sharpe)
    naive_a = rng.normal(0, 5, 5000)
    aware_a = naive_a + rng.normal(2, 1, 5000)
    # model B: larger mean edge, much larger spread (favours raw P&L)
    naive_b = rng.normal(0, 40, 5000)
    aware_b = naive_b + rng.normal(15, 30, 5000)

    def sharpe(x):
        return x.mean() / x.std(ddof=1)
    pnl_gap_a = aware_a.mean() - naive_a.mean()
    pnl_gap_b = aware_b.mean() - naive_b.mean()
    sharpe_gap_a = sharpe(aware_a) - sharpe(naive_a)
    sharpe_gap_b = sharpe(aware_b) - sharpe(naive_b)
    assert pnl_gap_b > pnl_gap_a           # B has the bigger P&L advantage
    assert sharpe_gap_b < sharpe_gap_a     # but the smaller Sharpe advantage
