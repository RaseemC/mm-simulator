"""C2: markout decomposition (distance_captured + impact = markout), weighting
under partial fills, and the informed/quiet split."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parent.parent / "scripts" / "markout_analysis.py"
    spec = importlib.util.spec_from_file_location("markout_analysis_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load()
per_session_sums, weighted_mean = _m.per_session_sums, _m.weighted_mean


def toy(n=2, N=6):
    """A small, hand-checkable scenario: mids known, fills placed by hand."""
    mids = np.zeros((n, N + 1))
    mids[0] = [100.0, 100.0, 101.0, 101.0, 102.0, 102.0, 103.0]
    mids[1] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    bid = mids[:, :-1] - 0.5
    ask = mids[:, :-1] + 0.5
    f_bid = np.zeros((n, N))
    f_ask = np.zeros((n, N))
    f_bid[0, 0] = 1.0        # session 0: bought at t=0, bid=99.5
    f_ask[0, 2] = 2.0        # session 0: sold size 2 at t=2, ask=101.5
    f_bid[1, 1] = 1.0        # session 1: flat prices throughout
    informed = np.zeros((n, N), dtype=np.int8)
    informed[0, 0] = 1       # the t=0 bid fill is tagged informed
    return mids, bid, ask, f_bid, f_ask, informed


def test_distance_captured_matches_the_static_half_spread():
    mids, bid, ask, f_bid, f_ask, informed = toy()
    s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h=2)
    # session 0's informed fill: bid at t=0, distance = mid[0] - bid[0] = 100 - 99.5 = 0.5
    assert weighted_mean(s["inf_dist"], s["inf_den"]) == pytest.approx(0.5)


def test_impact_is_the_pure_forward_price_move():
    mids, bid, ask, f_bid, f_ask, informed = toy()
    s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h=2)
    # mid[0]=100, mid[2]=101: impact for the bid fill = mid[t+h]-mid[t] = 1.0
    assert weighted_mean(s["inf_imp"], s["inf_den"]) == pytest.approx(1.0)


def test_markout_equals_distance_plus_impact_exactly():
    mids, bid, ask, f_bid, f_ask, informed = toy()
    for h in (1, 2, 3):
        s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h)
        for tag in ("inf", "q"):
            np.testing.assert_allclose(s[f"{tag}_mk"], s[f"{tag}_dist"] + s[f"{tag}_imp"])


def test_ask_fill_sign_convention_sells_profit_when_price_falls():
    """An ask fill (we sold) should show POSITIVE impact when the price falls
    afterward - we sold high before the drop."""
    n, N = 1, 4
    mids = np.array([[100.0, 100.0, 98.0, 98.0, 98.0]])
    bid, ask = mids[:, :-1] - 0.5, mids[:, :-1] + 0.5
    f_bid, f_ask = np.zeros((n, N)), np.zeros((n, N))
    f_ask[0, 0] = 1.0
    informed = np.zeros((n, N), dtype=np.int8)
    s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h=2)
    # sold at ask=100.5, mid fell to 98 by t+2: impact = mid_now - mid_future = 100-98=+2
    assert weighted_mean(s["q_imp"], s["q_den"]) == pytest.approx(2.0)
    assert weighted_mean(s["q_dist"], s["q_den"]) == pytest.approx(0.5)
    assert weighted_mean(s["q_mk"], s["q_den"]) == pytest.approx(2.5)


def test_partial_fills_are_size_weighted():
    """A size-2 fill must count twice as much as a size-1 fill at the same price
    and horizon; mixing two different distances must give their SIZE-weighted
    average, not a plain average of the two fill events."""
    n, N = 1, 3
    mids = np.array([[100.0, 100.0, 100.0, 104.0]])
    bid = mids[:, :-1].copy()
    bid[0, 0], bid[0, 1] = 99.0, 98.0     # distances 1.0 and 2.0
    ask = mids[:, :-1] + 10.0             # never filled
    f_bid = np.array([[3.0, 1.0, 0.0]])   # size 3 at distance 1.0, size 1 at distance 2.0
    f_ask = np.zeros((n, N))
    informed = np.zeros((n, N), dtype=np.int8)
    s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h=1)
    # size-weighted distance = (3*1.0 + 1*2.0) / 4 = 1.25, NOT the plain average 1.5
    assert weighted_mean(s["q_dist"], s["q_den"]) == pytest.approx(1.25)
    assert s["q_den"].sum() == pytest.approx(4.0)


def test_fills_too_close_to_the_session_end_are_dropped_for_that_horizon():
    n, N = 1, 4
    mids = np.array([[100.0] * (N + 1)])
    bid, ask = mids[:, :-1] - 0.5, mids[:, :-1] + 0.5
    f_bid = np.zeros((n, N))
    f_bid[0, N - 1] = 1.0                 # the very last tick: no mid[t+h] exists for h>=1
    f_ask = np.zeros((n, N))
    informed = np.zeros((n, N), dtype=np.int8)
    s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h=1)
    assert s["q_den"].sum() == 0.0        # the fill is entirely excluded, not padded


def test_informed_and_quiet_denominators_partition_total_filled_size():
    mids, bid, ask, f_bid, f_ask, informed = toy()
    s = per_session_sums(mids, bid, ask, f_bid, f_ask, informed, h=1)
    total = f_bid.sum() + f_ask.sum()
    assert s["inf_den"].sum() + s["q_den"].sum() == pytest.approx(total)


def test_weighted_mean_handles_zero_denominator():
    assert np.isnan(weighted_mean(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
