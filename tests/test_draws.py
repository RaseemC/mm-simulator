"""The queue-draws allocation fix: want_queue=False must draw nothing extra.

Found the hard way: `prepare_batch` always drew the queue-model blocks
(u_queue, u_depth0) regardless of whether `queue=` was ever passed to
`simulate_paths`, which made large exponential-only runs slow and
memory-heavy for no benefit (a 200,000-session diagnostic allocated several
GB of u_queue it never read, and had to be killed - see DESIGN_NOTES.md).
"""
import numpy as np

from mmsim.adverse import prepare_batch
from mmsim.config import AdverseConfig, MarketConfig
from mmsim.draws import MAX_LEVELS, N_QUEUE_COLS, draw_batch, draw_session, session_rng
from mmsim.midprice import ArithmeticRandomWalk

MKT = MarketConfig()


def test_want_queue_false_returns_none_and_draws_nothing():
    proc = ArithmeticRandomWalk(MKT)
    d = draw_session(proc, 5, 0, want_queue=False)
    assert d.u_queue is None and d.u_depth0 is None
    # every other field is drawn as normal
    for f in ("shocks", "u_bid", "u_ask", "u_start", "u_dir", "u_dur"):
        assert getattr(d, f) is not None


def test_want_queue_false_consumes_fewer_random_numbers():
    """The generator must stop exactly where the non-queue blocks end: nothing
    downstream depends on how much of a session's OWN generator is consumed,
    since each session's generator is independent (session_rng), but the
    remaining stream should be exactly what a want_queue=True run would have
    produced for u_queue/u_depth0, proving nothing was skipped mid-stream."""
    proc = ArithmeticRandomWalk(MKT)
    full = draw_session(proc, 9, 3, want_queue=True)
    lean = draw_session(proc, 9, 3, want_queue=False)
    for f in ("shocks", "u_bid", "u_ask", "u_start", "u_dir", "u_dur"):
        np.testing.assert_array_equal(getattr(full, f), getattr(lean, f))
    # the lean session's generator, resumed, must pick up exactly where the
    # full session's queue blocks came from
    rng = session_rng(9, 3)
    rng.standard_normal(MKT.N)                      # shocks
    rng.random((MKT.N, 2))                           # u_bid, u_ask
    rng.random((MKT.N, 3))                           # u_start, u_dir, u_dur
    w = rng.random((MKT.N, N_QUEUE_COLS))
    d0 = rng.random((2, MAX_LEVELS))
    np.testing.assert_array_equal(full.u_queue, w)
    np.testing.assert_array_equal(full.u_depth0, d0)


def test_draw_batch_and_prepare_batch_propagate_want_queue():
    proc = ArithmeticRandomWalk(MKT)
    batch = draw_batch(proc, 1, 5, want_queue=False)
    assert batch.u_queue is None and batch.u_depth0 is None
    prep = prepare_batch(proc, 1, 5, AdverseConfig(), want_queue=False)
    assert prep.u_queue is None and prep.u_depth0 is None
    prep2 = prepare_batch(proc, 1, 5, AdverseConfig(), want_queue=True)
    assert prep2.u_queue is not None and prep2.u_queue.shape == (5, MKT.N, N_QUEUE_COLS)
    assert prep2.u_depth0.shape == (5, 2, MAX_LEVELS)


def test_want_queue_false_is_the_default_for_backward_compatibility():
    """Old call sites that never pass want_queue keep drawing the blocks, so
    they cannot be silently affected by this change."""
    proc = ArithmeticRandomWalk(MKT)
    d = draw_session(proc, 1, 0)
    assert d.u_queue is not None and d.u_depth0 is not None


def test_non_queue_draws_are_bit_identical_regardless_of_want_queue_for_a_whole_batch():
    proc = ArithmeticRandomWalk(MKT)
    full = draw_batch(proc, 42, 20, want_queue=True)
    lean = draw_batch(proc, 42, 20, want_queue=False)
    for f in ("shocks", "u_bid", "u_ask", "u_start", "u_dir", "u_dur"):
        np.testing.assert_array_equal(getattr(full, f), getattr(lean, f))


def test_exponential_only_path_allocates_no_queue_memory():
    """A direct measurement, not just a None check: the array objects for
    u_queue/u_depth0 must never come into existence on the exponential path.
    Measured as the DELTA between want_queue=True and False (not an absolute
    ceiling), since the six always-drawn fields already cost several MB at this
    n on their own - what must be near-zero is specifically the QUEUE share."""
    import tracemalloc
    proc = ArithmeticRandomWalk(MarketConfig(N=200))
    n = 2000
    queue_block_bytes = n * MarketConfig(N=200).N * N_QUEUE_COLS * 8   # ~25.6 MB

    def peak_for(want_queue):
        tracemalloc.start()
        prepare_batch(proc, 1, n, AdverseConfig(), want_queue=want_queue)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    peak_with = peak_for(True)
    peak_without = peak_for(False)
    delta = peak_with - peak_without
    assert delta > queue_block_bytes * 0.5, (
        f"sanity check failed: want_queue=True should cost ~{queue_block_bytes} bytes more "
        f"than want_queue=False, only saw a delta of {delta}")
    prep = prepare_batch(proc, 1, n, AdverseConfig(), want_queue=False)
    assert prep.u_queue is None and prep.u_depth0 is None


def test_map_and_stack_handle_none_queue_fields():
    proc = ArithmeticRandomWalk(MKT)
    d = draw_session(proc, 2, 0, want_queue=False)
    mapped = d.map(lambda a: a[None, :])
    assert mapped.u_queue is None and mapped.shocks.shape == (1, MKT.N)
    batch = draw_batch(proc, 2, 3, want_queue=False)
    assert batch.u_queue is None and batch.shocks.shape == (3, MKT.N)
