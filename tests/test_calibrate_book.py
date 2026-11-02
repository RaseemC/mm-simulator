"""B+3 calibration script: its replay of the queue path must agree with the engine,
or every number it reports about published depth statistics is about a different book.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

from mmsim.config import QueueConfig


def load_calibrate():
    path = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_book.py"
    spec = importlib.util.spec_from_file_location("calibrate_book_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_matches_engine_fills_event_for_event():
    cb = load_calibrate()
    for cfg in (QueueConfig(), QueueConfig(mo_size_mean=QueueConfig().depth_base / math.log(2))):
        out = cb.measure(cfg, n=30, check=True)          # asserts internally
        assert out["engine_checked"]
        assert 0.0 <= out["p_sweep"] <= 1.0 and out["fill_rate"] > 0
        assert out["profile"].shape == (cfg.levels,) and np.all(out["profile"] >= 0)


def test_targets_are_kept_separate_and_stated():
    cb = load_calibrate()
    assert cb.EXP_TARGET == 140.0 * math.exp(-1.5 * 0.75) * 0.005   # (ii) from the exp model only
    assert cb.P_SWEEP_MAX == 0.5 and cb.CANCEL_RATIO_MIN == 10.0     # (i) assumed thresholds


def test_pareto_bound_is_monotone_in_queue_position():
    cb = load_calibrate()
    b = cb.pareto_bound(np.ones(16), median_over_touch=1.0)
    assert b["back"] <= b["front"] <= 1.0
    assert math.isclose(b["p_sweep"], 0.5)          # median = touch => half the MOs sweep it
