"""
THE MOST IMPORTANT TEST IN THE PROJECT.

Lookahead = the code secretly using information from after the decision.
It makes backtests look amazing and live trading lose. The classic SMC case: using a
swing point at the bar where it happened, when it can only be confirmed n bars later.

The test: run the scan on all bars. Then, for many setups, cut the data off right
after the setup's placement bar and run the scan again. The setup must come out
IDENTICAL both times (same bar, same entry, same stop, same features). If the answer
for bar 1500 changes because bars 1501+ exist, the code is peeking.

Cutting right at the decision bar matters: a leak that peeks only 5 bars ahead
is invisible if you cut the data somewhere random.

Run with:  python -m pytest tests      (or)      python tests/test_no_lookahead.py
Run it again every time you change detectors.py, setups.py or add a feature.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from smcml.data import simulate  # noqa: E402
from smcml.setups import scan, feature_columns  # noqa: E402
from smcml.chart_images import images_for  # noqa: E402

COLS = (["t_place", "direction", "entry", "stop"] + feature_columns(cfg)
        + [f"bias_{n}_{k}" for n in ("d1", "h4", "h1") for k in ("trend", "level", "since")])


class bias_filter:
    """Temporarily switch the D1/H4/H1 filter on or off inside a test."""
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old, cfg.BIAS_FILTER = cfg.BIAS_FILTER, self.on

    def __exit__(self, *exc):
        cfg.BIAS_FILTER = self.old


def _cuts(full, n=40, seed=0):
    """Cut points right after (a sample of) the placement bars."""
    t = full["t_place"].to_numpy()
    rng = np.random.default_rng(seed)
    return sorted(set(rng.choice(t, size=min(n, len(t)), replace=False) + 1))


def _check(df, cuts=None):
    full = scan(df, cfg)
    for T in (cuts or _cuts(full)):
        part = scan(df.iloc[:T], cfg)
        a = full[full["t_place"] < T][COLS].reset_index(drop=True)
        b = part[COLS].reset_index(drop=True)
        assert len(a) == len(b), f"cut at {T}: {len(a)} setups in full run vs {len(b)} in truncated run"
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)
    return full


def test_scan_has_no_lookahead_random():
    # filter off: many more setups, and the D1/H4/H1 biases are still checked as features
    with bias_filter(False):
        df = simulate(4000, mode="random", seed=11)
        full = _check(df)
        assert len(full) > 5


def test_scan_has_no_lookahead_with_bias_filter():
    with bias_filter(True):
        df = simulate(8000, mode="random", seed=15)
        full = _check(df)
        assert len(full) > 3


def test_scan_has_no_lookahead_planted():
    with bias_filter(False):
        df = simulate(4000, mode="planted", seed=12)
        _check(df)


def test_future_garbage_changes_nothing():
    """Replace everything after bar T with nonsense: setups before T must not move."""
    df = simulate(3000, mode="random", seed=13)
    with bias_filter(False):
        a_full = scan(df, cfg)
    rng = np.random.default_rng(0)
    for T in _cuts(a_full, n=10, seed=1):
        junk = df.copy()
        junk.iloc[T:] = junk.iloc[T:].to_numpy() * rng.uniform(0.5, 1.5, size=(len(df) - T, 1))
        junk["high"] = junk[["open", "high", "low", "close"]].max(axis=1)
        junk["low"] = junk[["open", "high", "low", "close"]].min(axis=1)
        with bias_filter(False):
            b = scan(junk, cfg)
        a = a_full[a_full["t_place"] < T][COLS].reset_index(drop=True)
        b = b[b["t_place"] < T][COLS].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_chart_pictures_have_no_lookahead():
    df = simulate(3000, mode="random", seed=14)
    all_setups = scan(df, cfg)
    for T in _cuts(all_setups, n=10, seed=2):
        full = all_setups[all_setups["t_place"] < T].reset_index(drop=True)
        part = scan(df.iloc[:T], cfg).reset_index(drop=True)
        assert np.array_equal(images_for(full, df), images_for(part, df.iloc[:T]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
