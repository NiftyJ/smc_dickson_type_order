"""
Checks for the Wyckoff range tools (wyckoff/).

  * starter boxes obey their own rules
  * the simple "range right now" rule and the model's pictures only use closed candles
  * the value used for a setup is from the last H4 candle that had CLOSED
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from smcml.bias import resample, bar_length  # noqa: E402
from smcml.data import simulate  # noqa: E402
from smcml.detectors import atr  # noqa: E402
from wyckoff.ranges import find_ranges, in_range_mask, range_now, value_at_decision, settings  # noqa: E402
from wyckoff.model import render, WINDOW  # noqa: E402

DF = simulate(40_000, mode="random", seed=11)
BARS = resample(DF, cfg.RANGE_TF)


def test_starter_boxes_follow_the_rules():
    s = settings(cfg)
    boxes = find_ranges(DF, cfg)
    assert len(boxes) > 3
    h, l, c = (BARS[k].to_numpy(float) for k in ("high", "low", "close"))
    a = atr(h, l, c, 14)
    for b in boxes:
        m = (BARS.index >= pd.Timestamp(b["start"])) & (BARS.index <= pd.Timestamp(b["end"]))
        assert m.sum() >= s["min_bars"]
        assert b["top"] - b["bottom"] <= s["max_width"] * np.nanmedian(a[m]) + 1e-9
    starts = [pd.Timestamp(b["start"]) for b in boxes]
    assert starts == sorted(starts)                            # no overlaps, in time order
    assert all(pd.Timestamp(x["end"]) < pd.Timestamp(y["start"]) for x, y in zip(boxes, boxes[1:]))


def test_range_now_uses_only_closed_candles():
    full = range_now(DF, cfg)
    rng = np.random.default_rng(0)
    for t in rng.choice(np.arange(200, len(BARS) - 5), 6, replace=False):
        cut = DF[DF.index < BARS.index[t] + pd.Timedelta(cfg.RANGE_TF)]     # data up to the end of candle t
        part = range_now(cut, cfg)
        assert part.iloc[-1] == full.iloc[t] and part.index[-1] == full.index[t]


def test_picture_ignores_the_future():
    h, l, c = (BARS[k].to_numpy(float) for k in ("high", "low", "close"))
    t = 500
    a = render(h, l, c, t)
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[t + 1:] *= 3
    l2[t + 1:] *= 0.3
    c2[t + 1:] *= 2
    assert (render(h2, l2, c2, t) == a).all()
    assert a.shape[1] == 2 * WINDOW


def test_setups_get_the_last_closed_candle():
    s = pd.Series(np.arange(len(BARS)), index=BARS.index)
    base = bar_length(DF.index)
    times = DF.index[[1000, 1015, 1016, 5000]]
    got = value_at_decision(s, times, base, cfg.RANGE_TF)
    for t, v in zip(times, got):
        closed = BARS.index[BARS.index + pd.Timedelta(cfg.RANGE_TF) <= t + base]
        assert v == s[closed[-1]]


def test_mask_matches_boxes():
    boxes = [dict(start=str(BARS.index[10]), end=str(BARS.index[20]), top=1, bottom=0)]
    m = in_range_mask(BARS.index, boxes)
    assert m[10:21].all() and m.sum() == 11
