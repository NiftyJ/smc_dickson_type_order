"""
STARTER RANGE LABELS, found with hindsight.

This is NOT the detector the bot trades with: it looks at the whole range, including
how it ended, which live trading can't do. It only proposes boxes to start from. You
correct them on the labelling page (label_tool.html), and the image model learns from
the corrected boxes.

A range = a stretch of at least RANGE_MIN_BARS candles on RANGE_TF where
  * the band (5th percentile of lows to 95th percentile of highs, so single wicks like
    springs and upthrusts may poke out) is at most RANGE_MAX_WIDTH_ATR ATRs tall,
  * no candle CLOSES more than half an ATR outside the band, and
  * price visits the top zone and the bottom zone (RANGE_EDGE_ZONE of the band) at least
    RANGE_TOUCHES times each, with a trip through the middle in between.
Each range is stretched as far as those rules allow, so its end is the breakout.
"""
import json

import numpy as np
import pandas as pd

from smcml.bias import resample
from smcml.detectors import atr


def settings(cfg):
    g = lambda k, d: getattr(cfg, k, d)
    return dict(tf=g("RANGE_TF", "4h"), min_bars=g("RANGE_MIN_BARS", 30), max_width=g("RANGE_MAX_WIDTH_ATR", 7.0),
                touches=g("RANGE_TOUCHES", 2), zone=g("RANGE_EDGE_ZONE", 0.25), q=g("RANGE_WICK_QUANTILE", 0.05))


def _visits(inside):
    """Number of separate visits: runs of True."""
    if not len(inside):
        return 0
    x = inside.astype(np.int8)
    return int(x[0] + np.sum(np.diff(x) == 1))


def _ok(h, l, c, a, i, j, s):
    H, L, C = h[i:j], l[i:j], c[i:j]
    top, bot = np.quantile(H, 1 - s["q"]), np.quantile(L, s["q"])
    width = top - bot
    ref = np.nanmedian(a[i:j])
    if not np.isfinite(ref) or ref <= 0 or width <= 0 or width > s["max_width"] * ref:
        return None
    if (C > top + 0.5 * ref).any() or (C < bot - 0.5 * ref).any():
        return None
    z = s["zone"] * width
    at_top, at_bot = H >= top - z, L <= bot + z
    middle = ~(at_top | at_bot)
    # a visit only counts again after price has been back through the middle
    top_v = _visits(at_top & ~at_bot) if middle.any() else 0
    bot_v = _visits(at_bot & ~at_top) if middle.any() else 0
    if top_v < s["touches"] or bot_v < s["touches"]:
        return None
    return float(top), float(bot)


def find_ranges(df, cfg, step=3):
    """Boxes on RANGE_TF bars: list of dict(start, end, top, bottom), times = candle start."""
    s = settings(cfg)
    bars = resample(df, s["tf"])
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    a = atr(h, l, c, 14)
    n, out, i = len(bars), [], 14
    while i + s["min_bars"] <= n:
        j = i + s["min_bars"]
        box = _ok(h, l, c, a, i, j, s)
        if box is None:
            i += step
            continue
        while j < n:
            nxt = _ok(h, l, c, a, i, j + 1, s)
            if nxt is None:
                break
            box, j = nxt, j + 1
        out.append(dict(start=str(bars.index[i]), end=str(bars.index[j - 1]), top=box[0], bottom=box[1], source="auto"))
        i = j
    return out


def save_boxes(boxes, path, tf):
    with open(path, "w") as fh:
        json.dump(dict(timeframe=tf, boxes=boxes), fh, indent=1)


def load_boxes(path, with_reviewed=False):
    """(timeframe, boxes) from a labels file; with_reviewed=True also returns the time up to
    which you checked the chart on the labelling page (None if not given)."""
    with open(path) as fh:
        d = json.load(fh)
    if with_reviewed:
        return d.get("timeframe", "4h"), d["boxes"], d.get("reviewed_until")
    return d.get("timeframe", "4h"), d["boxes"]


def in_range_mask(index, boxes):
    """For each time in index: 1 if it falls inside a box (start <= t <= end of its last candle)."""
    idx = pd.DatetimeIndex(index)
    m = np.zeros(len(idx), dtype=np.int8)
    for b in boxes:
        m[(idx >= pd.Timestamp(b["start"])) & (idx <= pd.Timestamp(b["end"]))] = 1
    return m


def range_now(df, cfg, lengths=None):
    """The simple causal detector (no model): for every RANGE_TF candle, is price in a
    range judged ONLY from the candles up to and including it? Checks the last L candles
    for several L. Returns a Series (1 = in a range) indexed by candle START time.
    A base bar may use the value of the last candle that had CLOSED by its own close."""
    s = settings(cfg)
    bars = resample(df, s["tf"])
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    a = atr(h, l, c, 14)
    lengths = lengths or (s["min_bars"], int(s["min_bars"] * 1.5), s["min_bars"] * 2)
    out = np.zeros(len(bars), dtype=np.int8)
    for t in range(len(bars)):
        for L in lengths:
            if t + 1 - L >= 14 and _ok(h, l, c, a, t + 1 - L, t + 1, s) is not None:
                out[t] = 1
                break
    return pd.Series(out, index=bars.index)


def value_at_decision(series, times, base_bar, tf):
    """series (indexed by candle start) as known at the close of base bars starting at `times`:
    the last candle whose END is at or before the base bar's close."""
    ends = series.index + pd.Timedelta(tf)
    closes = pd.DatetimeIndex(times) + base_bar
    k = np.searchsorted(ends.values, closes.values, side="right") - 1
    v = np.where(k >= 0, series.to_numpy()[np.clip(k, 0, None)], 0)
    return v
