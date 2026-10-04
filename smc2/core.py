"""
THE SHARED BUILDING BLOCKS. Every indicator in smc2/ uses these, and only these, so the
same word means the same thing in every indicator (one ATR, one swing, one FVG, one way
of putting a higher timeframe on your chart).

Copied from smcml/ on purpose: changing a definition here changes the new version only,
never the old bot's results.

The golden rule: at candle t, nothing may look at candle t+1 or later.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------- prices
def mirror(o, h, l, c, direction):
    """direction=+1: prices as they are. direction=-1: chart flipped upside down.

    In the flipped chart a down move looks exactly like an up move:
    new_high = -old_low, new_low = -old_high. Multiply a flipped price by -1 to get it back.
    """
    if direction == 1:
        return o, h, l, c
    return -o, -l, -h, -c


def atr(h, l, c, n):
    """Average true range (Wilder style). Uses only past and current candles."""
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


# ---------------------------------------------------------------------------- swings
@dataclass
class Swing:
    kind: int            # +1 swing high, -1 swing low
    idx: int             # the candle where the high/low is
    price: float
    confirmed_at: int    # first candle where we can KNOW it is a swing (idx + n)


class SwingTracker:
    """Fractal swings: a swing high is the highest high of the n candles on each side.

    The candles on the right side have to exist first, so a swing at candle i is only
    reported at candle i + n. Using it earlier than that is lookahead.
    """

    def __init__(self, n):
        self.n = n
        self.last_high = None
        self.last_low = None

    def update(self, t, h, l):
        n = self.n
        i = t - n
        if i - n < 0:
            return
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:t + 1].max():
            self.last_high = Swing(+1, i, h[i], t)
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:t + 1].min():
            self.last_low = Swing(-1, i, l[i], t)


def held_low(o, l, c, k, m):
    """Candle k is a low that HELD (resting liquidity): the next m candles all made higher
    lows, and k is a pullback candle (a down candle, or a lower low than the one before).
    Known only at the close of candle k + m."""
    return l[k] < l[k + 1:k + m + 1].min() and (c[k] < o[k] or l[k] < l[k - 1])


# ---------------------------------------------------------------------------- zones
def find_fvgs(h, l, start, end):
    """Up fair value gaps between candles start..end (inclusive).

    Three-candle gap: the low of candle k is above the high of candle k-2.
    Returns a list of (candle, top, bottom).
    """
    out = []
    for k in range(max(start + 2, 2), end + 1):
        if l[k] > h[k - 2]:
            out.append((k, l[k], h[k - 2]))
    return out


# ---------------------------------------------------------------------------- timeframes
def bar_length(index):
    """Typical time between candles (e.g. 15 minutes)."""
    return pd.Series(index[:2000]).diff().median()


def resample(df, tf):
    """Higher-timeframe candles. Each is labelled with its START time."""
    out = df[["open", "high", "low", "close"]].resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def align(table, df, tf, events=(), fill=None):
    """Put a higher-timeframe table (one row per tf candle, indexed by candle START) on
    your chart: one row per candle of df, showing the last tf candle that had CLOSED by
    the close of that candle. The tf candle still forming is never used.

    Columns named in `events` happen once (e.g. "the range ended"), so they show only on
    the first candle of df that can see them, and are 0 on the others.
    Before the first tf candle has closed, a column shows fill[col] (default: NaN for
    prices, NaT for times, 0 for flags).
    """
    fill = fill or {}
    ends = (table.index + pd.Timedelta(tf)).to_numpy().astype("datetime64[ns]")
    closes = (df.index + bar_length(df.index)).to_numpy().astype("datetime64[ns]")
    k = np.searchsorted(ends, closes, side="right") - 1        # last tf candle closed by then
    ok = k >= 0
    kk = np.clip(k, 0, None)
    first = ok & (np.r_[-1, k[:-1]] != k)                        # first candle that sees row k
    out = pd.DataFrame(index=df.index)
    for col in table.columns:
        s = table[col]
        empty = fill.get(col, {"f": np.nan, "M": np.datetime64("NaT")}.get(s.dtype.kind, 0))
        v = s.to_numpy()[kk] if len(s) else np.full(len(df), empty, dtype=s.dtype)
        keep = first if col in events else ok
        out[col] = pd.Series(np.where(keep, v, 0 if col in events else empty), index=df.index).astype(s.dtype)
    return out
