"""
LAYERS 1-3: bars -> swings -> structure -> zones.

The golden rule of this file: at bar t, nothing may look at bar t+1 or later.
Every tracker is updated one bar at a time and only reads arrays up to index t.
tests/test_no_lookahead.py checks this automatically.

Only LONG logic lives here. Shorts are handled by flipping the chart upside down
(see mirror() below), so there is exactly one copy of every rule and the long and
short versions can never drift apart.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------- helpers
def mirror(o, h, l, c, direction):
    """direction=+1: prices as they are. direction=-1: chart flipped upside down.

    In the flipped chart a short setup looks exactly like a long setup:
    new_high = -old_low, new_low = -old_high, and so on.
    Multiply a flipped price by -1 to get the real price back.
    """
    if direction == 1:
        return o, h, l, c
    return -o, -l, -h, -c


def atr(h, l, c, n):
    """Average true range (Wilder style). Uses only past and current bars."""
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


# ---------------------------------------------------------------------------- layer 1: swings
@dataclass
class Swing:
    kind: int            # +1 swing high, -1 swing low
    idx: int             # the bar where the high/low is
    price: float
    confirmed_at: int    # first bar where we can KNOW it is a swing (idx + n)
    swept_at: int = -1   # bar where price first traded beyond it (liquidity taken)
    broken_at: int = -1  # bar where price first CLOSED beyond it (structure broken)


class SwingTracker:
    """Fractal swings: a swing high is the highest high of the n bars on each side.

    The bars on the right side have to exist first, so a swing at bar i is only
    reported at bar i + n. Using it earlier than that is lookahead.
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


# ---------------------------------------------------------------------------- layer 2: structure
class StructureTracker:
    """Break of structure (BOS) and change of character (CHoCH), judged on CLOSES.

    trend = +1 after a bullish break, -1 after a bearish break, 0 at the start.
    A bullish break when the trend was not already up is a CHoCH, otherwise a BOS.
    """

    def __init__(self, swings):
        self.sw = swings
        self.trend = 0

    def update(self, t, c):
        """Returns the list of breaks that happened on bar t: [(+1/-1, "BOS"/"CHOCH", swing)]."""
        events = []
        sh, sl = self.sw.last_high, self.sw.last_low
        if sh is not None and sh.broken_at < 0 and c[t] > sh.price:
            sh.broken_at = t
            events.append((+1, "BOS" if self.trend == 1 else "CHOCH", sh))
            self.trend = 1
        if sl is not None and sl.broken_at < 0 and c[t] < sl.price:
            sl.broken_at = t
            events.append((-1, "BOS" if self.trend == -1 else "CHOCH", sl))
            self.trend = -1
        return events


# ---------------------------------------------------------------------------- layer 3: zones
def find_order_block(o, h, l, c, leg_start, lookback=10):
    """Bullish order block: the last down candle at or before the start of the up-leg.

    Returns (bar, top, bottom). Falls back to the leg-start candle itself.
    """
    for k in range(leg_start, max(leg_start - lookback, 0) - 1, -1):
        if c[k] < o[k]:
            return k, h[k], l[k]
    return leg_start, h[leg_start], l[leg_start]


def find_fvgs(h, l, start, end):
    """Bullish fair value gaps between bars start..end (inclusive).

    Three-candle gap: low of candle k is above the high of candle k-2.
    Returns a list of (bar, top, bottom).
    """
    out = []
    for k in range(max(start + 2, 2), end + 1):
        if l[k] > h[k - 2]:
            out.append((k, l[k], h[k - 2]))
    return out
