"""
THREE KINDS OF RANGE, each judged ONLY from candles that have already closed.

    from wyckoff.range_types import range_types
    r = range_types(df)            # df = your bars (M1, M5, M15 ...), one row per candle
    r = range_types(df, tf="1h")   # or build H1 candles from them first

Row t says what was known at the CLOSE of candle t. Columns (0/1 flags, plus each box):

  pause      the pause after an impulse (your picture 1)
             price spikes (a leg of IMPULSE_ATR ATRs or more in a few candles, leaving a
             fair value gap or one huge body), then goes sideways in the upper part of
             that leg. It never closes back below PAUSE_MAX_RETRACE of the leg.
             Ends when a candle closes above the pause high (the break of structure),
             or closes below that floor (the impulse failed).

  wyckoff    the directionless range (your picture 2)
             a band at most WY_WIDTH_K x sqrt(candles) ATRs tall that price has crossed from one
             edge to the other at least WY_CROSSINGS times, with no candle closing outside it.
             Wicks (springs, upthrusts) may poke out. Once found, the box is FROZEN and
             the flag stays on until a candle closes more than half an ATR outside it.

  staircase  the zig-zag range (your picture 3)
             structure keeps breaking to a new high, but each new high is only a small
             step (at most STAIR_STEP of the leg that made it) and price then gives back
             at least STAIR_RETRACE of that leg, back to the order block. STAIR_CYCLES
             of those in a row. Ends when a candle closes far beyond the last high
             (a real displacement), or closes below the last swing low.

  in_range   1 if any of the three is on.

pause_dir / staircase_dir: +1 = the impulse or the staircase points up, -1 = down.
Down versions are found by flipping the chart (mirror), like everywhere else in the bot.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smcml.bias import resample
from smcml.detectors import atr, mirror, find_fvgs, SwingTracker


@dataclass
class Settings:
    ATR_N: int = 14
    ATR_SLOW_N: int = 100             # a slow ATR, so a quiet hour does not make every ordinary move a "spike"
    # ---- picture 1: pause after an impulse
    IMPULSE_BARS: int = 12            # the spike happens within this many candles
    IMPULSE_ATR: float = 6.0          # and is at least this many ATRs tall (the larger of ATR and slow ATR)
    IMPULSE_BODY_ATR: float = 1.5     # "huge imbalance": one body this big counts if there is no FVG
    PAUSE_MAX_RETRACE: float = 0.5    # the pause may give back at most this share of the spike (on closes)
    PAUSE_MIN_BARS: int = 5           # sideways for at least this many candles before it is called a pause
    PAUSE_MAX_BARS: int = 80          # after this long it is no longer "the pause after that spike"
    # ---- picture 2: directionless range
    WY_LENGTHS: tuple = (30, 45, 60)  # window lengths tried, in candles
    WY_WIDTH_K: float = 0.5           # band at most WY_WIDTH_K x sqrt(length) ATRs tall. A random walk covers
                                      # about sqrt(length) ATRs in that time, so 0.5 = "half the ground
                                      # a directionless coin-flip market would have covered"
    WY_CROSSINGS: int = 4             # price went from one edge zone to the other at least this many times
                                      # (top, bottom, top, bottom, top = 4). A single V-shape is only 2.
    WY_ZONE: float = 0.25             # top/bottom zone = 25% of the band
    WY_WICK_Q: float = 0.05           # band = 5th percentile of lows to 95th percentile of highs
    WY_BREAK_ATR: float = 0.5         # a close this far outside the box ends it
    # ---- picture 3: staircase
    STAIR_SWING_N: int = 3            # swing = highest high with 3 candles on each side
    STAIR_STEP: float = 0.5           # each new high adds at most this share of the leg that made it
    STAIR_RETRACE: float = 0.5        # and price then gives back at least this share of that leg
    STAIR_CYCLES: int = 2             # this many in a row


# ----------------------------------------------------------------------------- picture 1
def _pause(o, h, l, c, a, a_slow, s):
    """Up version. Returns flag, top, bottom arrays."""
    n = len(c)
    flag = np.zeros(n, np.int8)
    top, bot = np.full(n, np.nan), np.full(n, np.nan)
    live = False
    for t in range(n):
        if live:
            if c[t] > hi:                                   # break of the pause high: the move continues
                live = False                                # (this candle may itself be a new impulse, see below)
            elif c[t] < hi - s.PAUSE_MAX_RETRACE * (hi - leg_low) or t - hi_bar > s.PAUSE_MAX_BARS:
                live = False                                # gave back too much, or went on too long
                continue
            else:
                if h[t] > hi:                               # a wick above without a close: the box grows
                    hi = h[t]
                low_since = min(low_since, l[t])
                if t - hi_bar >= s.PAUSE_MIN_BARS:
                    flag[t], top[t], bot[t] = 1, hi, low_since
                continue
        # ---- is this candle the top of a fresh impulse?
        j0 = max(t - s.IMPULSE_BARS + 1, 1)
        if t < j0 + 1:
            continue
        j = j0 + int(np.argmin(l[j0:t + 1]))                # where the spike started
        ref = max(a[j - 1], a_slow[j - 1])                  # ATR BEFORE the spike (the spike itself inflates ATR)
        if not np.isfinite(ref) or ref <= 0 or j == t:
            continue
        if h[t] < h[j:t + 1].max() or h[t] - l[j] < s.IMPULSE_ATR * ref:
            continue
        gaps = find_fvgs(h, l, j - 1, t)
        bodies = np.abs(c[j:t + 1] - o[j:t + 1]).max()
        if not gaps and bodies < s.IMPULSE_BODY_ATR * ref:
            continue
        live, hi, hi_bar, leg_low, low_since = True, h[t], t, l[j], np.inf   # the box bottom = lowest low AFTER the top candle
    return flag, top, bot


# ----------------------------------------------------------------------------- picture 2
def _box(h, l, c, a, i, j, s):
    """Do candles i..j-1 form a directionless box? Returns (top, bottom) or None."""
    H, L, C = h[i:j], l[i:j], c[i:j]
    top, bot = np.quantile(H, 1 - s.WY_WICK_Q), np.quantile(L, s.WY_WICK_Q)
    width, ref = top - bot, np.nanmedian(a[i:j])
    if not np.isfinite(ref) or ref <= 0 or width <= 0 or width > s.WY_WIDTH_K * (j - i) ** 0.5 * ref:
        return None
    if (C > top + s.WY_BREAK_ATR * ref).any() or (C < bot - s.WY_BREAK_ATR * ref).any():
        return None
    z = s.WY_ZONE * width
    side = (H >= top - z).astype(int) - (L <= bot + z).astype(int)      # +1 at the top, -1 at the bottom
    seq = side[side != 0]
    if (np.diff(seq) != 0).sum() < s.WY_CROSSINGS:                       # edge-to-edge trips
        return None
    return float(top), float(bot)


def _wyckoff(h, l, c, a, s):
    n = len(c)
    flag = np.zeros(n, np.int8)
    top, bot = np.full(n, np.nan), np.full(n, np.nan)
    box = None
    for t in range(n):
        if box is not None:                                 # frozen box: only a close outside ends it
            if c[t] > box[0] + s.WY_BREAK_ATR * a[t] or c[t] < box[1] - s.WY_BREAK_ATR * a[t]:
                box = None
            else:
                flag[t], top[t], bot[t] = 1, box[0], box[1]
            continue
        for L in s.WY_LENGTHS:
            if t + 1 - L >= s.ATR_N:
                found = _box(h, l, c, a, t + 1 - L, t + 1, s)
                if found is not None:
                    box = found
                    flag[t], top[t], bot[t] = 1, box[0], box[1]
                    break
    return flag, top, bot


# ----------------------------------------------------------------------------- picture 3
def _staircase(h, l, c, s):
    """Up version: small new highs, each followed by a deep pullback."""
    n = len(c)
    flag = np.zeros(n, np.int8)
    top, bot = np.full(n, np.nan), np.full(n, np.nan)
    sw = SwingTracker(s.STAIR_SWING_N)
    zz = []                    # confirmed swings in order, alternating: (kind, bar, price)
    seen_h = seen_l = -1

    def add(kind, idx, price):
        if zz and zz[-1][0] == kind:                        # two highs in a row: keep the higher one
            if (kind == 1 and price > zz[-1][2]) or (kind == -1 and price < zz[-1][2]):
                zz[-1] = (kind, idx, price)
        else:
            zz.append((kind, idx, price))

    for t in range(n):
        sw.update(t, h, l)
        new = []
        if sw.last_high is not None and sw.last_high.idx != seen_h:
            seen_h = sw.last_high.idx
            new.append((1, seen_h, sw.last_high.price))
        if sw.last_low is not None and sw.last_low.idx != seen_l:
            seen_l = sw.last_low.idx
            new.append((-1, seen_l, sw.last_low.price))
        for k in sorted(new, key=lambda x: x[1]):
            add(*k)

        # ---- count the qualifying cycles at the end of the zig-zag
        # one cycle = prev high, low L, high H, low L2:  H is a small step above the prev
        # high, and L2 gives back at least STAIR_RETRACE of the leg L -> H without breaking L
        i = len(zz) - 1
        if i >= 0 and zz[i][0] == 1:                        # the newest swing is a high: it must be a small step too
            if i >= 2:
                step, leg = zz[i][2] - zz[i - 2][2], zz[i][2] - zz[i - 1][2]
                if not (0 < step <= s.STAIR_STEP * leg):
                    continue
            i -= 1
        cycles, first_low = 0, None
        while i >= 3:
            L2, H, L, Hp = zz[i][2], zz[i - 1][2], zz[i - 2][2], zz[i - 3][2]
            leg = H - L
            if leg <= 0 or not (0 < H - Hp <= s.STAIR_STEP * leg) or not (s.STAIR_RETRACE * leg <= H - L2 <= leg):
                break
            cycles, first_low = cycles + 1, L
            i -= 2
        if cycles < s.STAIR_CYCLES:
            continue
        last_high = max(z[2] for z in zz[-2:] if z[0] == 1)
        last_low_pt = [z for z in zz[-2:] if z[0] == -1][0]
        last_leg = last_high - last_low_pt[2]
        seg_low = c[last_low_pt[1]:t + 1].min()             # no candle since the last swing low CLOSED below it
        if c[t] > last_high + s.STAIR_STEP * last_leg or seg_low < last_low_pt[2]:
            zz.clear()                                      # a real break, up or down: start counting again
            continue
        flag[t], top[t], bot[t] = 1, max(last_high, h[last_low_pt[1]:t + 1].max()), first_low
    return flag, top, bot


# ----------------------------------------------------------------------------- all three
def range_types(df, tf=None, settings=None):
    """One row per candle: the three flags, their boxes, and in_range.

    tf=None uses your bars as they are. tf="1h" (or "4h" ...) builds those candles first;
    the rows are then indexed by candle START time, and a setup on a smaller chart may only
    use the row of the last candle that had CLOSED (wyckoff.ranges.value_at_decision does that).
    """
    s = settings or Settings()
    bars = resample(df, tf) if tf else df
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a, a_slow = atr(h, l, c, s.ATR_N), atr(h, l, c, s.ATR_SLOW_N)
    out = pd.DataFrame(index=bars.index)

    for name, fn in (("pause", lambda O, H, L, C: _pause(O, H, L, C, a, a_slow, s)),
                     ("staircase", lambda O, H, L, C: _staircase(H, L, C, s))):
        up = fn(*mirror(o, h, l, c, +1))
        dn = fn(*mirror(o, h, l, c, -1))
        is_up = up[0] == 1
        out[name] = (up[0] | dn[0]).astype(np.int8)
        out[f"{name}_dir"] = np.where(is_up, 1, np.where(dn[0] == 1, -1, 0)).astype(np.int8)
        out[f"{name}_top"] = np.where(is_up, up[1], -dn[2])       # flipped chart: its bottom is the real top
        out[f"{name}_bottom"] = np.where(is_up, up[2], -dn[1])

    wf, wt, wb = _wyckoff(h, l, c, a, s)
    out["wyckoff"], out["wyckoff_top"], out["wyckoff_bottom"] = wf, wt, wb
    out["in_range"] = (out[["pause", "wyckoff", "staircase"]].sum(axis=1) > 0).astype(np.int8)
    return out
