"""
RANGE TYPE: one indicator, three kinds of range, judged ONLY from candles that have closed.

    from smc2.range_type import range_type
    r = range_type(df)              # df = your bars (M1, M5, M15 ...), one row per candle
    r = range_type(df, tf="1h")     # judged on H1 candles, shown on your chart

One row per candle of df. Row t = what was known at the CLOSE of candle t. With tf, each
row shows the last tf candle that had closed by then (the forming one is never used).

THE ANSWER (the range price is in now):
  range_type     "pause", "wyckoff", "staircase" or "none"
                 if two are on at once, the one that switched on most recently
  range_dir      +1 / -1 / 0     range_top, range_bottom     range_start
  range_liq      range_swept     in_range (1 if any of the three is on)

THE SAME EIGHT COLUMNS FOR EACH KIND (pause_*, wyckoff_*, staircase_*):
  <kind>         1 while it is on
  <kind>_dir     +1 = points up, -1 = points down, 0 = no direction (Wyckoff)
  <kind>_top     the box, in real prices
  <kind>_bottom
  <kind>_start   time of the first candle of the box
  <kind>_liq     the resting liquidity on the far side (stops sit there); NaN = none known
  <kind>_swept   1 once a candle has traded beyond <kind>_liq while it is still on
  <kind>_end     on the candle it switches off: +1 = it broke out upward, -1 = downward,
                 0 = it faded or ran out of time without a break. 0 on every other candle.

THE THREE KINDS (described for the up version; down = the chart flipped):

  pause      the pause after an impulse (your picture 1)
             price spikes (a leg of IMPULSE_ATR ATRs or more in a few candles, leaving a
             fair value gap or one huge body), then goes sideways in the upper part of
             that leg. It never closes back below PAUSE_MAX_RETRACE of the leg.
             start: the top candle of the spike. On from PAUSE_MIN_BARS candles after it.
             liq: a low that HELD inside the spike or the pause (core.held_low); before the
             takeout the nearest one, after it the one that was taken.
             end: +1 on a close above the pause high (the break of structure), -1 on a close
             below the floor (the impulse failed), 0 after PAUSE_MAX_BARS.
             The usual order is spike, pause, takeout, break.

  wyckoff    the directionless range (your picture 2)
             a band at most WY_WIDTH_K x sqrt(candles) ATRs tall that price has crossed from
             one edge to the other at least WY_CROSSINGS times, with no candle closing
             outside it. Wicks (springs, upthrusts) may poke out. Once found, the box is
             FROZEN. start: the first candle of the window it was found in.
             liq: none yet (it has no direction, so there is no single far side).
             end: a close more than WY_BREAK_ATR ATRs above (+1) or below (-1) the box.

  staircase  the zig-zag range (your picture 3)
             structure keeps breaking to a new high, but each new high is only a small
             step (at most STAIR_STEP of the leg that made it) and price then gives back
             at least STAIR_RETRACE of that leg, back to the order block. STAIR_CYCLES
             of those in a row. start: the low the first counted cycle starts from.
             liq: the last swing low (closing below it ends the staircase, so a wick
             below it is the takeout).
             end: +1 on a close far beyond the last high (more than STAIR_STEP of the last
             leg: a real displacement), -1 on a close below the last swing low, 0 when the
             zig-zag stops qualifying (e.g. a new high that is too big a step).

Same detection as wyckoff/range_types.py (the flags, boxes and pause liquidity are
identical); this version adds the shared columns and puts every timeframe on your chart.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import atr, mirror, find_fvgs, held_low, SwingTracker, resample, align

KINDS = ("pause", "wyckoff", "staircase")
FIELDS = ("dir", "top", "bottom", "start", "liq", "swept", "end")


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
    PAUSE_LIQ_HOLD: int = 5           # a low is resting liquidity once this many candles have stayed above it
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


def _empty(n):
    """The shared columns for one kind, as arrays (start = candle number, -1 = none)."""
    return dict(on=np.zeros(n, np.int8), top=np.full(n, np.nan), bottom=np.full(n, np.nan),
                start=np.full(n, -1), liq=np.full(n, np.nan), swept=np.zeros(n, np.int8),
                end=np.zeros(n, np.int8))


# ----------------------------------------------------------------------------- picture 1
def _pause(o, h, l, c, a, a_slow, s):
    """Up version."""
    n = len(c)
    r = _empty(n)
    live, m = False, s.PAUSE_LIQ_HOLD
    for t in range(n):
        if live:
            was_on = r["on"][t - 1] == 1
            floor = hi - s.PAUSE_MAX_RETRACE * (hi - leg_low)
            if c[t] > hi:                                   # break of the pause high: the move continues
                live = False                                # (this candle may itself be a new impulse, see below)
                if was_on:
                    r["end"][t] = 1
            elif c[t] < floor or t - hi_bar > s.PAUSE_MAX_BARS:
                live = False                                # gave back too much, or went on too long
                if was_on and c[t] < floor:
                    r["end"][t] = -1
                continue
            else:
                if h[t] > hi:                               # a wick above without a close: the box grows
                    hi = h[t]
                low_since = min(low_since, l[t])
                taken = [x for x in levels if l[t] < x]     # resting lows this candle traded below
                if taken:
                    if not swept:
                        swept, liq_taken = True, max(taken)  # the first one taken is the one that counts
                    levels = [x for x in levels if l[t] >= x]
                if t - m > leg_bar and held_low(o, l, c, t - m, m):   # a new resting low, usable from the next candle
                    levels.append(l[t - m])
                if t - hi_bar >= s.PAUSE_MIN_BARS:
                    r["on"][t], r["top"][t], r["bottom"][t], r["start"][t] = 1, hi, low_since, hi_bar
                    r["swept"][t] = swept
                    r["liq"][t] = liq_taken if swept else (max(levels) if levels else np.nan)
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
        leg_bar, swept, liq_taken = j, False, np.nan
        levels = [l[k] for k in range(j + 1, t - m + 1)    # lows inside the spike that held and are still untouched
                  if held_low(o, l, c, k, m) and l[k] < l[k + 1:t + 1].min()]
    return r


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
    """No up or down version: the box has no direction."""
    n = len(c)
    r = _empty(n)
    box = None
    for t in range(n):
        if box is not None:                                 # frozen box: only a close outside ends it
            if c[t] > box[0] + s.WY_BREAK_ATR * a[t]:
                box, r["end"][t] = None, 1
            elif c[t] < box[1] - s.WY_BREAK_ATR * a[t]:
                box, r["end"][t] = None, -1
            else:
                r["on"][t], r["top"][t], r["bottom"][t], r["start"][t] = 1, box[0], box[1], first
            continue
        for L in s.WY_LENGTHS:
            if t + 1 - L >= s.ATR_N:
                found = _box(h, l, c, a, t + 1 - L, t + 1, s)
                if found is not None:
                    box, first = found, t + 1 - L
                    r["on"][t], r["top"][t], r["bottom"][t], r["start"][t] = 1, box[0], box[1], first
                    break
    return r


# ----------------------------------------------------------------------------- picture 3
def _staircase(h, l, c, s):
    """Up version: small new highs, each followed by a deep pullback."""
    n = len(c)
    r = _empty(n)
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
        was_on = t > 0 and r["on"][t - 1] == 1
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
        cycles, first = 0, None
        while i >= 3:
            L2, H, L, Hp = zz[i][2], zz[i - 1][2], zz[i - 2][2], zz[i - 3][2]
            leg = H - L
            if leg <= 0 or not (0 < H - Hp <= s.STAIR_STEP * leg) or not (s.STAIR_RETRACE * leg <= H - L2 <= leg):
                break
            cycles, first = cycles + 1, zz[i - 2]
            i -= 2
        if cycles < s.STAIR_CYCLES:
            continue
        last_high = max(z[2] for z in zz[-2:] if z[0] == 1)
        last_low_pt = [z for z in zz[-2:] if z[0] == -1][0]
        last_leg = last_high - last_low_pt[2]
        seg_low = c[last_low_pt[1]:t + 1].min()             # no candle since the last swing low CLOSED below it
        if c[t] > last_high + s.STAIR_STEP * last_leg or seg_low < last_low_pt[2]:
            if was_on:
                r["end"][t] = 1 if c[t] > last_high + s.STAIR_STEP * last_leg else -1
            zz.clear()                                      # a real break, up or down: start counting again
            continue
        r["on"][t], r["top"][t], r["bottom"][t] = 1, max(last_high, h[last_low_pt[1]:t + 1].max()), first[2]
        r["start"][t], r["liq"][t] = first[1], last_low_pt[2]
        r["swept"][t] = int(l[last_low_pt[1] + 1:t + 1].min(initial=np.inf) < last_low_pt[2])
    return r


# ----------------------------------------------------------------------------- one indicator
def _on_since(on):
    """For each candle: the candle this stretch of 'on' began, -1 while off."""
    idx = np.arange(len(on))
    began = np.where((on == 1) & (np.r_[0, on[:-1]] == 0), idx, -1)
    return np.where(on == 1, np.maximum.accumulate(began), -1)


def _both_ways(fn, o, h, l, c):
    """Run an up-only detector on the chart and on the flipped chart; back to real prices."""
    up, dn = fn(*mirror(o, h, l, c, +1)), fn(*mirror(o, h, l, c, -1))
    is_up = up["on"] == 1                                   # (both on at once: the up one is shown)
    is_dn = ~is_up & (dn["on"] == 1)
    pick = lambda u, d: np.where(is_up, u, np.where(is_dn, d, u))
    on = (up["on"] | dn["on"]).astype(np.int8)
    end = np.where(up["end"] != 0, up["end"], -dn["end"])   # flipped "up" is a real break down
    return dict(
        on=on,
        dir=np.where(is_up, 1, np.where(is_dn, -1, 0)).astype(np.int8),
        top=pick(up["top"], -dn["bottom"]),                 # flipped chart: its bottom is the real top
        bottom=pick(up["bottom"], -dn["top"]),
        start=pick(up["start"], dn["start"]),
        liq=pick(up["liq"], -dn["liq"]),
        swept=pick(up["swept"], dn["swept"]).astype(np.int8),
        end=np.where(on == 0, end, 0).astype(np.int8),      # only when the kind as a whole switches off
    )


def range_type_bars(bars, settings=None):
    """The indicator on exactly these candles (no resampling). One row per candle."""
    s = settings or Settings()
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a, a_slow = atr(h, l, c, s.ATR_N), atr(h, l, c, s.ATR_SLOW_N)
    found = {
        "pause": _both_ways(lambda O, H, L, C: _pause(O, H, L, C, a, a_slow, s), o, h, l, c),
        "wyckoff": dict(_wyckoff(h, l, c, a, s), dir=np.zeros(len(c), np.int8)),
        "staircase": _both_ways(lambda O, H, L, C: _staircase(H, L, C, s), o, h, l, c),
    }
    out = pd.DataFrame(index=bars.index)
    times = bars.index.to_numpy().astype("datetime64[ns]")
    for k in KINDS:
        f = found[k]
        out[k] = f["on"].astype(np.int8)
        for field in FIELDS:
            v = f[field]
            if field == "start":
                v = np.where((f["on"] == 1) & (v >= 0), times[np.clip(v, 0, None)], np.datetime64("NaT"))
            out[f"{k}_{field}"] = v

    # ---- the answer: the kind that switched on most recently (a tie goes to the first in KINDS)
    since = np.stack([_on_since(out[k].to_numpy()) for k in KINDS])
    best = np.where(since.max(axis=0) < 0, -1, np.argmax(since, axis=0))
    names = np.array(KINDS + ("none",), dtype=object)
    out["range_type"] = names[best]
    for field in ("dir", "top", "bottom", "start", "liq", "swept"):
        cols = np.stack([out[f"{k}_{field}"].to_numpy() for k in KINDS])
        empty = {"top": np.nan, "bottom": np.nan, "liq": np.nan, "start": np.datetime64("NaT")}.get(field, 0)
        v = np.where(best >= 0, cols[np.clip(best, 0, None), np.arange(len(out))], empty)
        out[f"range_{field}"] = v.astype(out[f"{KINDS[0]}_{field}"].dtype)
    out["in_range"] = (best >= 0).astype(np.int8)
    return out


def range_type(df, tf=None, settings=None):
    """One row per candle of df. tf=None: judged on your candles. tf="1h", "4h" ...: judged
    on those candles, each row showing the last one that had CLOSED by the close of the row."""
    if not tf:
        return range_type_bars(df, settings)
    table = range_type_bars(resample(df, tf), settings)
    return align(table, df, tf, events=[f"{k}_end" for k in KINDS], fill={"range_type": "none"})
