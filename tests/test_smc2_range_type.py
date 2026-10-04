"""
Checks for the new version's range-type indicator (smc2/range_type.py).

  * nothing uses candles after the one being judged, on your chart and on a higher timeframe
  * every kind has the same columns, meaning the same thing
  * a drawn example of each picture switches on its own flag, starts where it should,
    and ends with the right break direction
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smcml.data import simulate  # noqa: E402
from smc2.core import resample  # noqa: E402
from smc2.range_type import range_type, range_type_bars, KINDS, FIELDS  # noqa: E402

DF = simulate(6_000, mode="random", seed=11)        # 15-minute candles from midnight: exactly 1,500 hours
FULL = range_type(DF)
H1 = range_type(DF, tf="1h")


def bars(closes, wick=0.3):
    """Candles from a list of closes: open = previous close, small wicks."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2024-01-01", periods=len(c), freq="5min", name="time")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c}, index=idx)


def quiet(n, level=100.0, seed=0):
    return level + np.random.default_rng(seed).normal(0, 0.4, n).cumsum() * 0.2


def spike_and_pause(n_pause=12):
    base = quiet(60)
    spike = base[-1] + np.array([2.5, 5.0, 7.5, 9.0])                  # 4 big candles up, gaps between them
    top = spike[-1]
    pause = top - np.resize([1.0, 2.0, 1.2, 2.5, 1.5, 2.2, 1.0, 2.4, 1.3, 2.0, 1.1, 1.8], n_pause)
    return base, spike, pause, top


# ----------------------------------------------------------------------------- no peeking
def test_only_closed_candles_are_used():
    rng = np.random.default_rng(0)
    for t in rng.choice(np.arange(500, len(DF) - 5), 6, replace=False):
        part = DF.iloc[:t + 1]
        pd.testing.assert_series_equal(range_type(part).iloc[-1], FULL.iloc[t], check_names=False)
        pd.testing.assert_series_equal(range_type(part, tf="1h").iloc[-1], H1.iloc[t], check_names=False)


def test_higher_timeframe_shows_only_closed_candles():
    native = range_type_bars(resample(DF, "1h"))
    # each M15 candle closes 15 minutes after it opens; the last H1 candle that had closed by then:
    seen = (DF.index + pd.Timedelta("15min")).floor("1h") - pd.Timedelta("1h")
    ok = seen >= native.index[0]
    ends = [f"{x}_end" for x in KINDS]
    want = native.reindex(seen[ok]).drop(columns=ends).set_axis(DF.index[ok])
    pd.testing.assert_frame_equal(H1[ok].drop(columns=ends), want)
    assert (H1["range_type"][~ok] == "none").all() and (H1["in_range"][~ok] == 0).all()
    for x in KINDS:                                                     # an end happens once, not once per M15 candle
        assert (H1[f"{x}_end"] != 0).sum() == (native[f"{x}_end"] != 0).sum(), x
        first = H1.index[H1[f"{x}_end"] != 0]
        assert ((first + pd.Timedelta("15min")) == (first + pd.Timedelta("15min")).floor("1h")).all(), x


# ----------------------------------------------------------------------------- one set of columns
def test_every_kind_has_the_same_columns():
    for k in KINDS:
        assert [k] + [f"{k}_{f}" for f in FIELDS] == [c for c in FULL.columns if c.split("_")[0] == k]
        on, off = FULL[FULL[k] == 1], FULL[FULL[k] == 0]
        assert len(on) and (on[f"{k}_top"] > on[f"{k}_bottom"]).all(), k
        assert on[f"{k}_start"].notna().all() and (on[f"{k}_start"] <= on.index).all(), k
        assert off[f"{k}_top"].isna().all() and off[f"{k}_start"].isna().all() and (off[f"{k}_swept"] == 0).all(), k
        assert (on[f"{k}_swept"] <= on[f"{k}_liq"].notna()).all(), k  # swept needs a level
        ended = FULL[f"{k}_end"] != 0                                   # an end only on the candle it switches off
        assert (FULL[k][ended] == 0).all() and (FULL[k].shift(1)[ended] == 1).all(), k
    for k in ("pause", "staircase"):
        up, dn = FULL[(FULL[k] == 1) & (FULL[f"{k}_dir"] == 1)], FULL[(FULL[k] == 1) & (FULL[f"{k}_dir"] == -1)]
        assert len(up) and len(dn), k
        assert (up[f"{k}_liq"].dropna() < up[f"{k}_top"][up[f"{k}_liq"].notna()]).all(), k   # stops sit below an up range
        assert (dn[f"{k}_liq"].dropna() > dn[f"{k}_bottom"][dn[f"{k}_liq"].notna()]).all(), k
    assert (FULL["wyckoff_dir"] == 0).all()


def test_the_answer_is_the_range_that_switched_on_last():
    both = 0
    for i in np.flatnonzero(FULL["in_range"].to_numpy() == 1)[::7]:
        row = FULL.iloc[i]
        on = [k for k in KINDS if row[k] == 1]
        since = {k: i - int(np.argmin(FULL[k].to_numpy()[i::-1])) + 1 for k in on}   # first candle of this stretch
        latest = max(on, key=lambda k: (since[k], -KINDS.index(k)))
        both += len(on) > 1
        assert row["range_type"] == latest
        for f in ("dir", "top", "bottom", "start", "liq", "swept"):
            assert row[f"range_{f}"] == row[f"{latest}_{f}"] or (pd.isna(row[f"range_{f}"]) and pd.isna(row[f"{latest}_{f}"]))
    assert both > 0                                                     # some overlaps were checked
    assert (FULL["range_type"][FULL["in_range"] == 0] == "none").all()
    assert set(FULL["range_type"]) == set(KINDS) | {"none"}


# ----------------------------------------------------------------------------- the pictures
def test_picture_1_pause_after_an_impulse():
    base, spike, pause, top = spike_and_pause()
    cont = top + np.array([1.5, 3.0, 4.5])                              # break of the pause high
    df = bars(np.r_[base, spike, pause, cont])
    r = range_type(df)
    p0 = len(base) + len(spike)
    on = r.iloc[p0 + 5:p0 + len(pause)]
    assert on["pause"].all() and (on["pause_dir"] == 1).all()          # on from the 5th sideways candle
    assert (on["pause_start"] == df.index[p0 - 1]).all()               # the box starts at the top of the spike
    assert (on["range_type"] == "pause").all()
    assert r["pause"].iloc[:p0].sum() == 0
    assert r["pause"].iloc[p0 + len(pause)] == 0 and r["pause_end"].iloc[p0 + len(pause)] == 1   # broke up
    assert (r["pause_end"] != 0).sum() == 1


def test_picture_1_failed_impulse_ends_downward():
    base, spike, pause, top = spike_and_pause(8)
    drop = top - np.array([5.0, 7.0, 8.5])                              # closes below half the spike
    r = range_type(bars(np.r_[base, spike, pause, drop]))
    p1 = len(base) + len(spike) + len(pause)
    assert r["pause"].iloc[p1 - 1] == 1
    assert r["pause"].iloc[p1] == 0 and r["pause_end"].iloc[p1] == -1


def test_picture_1_liquidity_takeout():
    base = quiet(60)
    b = base[-1]
    spike = b + np.array([2.5, 6.0, 5.6, 8.0, 9.0])                     # one down candle inside the spike
    pause = b + 9.0 - np.array([1.0, 1.6, 1.2, 1.8, 1.3, 1.7, 1.2, 1.6])
    sweep = b + np.array([4.9, 7.0, 7.8])                               # trades under that candle's low, closes above the floor
    cont = b + np.array([10.5, 12.0])
    df = bars(np.r_[base, spike, pause, sweep, cont])
    k = len(base) + 2
    df.iloc[k, df.columns.get_loc("low")] -= 0.2                        # its low: the next candles stay above it, so it HELD
    level = df["low"].iloc[k]
    r = range_type(df)
    p0 = len(base) + len(spike)
    before = r.iloc[p0 + 5:p0 + len(pause)]
    assert before["pause"].all() and (before["pause_swept"] == 0).all()
    assert np.allclose(before["pause_liq"], level)
    after = r.iloc[p0 + len(pause):p0 + len(pause) + len(sweep)]
    assert after["pause"].all() and (after["pause_swept"] == 1).all() and (after["range_swept"] == 1).all()
    assert np.allclose(after["pause_liq"], level)
    assert r["pause_end"].iloc[p0 + len(pause) + len(sweep)] == 1       # then the break ends the pause, upward


def test_picture_2_directionless_range_ends_on_a_close_outside():
    base = quiet(40)
    lvl = base[-1]
    wave = lvl + 1.6 * np.sin(np.arange(70) * 2 * np.pi / 14)           # up and down between two edges
    out = lvl + np.array([3.5, 5.0, 6.5, 8.0])                          # closes far above the box
    df = bars(np.r_[base, wave, out])
    r = range_type(df)
    w0, w1 = len(base), len(base) + len(wave)
    on = r.iloc[w1 - 10:w1]
    assert on["wyckoff"].all() and (on["wyckoff_dir"] == 0).all() and on["wyckoff_liq"].isna().all()
    first_on = int(np.flatnonzero(r["wyckoff"].to_numpy())[0])
    assert (on["wyckoff_start"] <= df.index[first_on - 29]).all()       # the window it was found in (30+ candles)
    assert r["wyckoff"].iloc[w1 + 1:].sum() == 0
    assert r["wyckoff_end"].iloc[w1:w1 + 2].tolist() in ([1, 0], [0, 1])   # broke up, once
    assert r["wyckoff"].iloc[:w0 + 20].sum() == 0


def staircase_path(cycles=5):
    base = quiet(40)
    path = [base[-1]]
    for k in range(cycles):                                             # up 6, back 4.5: each new high adds 1.5
        up = np.linspace(path[-1], path[-1] + 6, 7)[1:]
        dn = np.linspace(up[-1], up[-1] - 4.5, 7)[1:]
        path += list(up) + list(dn)
    return base, path


def test_picture_3_staircase():
    base, path = staircase_path()
    brk = path[-1] + np.array([4.0, 8.0, 12.0, 16.0])                   # a real displacement out of it
    df = bars(np.r_[base, path[1:], brk], wick=0.1)
    r = range_type(df)
    s0, s1 = len(base), len(base) + len(path) - 1
    on = r[r["staircase"] == 1]
    assert r["staircase"].iloc[s0:s1].sum() > 10
    assert (on["staircase_dir"] == 1).all()
    assert r["staircase"].iloc[s0:s0 + 24].sum() == 0                   # needs 2 full cycles first
    assert (on["staircase_start"] >= df.index[s0]).all()                # starts at a low inside the staircase
    assert (on["staircase_liq"] >= on["staircase_bottom"]).all() and (on["staircase_liq"] < on["staircase_top"]).all()
    assert r["staircase"].iloc[-2:].sum() == 0                          # off after the break
    assert (r["staircase_end"] == 1).sum() == 1 and (r["staircase_end"] == -1).sum() == 0


def test_picture_3_wick_below_the_last_low_is_the_takeout():
    base, path = staircase_path()
    df = bars(np.r_[base, path[1:]], wick=0.1)
    r = range_type(df)
    i = int(np.flatnonzero(r["staircase"].to_numpy())[-1])              # last candle it is on
    liq = r["staircase_liq"].iloc[i]
    assert r["staircase_swept"].iloc[i] == 0
    hit = df.copy()
    hit.iloc[i, hit.columns.get_loc("low")] = liq - 0.05                # a wick under the last swing low, close unchanged
    r2 = range_type(hit)
    assert r2["staircase"].iloc[i] == 1 and r2["staircase_swept"].iloc[i] == 1
    assert r2["staircase_liq"].iloc[i] == liq


def test_a_clean_trend_is_not_a_staircase():
    base = quiet(40)
    path = [base[-1]]
    for k in range(5):                                                  # up 6, back 1.5: shallow pullbacks
        up = np.linspace(path[-1], path[-1] + 6, 7)[1:]
        dn = np.linspace(up[-1], up[-1] - 1.5, 4)[1:]
        path += list(up) + list(dn)
    r = range_type(bars(np.r_[base, path[1:]], wick=0.1))
    assert r["staircase"].iloc[len(base):].sum() == 0
