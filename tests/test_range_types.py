"""
Checks for the three range types (wyckoff/range_types.py).

  * nothing uses candles after the one being judged
  * a drawn example of each picture switches on its own flag, and the break switches it off
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smcml.data import simulate  # noqa: E402
from wyckoff.range_types import range_types, range_features, RANGE_FEATURES  # noqa: E402

DF = simulate(6_000, mode="random", seed=11)
FULL = range_types(DF)


def bars(closes, wick=0.3):
    """Candles from a list of closes: open = previous close, small wicks."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2024-01-01", periods=len(c), freq="5min", name="time")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c}, index=idx)


def quiet(n, level=100.0, seed=0):
    return level + np.random.default_rng(seed).normal(0, 0.4, n).cumsum() * 0.2


def test_only_closed_candles_are_used():
    rng = np.random.default_rng(0)
    for t in rng.choice(np.arange(500, len(DF) - 5), 8, replace=False):
        part = range_types(DF.iloc[:t + 1])
        pd.testing.assert_series_equal(part.iloc[-1], FULL.iloc[t], check_names=False)


def test_flags_fire_on_random_prices_but_not_all_the_time():
    for k in ("pause", "wyckoff", "staircase"):
        assert 0 < FULL[k].mean() < 0.6, k
    on = FULL[FULL["wyckoff"] == 1]
    assert (on["wyckoff_top"] > on["wyckoff_bottom"]).all()


def test_picture_1_pause_after_an_impulse():
    base = quiet(60)
    spike = base[-1] + np.array([2.5, 5.0, 7.5, 9.0])                  # 4 big candles up, gaps between them
    top = spike[-1]
    pause = top - np.array([1.0, 2.0, 1.2, 2.5, 1.5, 2.2, 1.0, 2.4, 1.3, 2.0, 1.1, 1.8])
    cont = top + np.array([1.5, 3.0, 4.5])                              # break of the pause high
    r = range_types(bars(np.r_[base, spike, pause, cont]))
    p0 = len(base) + len(spike)
    assert r["pause"].iloc[p0 + 5:p0 + len(pause)].all()               # on from the 5th sideways candle
    assert (r["pause_dir"].iloc[p0 + 5:p0 + len(pause)] == 1).all()
    assert r["pause"].iloc[:p0].sum() == 0                              # not before the spike ended
    assert r["pause"].iloc[p0 + len(pause)] == 0                        # off on the break


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
    r = range_types(df)
    p0 = len(base) + len(spike)
    before = r.iloc[p0 + 5:p0 + len(pause)]
    assert before["pause"].all() and (before["pause_swept"] == 0).all()
    assert np.allclose(before["pause_liq"], level)                      # the resting low is known before it is taken
    after = r.iloc[p0 + len(pause):p0 + len(pause) + len(sweep)]
    assert after["pause"].all() and (after["pause_swept"] == 1).all()
    assert np.allclose(after["pause_liq"], level)
    assert r["pause"].iloc[p0 + len(pause) + len(sweep)] == 0           # then the break ends the pause


def test_picture_1_is_cancelled_when_the_spike_is_given_back():
    base = quiet(60)
    spike = base[-1] + np.array([2.5, 5.0, 7.5, 9.0])
    fail = spike[-1] - np.array([1.0, 2.0, 3.5, 5.0, 7.0, 8.0, 8.5, 8.0, 8.5, 8.2])
    r = range_types(bars(np.r_[base, spike, fail]))
    assert r["pause"].iloc[len(base) + len(spike) + 4:].sum() == 0


def test_picture_2_directionless_range_stays_on_until_a_close_outside():
    base = quiet(40)
    lvl = base[-1]
    wave = lvl + 1.6 * np.sin(np.arange(70) * 2 * np.pi / 14)           # up and down between two edges
    out = lvl + np.array([3.5, 5.0, 6.5, 8.0])                          # closes far above the box
    r = range_types(bars(np.r_[base, wave, out]))
    w0, w1 = len(base), len(base) + len(wave)
    assert r["wyckoff"].iloc[w1 - 10:w1].all()
    assert r["wyckoff"].iloc[w1 + 1:].sum() == 0
    assert r["wyckoff"].iloc[:w0 + 20].sum() == 0                       # needs enough candles first


def test_picture_3_staircase():
    base = quiet(40)
    lvl = base[-1]
    path = [lvl]
    for k in range(5):                                                  # up 6, back 4.5: each new high adds 1.5
        up = np.linspace(path[-1], path[-1] + 6, 7)[1:]
        dn = np.linspace(up[-1], up[-1] - 4.5, 7)[1:]
        path += list(up) + list(dn)
    brk = path[-1] + np.array([4.0, 8.0, 12.0, 16.0])                   # a real displacement out of it
    r = range_types(bars(np.r_[base, path[1:], brk], wick=0.1))
    s0, s1 = len(base), len(base) + len(path) - 1
    assert r["staircase"].iloc[s0:s1].sum() > 10
    assert (r["staircase_dir"][r["staircase"] == 1] == 1).all()
    assert r["staircase"].iloc[s0:s0 + 24].sum() == 0                   # needs 2 full cycles first
    assert r["staircase"].iloc[-2:].sum() == 0                          # off after the break


def test_a_clean_trend_is_not_a_staircase():
    base = quiet(40)
    path = [base[-1]]
    for k in range(5):                                                  # up 6, back 1.5: shallow pullbacks
        up = np.linspace(path[-1], path[-1] + 6, 7)[1:]
        dn = np.linspace(up[-1], up[-1] - 1.5, 4)[1:]
        path += list(up) + list(dn)
    r = range_types(bars(np.r_[base, path[1:]], wick=0.1))
    assert r["staircase"].iloc[len(base):].sum() == 0


def test_model_features_use_only_closed_candles():
    cands = pd.DataFrame({"t_place": [900, 2500, 2501, 4000, 5200, 5990],
                          "direction": [1, -1, 1, 1, -1, 1]})
    cands["entry"] = DF["close"].to_numpy()[cands["t_place"]]
    full = range_features(cands, DF)
    assert list(full.columns) == RANGE_FEATURES
    assert full.filter(like="rg_age").ge(0).all().all()
    for T in (2600, 4001, 5500):                                        # cut the data right after some of them
        keep = cands[cands["t_place"] < T]
        part = range_features(keep, DF.iloc[:T])
        pd.testing.assert_frame_equal(part, full.loc[keep.index])


def test_deep_cnn_trains_and_scores():
    import pytest
    pytest.importorskip("torch")
    from smcml.model import DeepCNNModel
    rng = np.random.default_rng(0)
    img = (rng.random((40, 2, 48, 144)) > 0.9).astype(np.uint8) * 255
    X, y = rng.normal(size=(40, 6)).astype(np.float32), (np.arange(40) % 4 == 0).astype(int)
    m = DeepCNNModel(epochs=1, width=8).fit(X, y, img)
    assert sum(k.__class__.__name__ == "Conv2d" and k.kernel_size != (1, 1) for k in m.net.modules()) == 17
    s = m.score(X, img)
    assert s.shape == (40,) and np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
