"""
TOP-DOWN BIAS: D1 -> H4 -> H1 must all point the same way before a setup may trade.

For each higher timeframe:
  1. build its candles from your M15 bars (D1 candle = the 96 M15 bars of that day, etc.)
  2. run the SAME swing + structure logic on those candles
  3. bias = direction of the last structure break on that timeframe
       bullish (+1) = last break was a CLOSE above a swing high (BOS or CHoCH up)
       bearish (-1) = last break was a CLOSE below a swing low
       0            = not enough history yet

No peeking: at the close of an M15 bar, only higher-timeframe candles that have
ALREADY CLOSED are used. The H4 candle that is still forming does not count, just
as on a live chart where it could still change. tests/test_no_lookahead.py checks it.
"""
import numpy as np
import pandas as pd

from .detectors import SwingTracker, StructureTracker

NAMES = {"1D": "d1", "1d": "d1", "D": "d1", "4h": "h4", "4H": "h4", "1h": "h1", "1H": "h1",
         "30min": "m30", "15min": "m15"}


def tf_name(tf):
    return NAMES.get(tf, tf.lower())


def bar_length(index):
    """Typical time between bars (e.g. 15 minutes)."""
    return pd.Series(index[:2000]).diff().median()


def resample(df, tf):
    """Higher-timeframe candles. Each is labelled with its START time."""
    out = df[["open", "high", "low", "close"]].resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def htf_bias(df, tf, swing_n):
    """Bias on timeframe tf, as known at the CLOSE of every base bar.

    Returns a DataFrame aligned with df: trend (+1/-1/0), the level of the swing whose
    break set the trend, and the start time of the candle that broke it.
    """
    htf = resample(df, tf)
    h, l, c = (htf[k].to_numpy(float) for k in ("high", "low", "close"))
    sw = SwingTracker(swing_n)
    st = StructureTracker(sw)
    n = len(htf)
    trend, level, since = np.zeros(n), np.full(n, np.nan), np.full(n, -1)
    last_level, last_k = np.nan, -1
    for k in range(n):
        sw.update(k, h, l)
        events = st.update(k, c)
        if events:
            last_level, last_k = events[-1][2].price, k
        trend[k], level[k], since[k] = st.trend, last_level, last_k

    ends = (htf.index + pd.Timedelta(tf)).to_numpy().astype("datetime64[ns]")          # HTF candle closes
    base_close = (df.index + bar_length(df.index)).to_numpy().astype("datetime64[ns]")  # base bar closes
    k = np.searchsorted(ends, base_close, side="right") - 1   # last HTF candle closed by then
    ok = k >= 0
    kk = np.clip(k, 0, None)
    since_k = np.where(ok, since[kk], -1)
    since_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    has = since_k >= 0
    since_time[has] = htf.index[since_k[has]].to_numpy()
    return pd.DataFrame({
        "trend": np.where(ok, trend[kk], 0).astype(int),
        "level": np.where(ok, level[kk], np.nan),
        "since": since_time,
    }, index=df.index)


def all_biases(df, cfg):
    """One htf_bias table per timeframe in cfg.BIAS_TIMEFRAMES."""
    return {tf: htf_bias(df, tf, cfg.BIAS_SWING_N) for tf in cfg.BIAS_TIMEFRAMES}
