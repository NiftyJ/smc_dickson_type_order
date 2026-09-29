"""
LAYER 4: the setup state machine, plus scan(), which runs every layer bar by bar.

The machine is deliberately SIMPLE. It only finds the minimum sequence:

    IDLE --(price trades beyond the last swing low: liquidity sweep)--> SWEPT
    SWEPT --(bullish structure break within N bars)--> place a limit order at the
             order block = one CANDIDATE, then back to IDLE
    SWEPT --(N bars pass without a break)--> IDLE

Everything that is a judgement call ("is the order block inside an FVG?", "are we in
discount?", "does the higher timeframe agree?") is NOT a rule here. It is written down
as a FEATURE of the candidate, and the model learns how much each one matters.
That is how the overlapping-rules bugs go away: there are no competing rules left.

Each machine sees a (possibly flipped) chart, so it only ever thinks about longs.
"""
import numpy as np
import pandas as pd

from .detectors import SwingTracker, StructureTracker, atr, mirror, find_order_block, find_fvgs
from .bias import all_biases, tf_name

IDLE, SWEPT = "IDLE", "SWEPT"

# The columns the model is allowed to learn from. All are known at the moment the
# order is placed (the structure-break bar). Nothing from the future.
FEATURES = [
    "direction",           # +1 long, -1 short
    "is_choch",            # 1 = the break was a change of character, 0 = break of structure
    "risk_atr",            # stop distance in ATRs
    "cost_R",              # spread/commission as a fraction of the risk
    "sweep_depth_atr",     # how far price went past the swept level
    "bars_sweep_to_shift", # how quickly the break followed the sweep
    "leg_atr",             # size of the move from the sweep extreme to the break close
    "displacement_atr",    # biggest candle body in that move
    "n_fvg",               # number of fair value gaps left in the move
    "fvg_size_atr",        # size of the biggest one
    "ob_fvg_overlap",      # 1 = order block and the biggest FVG overlap
    "retrace_needed_atr",  # how far price must pull back to fill the order
    "pd_pos",              # entry position in the last 100-bar range, 0 = deep discount, 1 = deep premium
    "htf_trend",           # higher-timeframe structure: +1 agrees with the trade, -1 against
    "room_R",              # distance to the 500-bar extreme (obvious liquidity), in R
    "vol_regime",          # current ATR vs its 500-bar average
    "trend_50_atr",        # 50-bar move in ATRs, in the trade's direction
    "hour_sin", "hour_cos", "dow",   # time of day / day of week
]


def feature_columns(cfg):
    """FEATURES plus one bias column per timeframe (bias_d1, bias_h4, bias_h1):
    +1 = that timeframe agrees with the trade, -1 = against, 0 = unknown yet."""
    return FEATURES + [f"bias_{tf_name(tf)}" for tf in cfg.BIAS_TIMEFRAMES]


class SetupMachine:
    def __init__(self, direction, o, h, l, c, atr_arr, cfg, biases=None):
        self.d = direction
        self.cfg = cfg
        self.biases = biases or {}               # tf -> DataFrame(trend, level, since) per bar
        self.o, self.h, self.l, self.c = mirror(o, h, l, c, direction)
        self.atr = atr_arr                       # ranges look the same on a flipped chart
        self.sw = SwingTracker(cfg.SWING_N)
        self.st = StructureTracker(self.sw)
        self.hsw = SwingTracker(cfg.HTF_SWING_N)
        self.hst = StructureTracker(self.hsw)
        self.state = IDLE
        # rolling helpers; pandas rolling windows only look backwards, so they are safe
        sh, sl = pd.Series(self.h), pd.Series(self.l)
        self.max_h_500 = sh.rolling(500, min_periods=50).max().to_numpy()
        self.max_h_100 = sh.rolling(100, min_periods=20).max().to_numpy()
        self.min_l_100 = sl.rolling(100, min_periods=20).min().to_numpy()
        self.atr_long = pd.Series(atr_arr).rolling(500, min_periods=50).mean().to_numpy()

    def step(self, t, active=True):
        """Process bar t. Returns a candidate dict or None."""
        self.sw.update(t, self.h, self.l)
        self.hsw.update(t, self.h, self.l)
        events = self.st.update(t, self.c)
        self.hst.update(t, self.c)
        if not active:
            return None

        if self.state == IDLE:
            sl = self.sw.last_low
            if sl is not None and sl.swept_at < 0 and self.l[t] < sl.price:
                sl.swept_at = t
                self.state = SWEPT
                self.sweep_bar = t
                self.swept_swing_bar = sl.idx
                self.swept_level = sl.price
                self.ext, self.ext_bar = self.l[t], t      # lowest point after the sweep
            return None

        # state == SWEPT
        if self.l[t] < self.ext:
            self.ext, self.ext_bar = self.l[t], t
        bullish = [e for e in events if e[0] == +1]
        if bullish:
            self.state = IDLE
            return self._candidate(t, bullish[0][1], bullish[0][2])
        if t - self.sweep_bar >= self.cfg.MAX_BARS_SWEEP_TO_SHIFT:
            self.state = IDLE
        return None

    def _candidate(self, t, kind, broken):
        cfg, o, h, l, c = self.cfg, self.o, self.h, self.l, self.c
        a = self.atr[t]
        # ---- top-down bias: every higher timeframe must agree with this direction
        bias = {tf: self.biases[tf].iloc[t] for tf in cfg.BIAS_TIMEFRAMES}
        agree = {tf: int(self.d * b["trend"]) for tf, b in bias.items()}
        if cfg.BIAS_FILTER and any(v != 1 for v in agree.values()):
            return None
        ob_bar, ob_top, ob_bot = find_order_block(o, h, l, c, self.ext_bar)
        fvgs = find_fvgs(h, l, self.ext_bar, t)
        entry = min(ob_top, c[t])                  # limit at the top (proximal edge) of the OB
        stop = self.ext - cfg.STOP_BUFFER_ATR * a  # beyond the sweep extreme
        risk = entry - stop
        if not np.isfinite(a) or a <= 0 or risk < cfg.MIN_RISK_ATR * a:
            return None
        cost_price = cfg.COST_ATR * a if cfg.COST_MODE == "atr" else cfg.COST_PRICE
        big = max(fvgs, key=lambda f: f[1] - f[2]) if fvgs else None
        bodies = np.abs(c[self.ext_bar:t + 1] - o[self.ext_bar:t + 1])
        hi100, lo100 = self.max_h_100[t], self.min_l_100[t]

        row = dict(
            t_place=t, direction=self.d, is_choch=int(kind == "CHOCH"),
            risk_atr=risk / a, cost_R=cost_price / risk,
            sweep_depth_atr=(self.swept_level - self.ext) / a,
            bars_sweep_to_shift=t - self.sweep_bar,
            leg_atr=(c[t] - self.ext) / a,
            displacement_atr=bodies.max() / a,
            n_fvg=len(fvgs),
            fvg_size_atr=(big[1] - big[2]) / a if big else 0.0,
            ob_fvg_overlap=int(big is not None and big[2] <= ob_top and big[1] >= ob_bot),
            retrace_needed_atr=(c[t] - entry) / a,
            pd_pos=(entry - lo100) / (hi100 - lo100) if hi100 > lo100 else np.nan,
            htf_trend=self.hst.trend,
            room_R=min((self.max_h_500[t] - entry) / risk, 50.0),
            vol_regime=a / self.atr_long[t],
            trend_50_atr=(c[t] - c[t - 50]) / a if t >= 50 else np.nan,
            # ----- bookkeeping (not features) -----
            entry_f=entry, stop_f=stop, risk=risk, cost_price=cost_price,
            entry=self.d * entry, stop=self.d * stop,          # real prices
            sweep_bar=self.sweep_bar, ext_bar=self.ext_bar, ob_bar=ob_bar,
            swept_level=self.d * self.swept_level,
            ob_high=max(self.d * ob_top, self.d * ob_bot), ob_low=min(self.d * ob_top, self.d * ob_bot),
            swept_swing_bar=self.swept_swing_bar,
            break_kind=kind, broken_swing_bar=broken.idx, broken_level=self.d * broken.price,
            fvg_bar=big[0] if big else -1,
            fvg_high=max(self.d * big[1], self.d * big[2]) if big else np.nan,
            fvg_low=min(self.d * big[1], self.d * big[2]) if big else np.nan,
        )
        for tf, b in bias.items():
            name = tf_name(tf)
            row[f"bias_{name}"] = agree[tf]                  # feature: +1 agrees with the trade
            row[f"bias_{name}_trend"] = int(b["trend"])      # real direction: +1 bullish, -1 bearish
            row[f"bias_{name}_level"] = b["level"]           # swing whose break set the bias
            row[f"bias_{name}_since"] = b["since"]           # candle that broke it
        return row


def scan(df, cfg):
    """Run every layer bar by bar and return one row per candidate setup."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = atr(h, l, c, cfg.ATR_N)
    biases = all_biases(df, cfg)
    machines = [SetupMachine(d, o, h, l, c, a, cfg, biases) for d in (+1, -1)]
    warmup = max(2 * cfg.HTF_SWING_N + 1, 100)
    rows = []
    for t in range(len(df)):
        for m in machines:
            cand = m.step(t, active=t >= warmup)
            if cand is not None:
                rows.append(cand)
    cands = pd.DataFrame(rows)
    if cands.empty:
        return cands
    cands = cands.sort_values(["t_place", "direction"]).reset_index(drop=True)
    for col in [c for c in cands.columns if c.endswith("_since")]:
        cands[col] = pd.to_datetime(cands[col]).astype("datetime64[ns]")
    times = df.index[cands["t_place"].to_numpy()]
    cands.insert(0, "time", times)
    hour = times.hour + times.minute / 60
    cands["hour"] = hour
    cands["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    cands["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    cands["dow"] = times.dayofweek
    cands.insert(0, "candidate_id", np.arange(len(cands)))
    return cands
