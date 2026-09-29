"""
MORE SHOTS: re-entries after a setup is stopped out (your journal: "max 3 tries").

After a trade of a setup hits its stop (and the setup has shots left):
  1. Watch the entry timeframe (your bars, e.g. M15) for up to REENTRY_WINDOW_BARS bars.
  2. Wait for a break of structure IN THE SAME DIRECTION as the failed trade
     (for a long: a candle CLOSES above the last confirmed swing high).
  3. The D1/H4/H1 bias must still agree at that moment (if BIAS_FILTER is on).
  4. Place the re-entry exactly like the original entry model:
       new low   = lowest low between the stop-out and that break
       order     = limit at the top of the order block at that new low
                   (or, with REENTRY_ENTRY = "break", at the close of the break candle)
       stop      = just below the new low
       target    = the same target rule as every other order
  5. At most REENTRY_MAX_SHOTS trades per setup in total (the first one included).
     REENTRY_MAX_SHOTS = 2 means one re-entry, 3 means up to two. A break-even exit
     is not a stop-out, so it doesn't lead to another shot.

Everything here only uses bars up to the re-entry decision; the no-lookahead test
checks it.
"""
import numpy as np
import pandas as pd

from .detectors import SwingTracker, StructureTracker, atr, mirror, find_order_block
from .bias import all_biases
from .labels import label


def structure_breaks(df, cfg):
    """Bars where a close broke the last confirmed swing, per trade direction."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    out = {}
    for d in (+1, -1):
        fo, fh, fl, fc = mirror(o, h, l, c, d)
        sw = SwingTracker(cfg.SWING_N)
        st = StructureTracker(sw)
        bars = []
        for t in range(len(df)):
            sw.update(t, fh, fl)
            if any(e[0] == +1 for e in st.update(t, fc)):
                bars.append(t)
        out[d] = np.array(bars, dtype=int)
    return out


ID_START = 1_000_000_000       # re-entry ids start here, so they never clash with setup ids


def reentry_id(parent_id, target):
    """A re-entry's id comes only from its parent and target, so it is the same however
    much later data exists (the no-lookahead tests compare ids)."""
    return ID_START + (int(parent_id) + 1) * 1000 + int(target)


def find_reentries(labeled, df, cfg):
    """Every possible re-entry, shot by shot, up to REENTRY_MAX_SHOTS per setup.

    For each trade that could be stopped out (per target it could be traded with), the
    next shot if one forms. Shot 3 is looked for after each possible shot 2, and so on.
    Returns them graded by labels.label(): parent_id / parent_target say which stopped
    trade each one follows, candidate_id is the re-entry's own id, shot is 2, 3, ...
    """
    max_shots = int(getattr(cfg, "REENTRY_MAX_SHOTS", 2))
    ctx = _context(df, cfg)
    rounds, parents = [], labeled
    for shot in range(2, max_shots + 1):
        if parents is None or len(parents) == 0:
            break
        re = reentry_round(parents, df, cfg, shot, ctx)
        if re.empty:
            break
        rounds.append(re)
        parents = re
    return pd.concat(rounds, ignore_index=True) if rounds else pd.DataFrame()


def _context(df, cfg):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    return dict(a=atr(h, l, c, cfg.ATR_N), flipped={d: mirror(o, h, l, c, d) for d in (+1, -1)},
                breaks=structure_breaks(df, cfg), biases=all_biases(df, cfg) if cfg.BIAS_FILTER else None)


def reentry_round(parents, df, cfg, shot, ctx=None):
    """One round: the next shot after each stopped parent trade (max one per parent and target)."""
    ctx = ctx or _context(df, cfg)
    a, flipped, breaks, biases = ctx["a"], ctx["flipped"], ctx["breaks"], ctx["biases"]
    targets = sorted({cfg.TARGET_R, cfg.TARGET_R_IN_DRAWDOWN})
    rows = []
    for r in parents[parents["filled"] == 1].itertuples(index=False):
        d = int(r.direction)
        fo, fh, fl, fc = flipped[d]
        for tr in targets:
            R, s = getattr(r, f"R_{tr}"), getattr(r, f"t_exit_{tr}")
            if not (np.isfinite(R) and np.isfinite(s) and R < -0.99):
                continue                                   # only after a stop-out
            s = int(s)
            b = breaks[d]
            for k in b[(b > s) & (b <= s + cfg.REENTRY_WINDOW_BARS)]:
                if biases is not None and any(d * biases[tf]["trend"].iloc[k] != 1 for tf in cfg.BIAS_TIMEFRAMES):
                    continue
                low_bar = s + int(np.argmin(fl[s:k + 1]))
                ob_bar, ob_top, ob_bot = find_order_block(fo, fh, fl, fc, low_bar)
                entry = fc[k] if getattr(cfg, "REENTRY_ENTRY", "order_block") == "break" else min(ob_top, fc[k])
                stop = fl[low_bar] - cfg.STOP_BUFFER_ATR * a[k]
                risk = entry - stop
                if not np.isfinite(a[k]) or risk < cfg.MIN_RISK_ATR * a[k]:
                    continue
                cost_price = cfg.COST_ATR * a[k] if cfg.COST_MODE == "atr" else cfg.COST_PRICE
                rows.append(dict(
                    parent_id=int(r.candidate_id), parent_target=tr, stopout_bar=s, shot=shot,
                    candidate_id=reentry_id(r.candidate_id, tr), t_place=int(k), direction=d,
                    entry_f=entry, stop_f=stop, risk=risk, cost_price=cost_price,
                    entry=d * entry, stop=d * stop, ext_bar=low_bar, ob_bar=ob_bar,
                    ob_high=max(d * ob_top, d * ob_bot), ob_low=min(d * ob_top, d * ob_bot),
                ))
                break                                      # max one re-entry per stop-out
    re = pd.DataFrame(rows)
    if re.empty:
        return re
    re.insert(0, "time", df.index[re["t_place"].to_numpy()])
    return label(re, df, cfg)
