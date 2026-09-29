"""
LAYER 5: what actually happened to each candidate (this is the only place that
is ALLOWED to look into the future, because it is grading the past).

For every candidate:
  1. Was the limit order filled within MAX_BARS_WAIT_FILL bars?  (if not: 0R)
  2. After the fill, what came first: the stop (-1R) or the target (+TARGET_R)?
  3. How far did price run in our favour before the stop (MFE, in R)?
  4. Costs are subtracted from every filled trade, in R.

Break-even (BREAKEVEN_AT_R, from your journal): once price has gone that many R in our
favour, the stop moves to the entry from the NEXT bar on. A trade that then comes back
to the entry closes at 0R (minus costs) instead of -1R.

Conservative choices (so the backtest is never nicer than reality):
  * if the stop and the target are both touched inside the same bar -> counted as a LOSS
  * on the fill bar, only the CLOSE counts towards the target (the high may have
    happened before we were filled)
"""
import numpy as np
import pandas as pd

from .detectors import mirror


def _first(mask):
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else np.inf


def label(cands, df, cfg):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    flipped = {d: mirror(o, h, l, c, d) for d in (+1, -1)}
    n = len(df)
    targets = sorted(set(cfg.EXTRA_TARGETS) | {cfg.TARGET_R})
    be = getattr(cfg, "BREAKEVEN_AT_R", None)
    out = []
    for row in cands.itertuples(index=False):
        _, fh, fl, fc = flipped[row.direction]
        entry, stop, risk = row.entry_f, row.stop_f, row.risk
        cost_R = row.cost_price / risk
        res = {"filled": 0, "t_fill": -1, "mfe_R": 0.0}
        t0 = row.t_place + 1
        t1 = row.t_place + cfg.MAX_BARS_WAIT_FILL
        k_fill = _first(fl[t0:min(t1, n - 1) + 1] <= entry) if t0 < n else np.inf

        if not np.isfinite(k_fill):
            complete = t1 <= n - 1                   # did we see the whole waiting period?
            for tr in targets:
                res[f"R_{tr}"] = 0.0 if complete else np.nan
                res[f"hit_{tr}"] = 0 if complete else np.nan
                res[f"t_exit_{tr}"] = t1
            res["t_exit"] = t1
        else:
            f = t0 + int(k_fill)
            res.update(filled=1, t_fill=f)
            end = f + cfg.MAX_HOLD_BARS - 1
            H, L, Cl = fh[f:end + 1], fl[f:end + 1], fc[f:end + 1]
            seen_all = end <= n - 1
            fav = (H - entry) / risk
            fav[0] = (Cl[0] - entry) / risk          # fill bar: only the close counts
            k_stop = _first(L <= stop)
            before_stop = fav[:int(k_stop)] if np.isfinite(k_stop) else fav
            res["mfe_R"] = float(np.clip(before_stop.max() if len(before_stop) else 0.0, 0, 50))
            for tr in targets:
                k_tgt = _first(fav >= tr)
                k_be = _first(fav >= be) if (be is not None and be < tr) else np.inf
                if np.isfinite(k_be) and k_be < k_stop and k_be < k_tgt:
                    # stop moved to the entry from the bar after price reached +be R
                    k2 = k_be + 1 + _first(L[int(k_be) + 1:] <= entry)
                    if k_tgt < k2:
                        r, k_exit, hit = tr, k_tgt, 1
                    elif np.isfinite(k2):
                        r, k_exit, hit = 0.0, k2, 0            # break-even exit
                    elif seen_all:
                        r, k_exit, hit = (Cl[-1] - entry) / risk, len(Cl) - 1, 0
                    else:
                        r, k_exit, hit = np.nan, np.nan, np.nan
                elif k_tgt < k_stop:
                    r, k_exit, hit = tr, k_tgt, 1
                elif np.isfinite(k_stop):
                    r, k_exit, hit = -1.0, k_stop, 0  # includes "both in the same bar"
                elif seen_all:
                    r, k_exit, hit = (Cl[-1] - entry) / risk, len(Cl) - 1, 0  # time exit
                else:
                    r, k_exit, hit = np.nan, np.nan, np.nan  # ran out of data
                res[f"R_{tr}"] = r - cost_R if np.isfinite(r) else np.nan
                res[f"hit_{tr}"] = hit
                res[f"t_exit_{tr}"] = f + k_exit if np.isfinite(k_exit) else np.nan
                if tr == cfg.TARGET_R:
                    res["t_exit"] = f + k_exit if np.isfinite(k_exit) else np.nan
        out.append(res)

    lab = pd.DataFrame(out, index=cands.index)
    lab["outcome_R"] = lab[f"R_{cfg.TARGET_R}"]
    lab["y"] = lab[f"hit_{cfg.TARGET_R}"]
    lab["complete"] = lab["outcome_R"].notna()
    return pd.concat([cands, lab], axis=1)


def target_table(labeled, cfg):
    """Win rate and average R at each target, next to what a pure random walk gives."""
    done = labeled[labeled["complete"]]
    filled = done[done["filled"] == 1]
    rows = []
    for tr in sorted(set(cfg.EXTRA_TARGETS) | {cfg.TARGET_R}):
        col = filled[f"R_{tr}"].dropna()
        hits = filled[f"hit_{tr}"].dropna()
        rows.append({
            "target_R": tr,
            "filled_trades": len(col),
            "win_rate": hits.mean() if len(hits) else np.nan,
            "random_walk_win_rate": chance_rate(tr, cfg),
            "avg_R_per_trade": col.mean() if len(col) else np.nan,
        })
    return pd.DataFrame(rows)


def chance_rate(tr, cfg):
    """How often a target of tr R is hit on a pure random walk (the luck yardstick).
    Without break-even: 1 / (tr + 1). With the stop moved to entry at +be R:
    reach +be before -1R (1 / (be + 1)), then reach tr before falling back to entry (be / tr)."""
    be = getattr(cfg, "BREAKEVEN_AT_R", None)
    if be is None or be >= tr:
        return 1.0 / (tr + 1)
    return (1.0 / (be + 1)) * (be / tr)
