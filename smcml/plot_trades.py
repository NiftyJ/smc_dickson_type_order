"""
PICTURES OF TRADES, so you can check with your own eyes that the code marks setups
the way you would.

Each picture has two panels:
  left:  the setup close up - the swing that got swept, the sweep, the structure
         break (CHoCH/BOS), the order block, the biggest FVG, entry/stop/target, the fill
  right: the whole trade from the order to the exit, and the result

    python run_all.py --plot-trades 12            (simulator)
    python run_all.py --data "data/V75.csv" --plot-trades 12
"""
import os

import numpy as np
import pandas as pd

from .bias import resample, bar_length, tf_name

UP, DOWN, WICK = "#2a9d8f", "#e76f51", "#555555"


def _candles(ax, o, h, l, c, s, e, width=3):
    x = np.arange(s, e + 1)
    O, H, L, C = o[s:e + 1], h[s:e + 1], l[s:e + 1], c[s:e + 1]
    up = C >= O
    lw = max(0.4, min(width, 900 / max(len(x), 1)))
    ax.vlines(x, L, H, color=WICK, linewidth=0.6, zorder=2)
    ax.vlines(x[up], O[up], C[up], color=UP, linewidth=lw, zorder=3)
    ax.vlines(x[~up], C[~up], O[~up], color=DOWN, linewidth=lw, zorder=3)


def _dates(ax, index, s, e, n=5):
    ticks = np.linspace(s, e, n).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([index[t].strftime("%b %d\n%H:%M") for t in ticks], fontsize=8)


def _bias_panel(ax, df, t, tf, r, n=40):
    """Last n CLOSED candles of a higher timeframe at the moment the order was placed."""
    name = tf_name(tf)
    closed_at = df.index[t] + bar_length(df.index)
    htf = resample(df.iloc[:t + 1], tf)
    htf = htf[htf.index + pd.Timedelta(tf) <= closed_at].iloc[-n:]
    if htf.empty:
        ax.set_title(f"{name.upper()}: not enough history")
        return
    o, h, l, c = (htf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    _candles(ax, o, h, l, c, 0, len(htf) - 1, width=5)
    trend = int(getattr(r, f"bias_{name}_trend"))
    level = getattr(r, f"bias_{name}_level")
    since = getattr(r, f"bias_{name}_since")
    since = pd.Timestamp(since) if isinstance(since, str) or since is not None else pd.NaT
    word, color = {1: ("BULLISH", "#2a9d8f"), -1: ("BEARISH", "#c1121f")}.get(trend, ("no bias yet", "#555"))
    if np.isfinite(level):
        pos = htf.index.get_indexer([since])[0] if pd.notna(since) and since in htf.index else -1
        start = max(pos - 8, 0) if pos >= 0 else 0
        ax.hlines(level, start, len(htf) - 1, color=color, linewidth=1.4)
        if pos >= 0:
            ax.annotate("close beyond swing", (pos, c[pos]), xytext=(0, 16 * (1 if trend == 1 else -1)),
                        textcoords="offset points", ha="center", fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="->", color=color))
    ticks = np.linspace(0, len(htf) - 1, 4).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([htf.index[i].strftime("%b %d\n%H:%M") for i in ticks], fontsize=7)
    ax.set_xlim(-1, len(htf))
    ax.set_title(f"{name.upper()} bias at entry: {word}", fontsize=11, color=color, fontweight="bold")


def result_text(r, cfg):
    if r.filled == 0:
        return "order never filled (0R)"
    if r.y == 1:
        return f"TARGET HIT: {r.outcome_R:+.1f}R"
    if r.outcome_R <= -1 - r.cost_price / r.risk + 1e-9:
        return f"STOPPED OUT: {r.outcome_R:+.2f}R"
    return f"time exit: {r.outcome_R:+.2f}R"


def plot_trade(r, df, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = df.index
    d = int(r.direction)
    t = int(r.t_place)
    fill = int(r.t_fill) if r.filled == 1 else None
    t_exit = int(r.t_exit)
    target = r.entry + d * cfg.TARGET_R * r.risk
    side = "LONG" if d == 1 else "SHORT"

    tfs = [tf for tf in getattr(cfg, "BIAS_TIMEFRAMES", ()) if hasattr(r, f"bias_{tf_name(tf)}_trend")]
    if tfs:
        fig = plt.figure(figsize=(16, 10.5))
        gs = fig.add_gridspec(2, 6, height_ratios=[0.8, 1])
        for i, tf in enumerate(tfs[:3]):
            _bias_panel(fig.add_subplot(gs[0, 2 * i:2 * i + 2]), df, t, tf, r)
        a1, a2 = fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------------- left: the setup
    s = max(0, min(int(r.swept_swing_bar), int(r.broken_swing_bar), int(r.sweep_bar)) - 15)
    e = min(len(df) - 1, (fill if fill is not None else t + cfg.MAX_BARS_WAIT_FILL) + 6)
    _candles(a1, o, h, l, c, s, e)
    a1.hlines(r.swept_level, r.swept_swing_bar, r.sweep_bar, color="#6c757d", linestyles=":", linewidth=1.6,
              label="swing that got swept (liquidity)")
    ext = l[int(r.ext_bar)] if d == 1 else h[int(r.ext_bar)]
    a1.annotate("sweep", (r.ext_bar, ext), xytext=(0, -22 * d), textcoords="offset points",
                ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
    a1.hlines(r.broken_level, r.broken_swing_bar, t, color="#7b2cbf", linewidth=1.6,
              label=f"swing broken on a close ({r.break_kind})")
    a1.annotate(r.break_kind, (t, c[t]), xytext=(0, 18 * d), textcoords="offset points",
                ha="center", fontsize=9, color="#7b2cbf", arrowprops=dict(arrowstyle="->", color="#7b2cbf"))
    a1.add_patch(Rectangle((r.ob_bar - 0.5, r.ob_low), e - r.ob_bar + 0.5, r.ob_high - r.ob_low,
                           color="#8ab6d6", alpha=0.45, zorder=1, label="order block"))
    if r.fvg_bar >= 0:
        a1.add_patch(Rectangle((r.fvg_bar - 2.5, r.fvg_low), e - r.fvg_bar + 2.5, r.fvg_high - r.fvg_low,
                               color="#ffd166", alpha=0.45, zorder=1, label="biggest FVG in the move"))
    a1.hlines(r.entry, t, e, color="#1d3557", linestyles="--", label="entry (limit order)")
    a1.hlines(r.stop, t, e, color="#c1121f", label="stop")
    a1.axvline(t + 0.5, color="black", linewidth=0.8)
    a1.text(t + 0.8, a1.get_ylim()[1], " order placed", va="top", fontsize=8)
    if fill is not None:
        a1.plot([fill], [r.entry], marker="o", color="#1d3557", markersize=8, zorder=5, label="filled")
    a1.set_xlim(s - 1, e + 1)
    _dates(a1, idx, s, e)
    a1.set_title("The setup", fontsize=11)
    a1.legend(loc="best", fontsize=7.5, framealpha=0.85)

    # ---------------- right: the whole trade
    s2 = max(0, t - 30)
    e2 = min(len(df) - 1, t_exit + 30)
    _candles(a2, o, h, l, c, s2, e2)
    a2.hlines(r.entry, t, t_exit, color="#1d3557", linestyles="--", label="entry")
    a2.hlines(r.stop, t, t_exit, color="#c1121f", label="stop (-1R)")
    a2.hlines(target, t, t_exit, color="#2a9d8f", linewidth=1.6, label=f"target (+{cfg.TARGET_R}R)")
    if fill is not None:
        a2.plot([fill], [r.entry], marker="o", color="#1d3557", markersize=7, zorder=5)
        exit_price = target if r.y == 1 else (r.stop if r.outcome_R < -0.99 else c[t_exit])
        a2.plot([t_exit], [exit_price], marker="X", color="black", markersize=9, zorder=5, label="exit")
    lo = min(l[s2:e2 + 1].min(), r.stop, target)
    hi = max(h[s2:e2 + 1].max(), r.stop, target)
    pad = (hi - lo) * 0.04
    a2.set_ylim(lo - pad, hi + pad)
    a2.set_xlim(s2 - 1, e2 + 1)
    _dates(a2, idx, s2, e2)
    a2.set_title(f"The whole trade ({t_exit - t} bars)", fontsize=11)
    a2.legend(loc="best", fontsize=7.5, framealpha=0.85)

    fig.suptitle(f"Setup {r.candidate_id}: {side}, {idx[t]:%Y-%m-%d %H:%M}.  {result_text(r, cfg)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def plot_sample(trades, df, out_dir, cfg, n=12, seed=0):
    """A mix of target hits, stop-outs and the rest, in time order."""
    os.makedirs(out_dir, exist_ok=True)
    t = pd.DataFrame(trades)
    wins = t[t["y"] == 1]
    stops = t[(t["filled"] == 1) & (t["outcome_R"] < -0.99)]
    rest = t.drop(wins.index.union(stops.index))
    k = max(1, n // 3)
    pick = pd.concat([wins.sample(min(k, len(wins)), random_state=seed),
                      stops.sample(min(k, len(stops)), random_state=seed),
                      rest.sample(min(n - 2 * k, len(rest)), random_state=seed)]).sort_values("t_place")
    paths = []
    for r in pick.itertuples(index=False):
        tag = "win" if r.y == 1 else "stop" if (r.filled == 1 and r.outcome_R < -0.99) else \
              "nofill" if r.filled == 0 else "timeexit"
        p = os.path.join(out_dir, f"setup_{r.candidate_id}_{tag}.png")
        plot_trade(r, df, p, cfg)
        paths.append(p)
    return paths


def plot_simple(df, o, path, before=60, after=30):
    """One trade on one chart: entry, stop, target, fill and exit. Used for live/replay
    trades and re-entries (which don't carry the full setup drawing).

    o: dict with placed_time, fill_time, exit_time, entry, stop, tp, direction, key,
       R, exit_reason, target_R (the columns of live/<symbol>/trades.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = df.index
    pos = lambda t: int(idx.searchsorted(pd.Timestamp(t), side="right") - 1)
    t = pos(o["placed_time"])
    te = pos(o["exit_time"]) if o.get("exit_time") else len(df) - 1
    tf = pos(o["fill_time"]) if o.get("fill_time") else None
    s, e = max(0, t - before), min(len(df) - 1, te + after)
    o_, h_, l_, c_ = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    fig, ax = plt.subplots(figsize=(13, 6))
    _candles(ax, o_, h_, l_, c_, s, e)
    px = lambda x: f"{x:,.2f}" if abs(x) >= 1000 else f"{x:.5g}"
    ax.hlines(o["entry"], t, te, color="#1d3557", linestyles="--", label=f"entry {px(o['entry'])}")
    ax.hlines(o["stop"], t, te, color="#c1121f", label=f"stop {px(o['stop'])}")
    if o.get("tp") is not None and np.isfinite(o["tp"]):
        ax.hlines(o["tp"], t, te, color="#2a9d8f", linewidth=1.4,
                  label=f"target {px(o['tp'])} ({int(o.get('target_R') or 0)}R, may be off the chart)")
    ax.axvline(t + 0.5, color="black", linewidth=0.8)
    if tf is not None:
        ax.plot([tf], [o["entry"]], marker="o", color="#1d3557", markersize=8, zorder=5, label="filled")
    if o.get("exit_time"):
        ax.plot([te], [c_[te] if o.get("exit_reason") == "time" else (o["stop"] if (o.get("R") or 0) < 0 else o["tp"])],
                marker="X", color="black", markersize=10, zorder=5, label="exit")
    lo = min(l_[s:e + 1].min(), o["stop"])
    hi = max(h_[s:e + 1].max(), o["stop"])
    pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(t + 0.8, hi + pad, " order placed", va="top", fontsize=8)
    ax.set_xlim(s - 1, e + 1)
    _dates(ax, idx, s, e, n=6)
    side = "LONG" if int(o["direction"]) == 1 else "SHORT"
    res = f"{o['R']:+.2f}R ({o.get('exit_reason')})" if o.get("R") is not None else "open"
    kind = "re-entry" if o.get("is_reentry") else "setup"
    ax.set_title(f"{o['key']}: {side} {kind}, placed {pd.Timestamp(o['placed_time']):%Y-%m-%d %H:%M}.  {res}",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
