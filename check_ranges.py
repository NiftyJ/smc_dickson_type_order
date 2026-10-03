"""
TRY THE THREE RANGE TYPES (wyckoff/range_types.py) ON REAL PRICE HISTORY.

    python check_ranges.py --data data/real/XAUUSDm15.csv
    python check_ranges.py --data "data/V75_M15.csv" --last 30000 --charts 12
    python check_ranges.py --data "data/V75_M1.csv" --tf 5min

It does three things, and writes everything into outputs/range_check/:

  1. HOW OFTEN   share of candles each flag is on, how many separate ranges, how long.
  2. DID IT HOLD from the candle a range is first flagged, did the next --ahead closes
                 all stay inside its box? Next to it: the same-size box dropped at
                 random unflagged candles. If the two numbers are equal, the flag
                 tells you nothing about what comes next.
  3. PICTURES    --charts random examples of each type, so you can say yes / no to
                 "would I call that a range?". <name>_ranges.csv lists every range
                 with its start time, so you can find any of them on your MT5 chart.
"""
import argparse
import os

import numpy as np
import pandas as pd

from smcml.bias import resample
from smcml.data import load_mt5_csv
from smcml.detectors import atr
from smcml.plot_trades import _candles, _dates, UP, DOWN
from wyckoff.range_types import range_types

TYPES = ("pause", "wyckoff", "staircase")
BOX, LIQ, INK = "#3b6fb6", "#c0392b", "#444444"


def episodes(r, k):
    """Every separate range of type k: (first flagged candle, last flagged candle)."""
    f = r[k].to_numpy()
    edge = np.diff(np.r_[0, f, 0])
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1))


def held(c, t, top, bot, n):
    """Did the n closes after candle t all stay inside the box?"""
    nxt = c[t + 1:t + 1 + n]
    return bool((nxt <= top).all() and (nxt >= bot).all())


def hold_test(r, k, c, a, n, rng, per=10):
    """Share of ranges whose box held for n candles, and the same for same-size random boxes."""
    free = np.flatnonzero((r["in_range"].to_numpy() == 0) & np.isfinite(a))
    free = free[free < len(c) - n - 1]
    top, bot = r[f"{k}_top"].to_numpy(), r[f"{k}_bottom"].to_numpy()
    real, rand, rows = [], [], []
    for s, e in episodes(r, k):
        row = {"type": k, "start": r.index[s], "end": r.index[e], "candles": e - s + 1,
               "top": top[s:e + 1].max(), "bottom": bot[s:e + 1].min()}
        if s < len(c) - n - 1 and top[s] > bot[s] and a[s] > 0:
            row[f"held_{n}"] = held(c, s, top[s], bot[s], n)
            real.append(row[f"held_{n}"])
            height, pos = (top[s] - bot[s]) / a[s], np.clip((c[s] - bot[s]) / (top[s] - bot[s]), 0, 1)
            for u in rng.choice(free, min(per, len(free)), replace=False):
                b = c[u] - pos * height * a[u]
                rand.append(held(c, u, b + height * a[u], b, n))
        rows.append(row)
    return (np.mean(real) if real else np.nan), (np.mean(rand) if rand else np.nan), rows


def picture(bars, r, k, eps, path, title, before=50, after=25):
    """A grid of examples: candles, the box while the flag is on, and what came next."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    o, h, l, c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    cols = min(3, len(eps))
    rows_n = int(np.ceil(len(eps) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.6 * cols, 3.7 * rows_n), squeeze=False)
    top, bot = r[f"{k}_top"].to_numpy(), r[f"{k}_bottom"].to_numpy()
    for ax, (s, e) in zip(axes.ravel(), eps):
        a0, a1 = max(s - before, 0), min(e + after, len(c) - 1)
        _candles(ax, o, h, l, c, a0, a1)
        x = np.arange(s, e + 1)
        ax.axvspan(s - 0.5, e + 0.5, color=BOX, alpha=0.08, lw=0)
        ax.step(x, top[s:e + 1], where="mid", color=BOX, lw=1.6)
        ax.step(x, bot[s:e + 1], where="mid", color=BOX, lw=1.6)
        if k == "pause":
            liq = r["pause_liq"].to_numpy()[s:e + 1]
            if np.isfinite(liq).any():
                ax.step(x, liq, where="mid", color=LIQ, lw=1.3, ls=(0, (4, 2)))
        _dates(ax, bars.index, a0, a1, n=4)
        took = " - liquidity taken" if k == "pause" and r["pause_swept"].to_numpy()[s:e + 1].any() else ""
        ax.set_title(f"{bars.index[s]:%Y-%m-%d %H:%M}   {e - s + 1} candle{"" if e == s else "s"}{took}", fontsize=9.5, color=INK, loc="left")
        ax.tick_params(labelsize=8, colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(eps):]:
        ax.axis("off")
    keys = [Patch(color=UP, label="up candle"), Patch(color=DOWN, label="down candle"),
            Line2D([0], [0], color=BOX, lw=1.6, label="box edges while the flag is on"),
            Patch(color=BOX, alpha=0.15, label="flag on (no new orders)")]
    if k == "pause":
        keys.append(Line2D([0], [0], color=LIQ, lw=1.3, ls=(0, (4, 2)), label="resting low / high (liquidity)"))
    fig.suptitle(title, fontsize=12.5, color="#222222", x=0.01, ha="left")
    fig.legend(handles=keys, loc="lower center", ncol=len(keys), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06 / rows_n + 0.02, 1, 0.97))
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Try the three range types on real price history.")
    p.add_argument("--data", required=True, help="your MT5 export (CSV)")
    p.add_argument("--tf", default=None, help='build these candles first, e.g. "5min", "1h"')
    p.add_argument("--last", type=int, default=0, help="only the last N candles (0 = all)")
    p.add_argument("--ahead", type=int, default=20, help="candles ahead for the did-it-hold test")
    p.add_argument("--charts", type=int, default=6, help="examples of each type to draw")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = load_mt5_csv(args.data)
    bars = resample(df, args.tf) if args.tf else df
    if args.last:
        bars = bars.iloc[-args.last:]
    name = os.path.splitext(os.path.basename(args.data))[0] + (f"_{args.tf}" if args.tf else "")
    out = os.path.join("outputs", "range_check")
    os.makedirs(out, exist_ok=True)
    print(f"{name}: {len(bars):,} candles, {bars.index[0]} to {bars.index[-1]}. Working ...")

    r = range_types(bars)
    h, l, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    a = atr(h, l, c, 14)
    rng = np.random.default_rng(args.seed)

    table, every = [], []
    for k in TYPES:
        real, rand, rows = hold_test(r, k, c, a, args.ahead, rng)
        every += rows
        lengths = [x["candles"] for x in rows]
        table.append({"type": k, "candles flagged": f"{r[k].mean():.1%}", "ranges": len(rows),
                      "typical length": f"{int(np.median(lengths))} candles" if rows else "-",
                      f"box held {args.ahead} candles": f"{real:.0%}" if rows else "-",
                      "same box, random time": f"{rand:.0%}" if rows else "-"})
        eps = episodes(r, k)
        if eps and args.charts:
            pick = sorted(rng.choice(len(eps), min(args.charts, len(eps)), replace=False))
            picture(bars, r, k, [eps[i] for i in pick], os.path.join(out, f"{name}_{k}.png"),
                    f"{name}: {len(pick)} random '{k}' ranges out of {len(eps)}")
    print()
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\nany of the three on: {r['in_range'].mean():.1%} of candles")
    if not every:
        print("No range of any type was found. A few thousand candles are needed before the flags can fire.")
        return
    pd.DataFrame(every).sort_values("start").to_csv(os.path.join(out, f"{name}_ranges.csv"), index=False)
    r.to_csv(os.path.join(out, f"{name}_flags.csv"))
    print(f"pictures, {name}_ranges.csv (one row per range) and {name}_flags.csv (one row per candle) are in {out}/")


if __name__ == "__main__":
    main()
