"""
A picture of the range-type indicator on your chart, to check it marks what you would.

    python -m smc2.show --data "data/V75_M15.csv" --tf 1h --start 2024-03-01 --candles 300

Top: candles with each range drawn only on the candles where it was ON (what was known at
each close): pause = orange, wyckoff = blue, staircase = green. Dashed line = its resting
liquidity, x = the candle that took it, triangle = how it ended (up or down).
Bottom: one row per kind, and the answer (range_type) on the last row.
Saves outputs/smc2/range_type_<tf>_<start>.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smcml.data import load_mt5_csv, simulate  # noqa: E402
from smc2.core import resample  # noqa: E402
from smc2.range_type import range_type_bars, KINDS  # noqa: E402

COLORS = {"pause": "#E69F00", "wyckoff": "#0072B2", "staircase": "#009E73"}


def runs(on):
    """(first, last) candle of each stretch where on is 1."""
    x = np.r_[0, np.asarray(on, int), 0]
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def draw(bars, r, title, path):
    n = len(bars)
    x = np.arange(n)
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[5, 1.1], hspace=0.05))
    up = c >= o
    ax.vlines(x, l, h, color="#555555", lw=0.7, zorder=2)
    ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.7, color="#ffffff", edgecolor="#333333", lw=0.6, zorder=3)
    ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.7, color="#333333", edgecolor="#333333", lw=0.6, zorder=3)
    pad = 0.04 * (h.max() - l.min())
    for k in KINDS:
        col, on = COLORS[k], r[k].to_numpy() == 1
        top, bot = r[f"{k}_top"].to_numpy(), r[f"{k}_bottom"].to_numpy()
        ax.fill_between(x, bot, top, where=on, step="mid", color=col, alpha=0.22, lw=0, zorder=1)
        liq, swept = r[f"{k}_liq"].to_numpy(), r[f"{k}_swept"].to_numpy()
        for a, b in runs(on):
            arrow = {1: " up", -1: " down", 0: ""}[int(r[f"{k}_dir"].iloc[a])]
            ax.text(a - 0.4, top[a], f"{k}{arrow}", color=col, fontsize=8, va="bottom", fontweight="bold")
            seg = np.arange(a, b + 1)
            ax.step(seg, liq[seg], where="mid", color=col, lw=1.1, ls="--", zorder=4)
            took = seg[(swept[seg] == 1) & (np.r_[0, swept[seg][:-1]] == 0)]
            ax.scatter(took, liq[took], marker="x", s=60, color=col, lw=2, zorder=5)
        end = r[f"{k}_end"].to_numpy()
        ax.scatter(x[end == 1], h[end == 1] + pad * 0.5, marker="^", s=70, color=col, zorder=5)
        ax.scatter(x[end == -1], l[end == -1] - pad * 0.5, marker="v", s=70, color=col, zorder=5)
    ax.set_ylim(l.min() - pad, h.max() + pad)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.25)

    rows = list(KINDS) + ["answer"]
    for i, k in enumerate(rows):
        y = len(rows) - 1 - i
        for kind in KINDS:
            on = (r[kind] == 1) if k != "answer" else (r["range_type"] == kind)
            if k not in (kind, "answer"):
                continue
            for a, b in runs(on.to_numpy()):
                strip.add_patch(plt.Rectangle((a - 0.5, y - 0.38), b - a + 1, 0.76, color=COLORS[kind], lw=0))
    strip.set_yticks(range(len(rows)))
    strip.set_yticklabels(rows[::-1], fontsize=9)
    strip.set_ylim(-0.6, len(rows) - 0.4)
    strip.grid(axis="x", alpha=0.25)
    ticks = np.linspace(0, n - 1, 9).astype(int)
    strip.set_xticks(ticks)
    strip.set_xticklabels([bars.index[i].strftime("%d %b %Y\n%H:%M") for i in ticks], fontsize=8)
    strip.set_xlim(-1, n)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="your MT5 export (leave out to use the random simulator)")
    p.add_argument("--tf", help="draw on these candles, e.g. 1h or 4h (default: your candles as they are)")
    p.add_argument("--start", help="first candle of the picture, e.g. 2024-03-01 (default: the last candles)")
    p.add_argument("--candles", type=int, default=300)
    p.add_argument("--out", help="where to save the PNG")
    args = p.parse_args()

    df = load_mt5_csv(args.data) if args.data else simulate(20_000, mode="random", seed=7)
    bars = resample(df, args.tf) if args.tf else df
    i0 = int(bars.index.searchsorted(args.start)) if args.start else len(bars) - args.candles
    i0 = max(min(i0, len(bars) - args.candles), 0)
    hist = bars.iloc[:i0 + args.candles]                        # all history before it, so nothing starts cold
    r = range_type_bars(hist).iloc[-args.candles:]
    win = hist.iloc[-args.candles:]
    name = os.path.splitext(os.path.basename(args.data))[0] if args.data else "simulator"
    title = f"{name}, {args.tf or 'your candles'}: range type (each range shown only where it was ON at the close)"
    out = args.out or os.path.join("outputs", "smc2", f"range_type_{args.tf or 'base'}_{win.index[0]:%Y%m%d}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    draw(win, r, title, out)
    counts = {k: int(r[k].sum()) for k in KINDS}
    print(f"saved {out}  (candles on: {counts})")


if __name__ == "__main__":
    main()
