"""
A PICTURE OF EVERY RANGE THE DETECTOR FINDS, so you can check it against your own eye.

    python -m wyckoff.range_report --data "data/XAUUSD_M5.csv"                 the last 365 days
    python -m wyckoff.range_report --data "data/NETH25_M1.csv" --last-days 30
    python -m wyckoff.range_report --data "data/V75_M1.csv" --tf 5min --from 2026-09-01 --to 2026-10-01

Writes three PDFs into outputs/range_report/<file name>/ :
    pause.pdf       the pause after an impulse
    wyckoff.pdf     the directionless range
    staircase.pdf   the zig-zag range

Page 1 of each PDF: the rule, how many ranges were found, how long they lasted and how
they ended. Then one chart per detected range, in time order, three to a page:
  * the blue box is drawn only while the flag was ON (the shaded candles)
  * the dashed blue lines are the candles that formed it, before the flag came on
  * pause charts also show the resting low (violet) and the candle that took it out

The detector runs over the WHOLE file first (it needs earlier candles to warm up), and
only ranges that START inside the chosen period are drawn. Nothing uses future candles.
"""
import argparse
import os

import numpy as np
import pandas as pd

UP, DOWN, BOX, LIQ = "#1baf7a", "#eb6834", "#2a78d6", "#4a3aa7"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"

KINDS = {
    "pause": dict(
        title="Pause after an impulse",
        rule=["A spike of 6 or more ATRs within 12 candles that leaves a fair value gap (or has one huge body),",
              "then at least 5 candles with no close above the spike high and no close below half of the spike.",
              "It ends when a candle closes beyond the pause high (the break), when a candle closes back",
              "past half of the spike (failed), or after 80 candles."]),
    "wyckoff": dict(
        title="Directionless (Wyckoff) range",
        rule=["Over the last 30 to 60 candles: a box from the 5th percentile of lows to the 95th percentile of highs",
              "that is at most 0.5 x sqrt(candles) ATRs tall and that price has crossed edge to edge 4 or more times,",
              "with no candle closing outside it. The box is then frozen.",
              "It ends when a candle closes more than half an ATR outside the box."]),
    "staircase": dict(
        title="Staircase (zig-zag) range",
        rule=["Two cycles in a row where the new swing high beats the previous one by at most half of the leg",
              "that made it, and the pullback then gives back at least half of that leg (swings: 3 candles each side).",
              "It ends when a candle closes far beyond the last high, closes below the last swing low,",
              "or the next step no longer fits."]),
}


def find_episodes(r, kind):
    """Runs of candles where the flag is on (and points the same way). Returns (start, end, direction):
    end is the first candle after the run."""
    flag = r[kind].to_numpy()
    d = r[f"{kind}_dir"].to_numpy() if f"{kind}_dir" in r else np.ones(len(r), dtype=int)
    key = flag * np.where(d == 0, 1, d)
    cuts = np.flatnonzero(np.diff(np.r_[0, key, 0]) != 0)
    return [(int(a), int(b), int(key[a])) for a, b in zip(cuts[:-1], cuts[1:]) if key[a] != 0]


def how_it_ended(kind, s, e, d, c, top, bot, max_flagged):
    if e >= len(c):
        return "still on at the end of the data"
    up, down = c[e] > top[e - 1], c[e] < bot[e - 1]
    if kind == "pause":
        if (d == 1 and up) or (d == -1 and down):
            return "break " + ("up" if d == 1 else "down") + " (continued)"
        if e - s >= max_flagged:
            return "timed out"
        return "failed (gave back half the spike)"
    if up:
        return "break up"
    if down:
        return "break down"
    return "steps stopped fitting" if kind == "staircase" else "ended inside the box"


def _span(n, bar):
    mins = int(round(n * bar.total_seconds() / 60))
    if mins < 60:
        return f"{mins}m"
    if mins < 48 * 60:
        return f"{mins // 60}h {mins % 60:02d}m"
    return f"{mins / 1440:.1f} days"


def _fmt_price(x):
    return f"{x:,.2f}" if abs(x) < 10_000 else f"{x:,.0f}"


def draw_range(ax, kind, n, s, e, d, ending, bars, r, bar, scale):
    """One chart: candles around the range, the box, and (pause) the liquidity takeout."""
    o, h, l, c = (bars[k].to_numpy(float) * scale for k in ("open", "high", "low", "close"))
    top, bot = r[f"{kind}_top"].to_numpy() * scale, r[f"{kind}_bottom"].to_numpy() * scale
    a0, a1 = max(s - (45 if kind == "pause" else 75), 0), min(e + 20, len(c))
    x = np.arange(a0, a1) - a0
    col = np.where(c[a0:a1] >= o[a0:a1], UP, DOWN)
    ax.set_facecolor(SURFACE)
    ax.axvspan(s - a0 - 0.5, e - a0 - 0.5, color=BOX, alpha=0.10, lw=0)
    ax.vlines(x, l[a0:a1], h[a0:a1], color=col, lw=0.9)
    body = np.abs(c[a0:a1] - o[a0:a1])
    ax.bar(x, np.maximum(body, 0.002 * (h[a0:a1].max() - l[a0:a1].min())), bottom=np.minimum(o[a0:a1], c[a0:a1]),
           color=col, width=0.72, lw=0)
    xs = np.r_[s - 0.5, np.arange(s, e), e - 0.5] - a0       # half a candle either side, so a 1-candle box still shows
    ax.plot(xs, np.r_[top[s], top[s:e], top[e - 1]], color=BOX, lw=1.8)
    ax.plot(xs, np.r_[bot[s], bot[s:e], bot[e - 1]], color=BOX, lw=1.8)
    f0, tol = s, 0.02 * (top[s] - bot[s])                    # dashed: the candles that formed the box
    while f0 > a0 and bot[s] - tol <= c[f0 - 1] <= top[s] + tol:
        f0 -= 1
    for edge in (top[s], bot[s]):
        ax.plot([f0 - a0, s - a0], [edge, edge], color=BOX, lw=1.1, ls=(0, (4, 3)))

    extra = ""
    if kind == "pause":
        liq, swept = r["pause_liq"].to_numpy() * scale, r["pause_swept"].to_numpy()
        lv = liq[e - 1]
        if np.isfinite(lv):
            beyond = (l < lv) if d == 1 else (h > lv)        # an up pause rests on a low, a down pause under a high
            at_level = np.isclose(l if d == 1 else h, lv)
            lo_k = max(a0, s - 12)
            form = [k for k in range(a0, e) if at_level[k]]
            k_form = form[-1] if form else lo_k
            if swept[e - 1]:
                took = [k for k in range(k_form + 1, e) if beyond[k]]
                k_take = took[0] if took else e - 1
                ax.plot([k_form - a0, k_take - a0], [lv, lv], color=LIQ, lw=1.8)
                ax.plot([k_take - a0], [l[k_take] if d == 1 else h[k_take]], marker="v" if d == 1 else "^", ms=8,
                        color=LIQ, mec=SURFACE, mew=1.5, ls="none", zorder=5)
                extra = "  ·  takeout first"
            else:
                ax.plot([k_form - a0, e - 1 - a0], [lv, lv], color=LIQ, lw=1.8, ls=(0, (1, 2)))
                extra = "  ·  no takeout"
        else:
            extra = "  ·  no resting level"

    ylo, yhi = l[a0:a1].min(), h[a0:a1].max()
    pad = 0.06 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlim(-1, a1 - a0)
    way = {"pause": {1: "Up pause", -1: "Down pause"}, "staircase": {1: "Up staircase", -1: "Down staircase"},
           "wyckoff": {1: "Box"}}[kind][d]
    when = bars.index[s]
    ax.set_title(f"#{n}   {when:%d %b %Y  %H:%M}", loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=19)
    ax.text(0, 1.03, f"{way}  ·  flagged {e - s} candle{'s' if e - s != 1 else ''} ({_span(e - s, bar)})  ·  box {_fmt_price(top[e - 1] - bot[e - 1])} tall"
            f"{extra}  ·  {ending}", transform=ax.transAxes, color=INK2, fontsize=9.2, va="bottom")
    ticks = (np.array([0.10, 0.37, 0.63, 0.90]) * (a1 - a0 - 1)).astype(int)
    fmt = "%d %b %H:%M" if bar < pd.Timedelta("1D") else "%d %b %Y"
    ax.set_xticks(ticks)
    ax.set_xticklabels([bars.index[a0 + t].strftime(fmt) for t in ticks], color=INK2, fontsize=8.8)
    ax.tick_params(axis="y", colors=INK2, labelsize=8.8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.tick_right()
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _summary_page(pdf, kind, eps, endings, info, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(8.5, 11), facecolor=SURFACE)
    y = 0.93
    fig.text(0.08, y, KINDS[kind]["title"], fontsize=20, fontweight="bold", color=INK)
    y -= 0.03
    fig.text(0.08, y, "Every range the detector found, one chart each", fontsize=12.5, color=INK2)
    y -= 0.045
    for line in (f"Data: {info['source']}, {info['tf']} candles",
                 f"Period: {info['start']:%d %b %Y} to {info['end']:%d %b %Y}  ({info['bars']:,} candles)"):
        fig.text(0.08, y, line, fontsize=11, color=INK)
        y -= 0.022
    y -= 0.02
    fig.text(0.08, y, "The rule", fontsize=13, fontweight="bold", color=INK)
    y -= 0.026
    for line in KINDS[kind]["rule"]:
        fig.text(0.08, y, line, fontsize=10, color=INK2)
        y -= 0.02
    y -= 0.022
    fig.text(0.08, y, "What it found", fontsize=13, fontweight="bold", color=INK)
    y -= 0.03
    L = np.array([e - s for s, e, _ in eps]) if eps else np.array([0])
    rows = [("Ranges found", f"{len(eps):,}"),
            ("Candles with the flag on", f"{info['share']:.1%} of the period"),
            ("Length while flagged: typical (median)", f"{np.median(L):.0f} candles ({_span(np.median(L), info['bar'])})"),
            ("Length while flagged: 9 in 10 are shorter than", f"{np.percentile(L, 90):.0f} candles ({_span(np.percentile(L, 90), info['bar'])})"),
            ("Longest", f"{L.max():.0f} candles ({_span(L.max(), info['bar'])})")]
    if kind != "wyckoff" and eps:
        ups = sum(d == 1 for _, _, d in eps)
        rows.append(("Pointing up / down", f"{ups:,} / {len(eps) - ups:,}"))
    for k, v in sorted(pd.Series(endings).value_counts().items(), key=lambda kv: -kv[1]):
        rows.append((f"Ended: {k}", f"{v:,}  ({v / max(len(eps), 1):.0%})"))
    if kind == "pause" and info.get("takeout") is not None:
        rows.append(("Breaks that had a liquidity takeout first", info["takeout"]))
    for k, v in rows:
        fig.text(0.08, y, k, fontsize=10.5, color=INK2)
        fig.text(0.62, y, v, fontsize=10.5, color=INK, fontweight="bold")
        fig.add_artist(Line2D([0.08, 0.92], [y - 0.008, y - 0.008], color=GRID, lw=0.7))
        y -= 0.027
    y -= 0.02
    fig.text(0.08, y, "How to read the charts", fontsize=13, fontweight="bold", color=INK)
    handles = [Patch(color=UP, label="up candle"), Patch(color=DOWN, label="down candle"),
               Patch(color=BOX, alpha=0.10, label="flag on (no new orders)"),
               Line2D([0], [0], color=BOX, lw=1.8, label="the box while the flag is on"),
               Line2D([0], [0], color=BOX, lw=1.1, ls=(0, (4, 3)), label="the candles that formed it")]
    if kind == "pause":
        handles += [Line2D([0], [0], color=LIQ, lw=1.8, marker="v", ms=7, label="resting low or high, and its takeout"),
                    Line2D([0], [0], color=LIQ, lw=1.8, ls=(0, (1, 2)), label="resting level that was not taken")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, y - 0.008), ncol=2, frameon=False,
               fontsize=10, labelcolor=INK2, handlelength=2.4, columnspacing=2.5)
    y -= 0.05 + 0.024 * ((len(handles) + 1) // 2)
    fig.text(0.08, y, "What to check", fontsize=13, fontweight="bold", color=INK)
    y -= 0.026
    for line in ("Go through the charts and mark each one: would you call this that kind of range, yes or no?",
                 "The flag always comes on late. The dashed lines show how much of the range had already",
                 "happened before the bot could know about it."):
        fig.text(0.08, y, line, fontsize=10, color=INK2)
        y -= 0.02
    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def make_report(df, out_dir, source, start=None, end=None, tf=None, scale=1.0, per_page=3, max_each=None,
                settings=None):
    """Runs the detector over df and writes pause.pdf, wyckoff.pdf and staircase.pdf to out_dir.
    Returns {kind: dict(ranges, share, endings, path)}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from smcml.bias import resample, bar_length
    from wyckoff.range_types import range_types, Settings

    s_ = settings or Settings()
    bars = resample(df, tf) if tf else df
    r = range_types(df, tf=tf, settings=s_)
    bar = bar_length(bars.index)
    start = pd.Timestamp(start) if start is not None else bars.index[0]
    end = pd.Timestamp(end) if end is not None else bars.index[-1]
    inside = (bars.index >= start) & (bars.index <= end)
    c = bars["close"].to_numpy(float)
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    tf_name = tf or (f"{int(bar.total_seconds() // 60)}-minute" if bar < pd.Timedelta("1h") else str(bar))
    result = {}
    for kind in KINDS:
        eps = [ep for ep in find_episodes(r, kind) if inside[ep[0]]]
        top, bot = r[f"{kind}_top"].to_numpy(), r[f"{kind}_bottom"].to_numpy()
        endings = [how_it_ended(kind, s, e, d, c, top, bot, s_.PAUSE_MAX_BARS - s_.PAUSE_MIN_BARS) for s, e, d in eps]
        info = dict(source=source, tf=tf_name, start=bars.index[inside][0], end=bars.index[inside][-1],
                    bars=int(inside.sum()), bar=bar, share=float(r[kind].to_numpy()[inside].mean()), takeout=None)
        if kind == "pause":
            sw = r["pause_swept"].to_numpy()
            brk = [sw[e - 1] for (s, e, d), end_ in zip(eps, endings) if end_.startswith("break")]
            if brk:
                info["takeout"] = f"{int(np.sum(brk)):,} of {len(brk):,}  ({np.mean(brk):.0%})"
        shown = eps if max_each is None else eps[:max_each]
        path = os.path.join(out_dir, f"{kind}.pdf")
        pages = 1 + int(np.ceil(len(shown) / per_page))
        with PdfPages(path, metadata={"Title": f"{KINDS[kind]['title']}: every range detected ({source})"}) as pdf:
            _summary_page(pdf, kind, eps, endings, info, plt)
            for p in range(0, len(shown), per_page):
                fig, axes = plt.subplots(per_page, 1, figsize=(8.5, 11), facecolor=SURFACE, squeeze=False)
                for ax, i in zip(axes[:, 0], range(p, p + per_page)):
                    if i >= len(shown):
                        ax.axis("off")
                        continue
                    s, e, d = shown[i]
                    draw_range(ax, kind, i + 1, s, e, d, endings[i], bars, r, bar, scale)
                fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.07, hspace=0.52)
                fig.text(0.07, 0.025, f"{KINDS[kind]['title']}  ·  {source}  ·  {tf_name} candles", fontsize=8.5, color=INK2)
                fig.text(0.90, 0.025, f"page {p // per_page + 2} of {pages}", fontsize=8.5, color=INK2, ha="right")
                pdf.savefig(fig, facecolor=SURFACE)
                plt.close(fig)
        result[kind] = dict(ranges=len(eps), shown=len(shown), share=info["share"], path=path, pages=pages,
                            endings=pd.Series(endings).value_counts().to_dict() if endings else {},
                            median=float(np.median([e - s for s, e, _ in eps])) if eps else 0.0, takeout=info["takeout"])
    return result


def main():
    from smcml.data import load_mt5_csv
    ap = argparse.ArgumentParser(description="One PDF per range type, with a chart of every range detected.")
    ap.add_argument("--data", required=True, help="MT5 export (csv)")
    ap.add_argument("--last-days", type=int, default=365, help="draw ranges that start in the last N days of the file")
    ap.add_argument("--from", dest="start", default=None, help="or: first day to draw (YYYY-MM-DD)")
    ap.add_argument("--to", dest="end", default=None, help="last day to draw (YYYY-MM-DD)")
    ap.add_argument("--tf", default=None, help="build bigger candles first, e.g. 5min, 15min, 1h")
    ap.add_argument("--price-scale", type=float, default=1.0, help="multiply prices for display (0.01 if the file is in cents)")
    ap.add_argument("--max", type=int, default=None, help="draw at most this many ranges per type")
    ap.add_argument("--out", default=None, help="output folder")
    a = ap.parse_args()
    df = load_mt5_csv(a.data)
    end = pd.Timestamp(a.end) + pd.Timedelta(days=1) if a.end else df.index[-1]
    start = pd.Timestamp(a.start) if a.start else end - pd.Timedelta(days=a.last_days)
    name = os.path.splitext(os.path.basename(a.data))[0]
    out = a.out or os.path.join("outputs", "range_report", name)
    res = make_report(df, out, name, start, end, a.tf, a.price_scale, max_each=a.max)
    for kind, v in res.items():
        print(f"{kind:10s} {v['ranges']:5d} ranges, flag on {v['share']:.1%} of candles, median {v['median']:.0f} candles -> {v['path']} ({v['pages']} pages)")
        for k, n in v["endings"].items():
            print(f"{'':10s}   {n:5d}  {k}")


if __name__ == "__main__":
    main()
