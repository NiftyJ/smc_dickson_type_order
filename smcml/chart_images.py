"""
IDEA FROM "(RE-)IMAG(IN)ING PRICE TRENDS" (Jiang, Kelly & Xiu, Journal of Finance 2023):
let the model LOOK at the chart, the way you do.

They drew each price history as a tiny black-and-white picture (every bar = 3 pixels
wide: open tick, high-low line, close tick) and trained a CNN on those pictures.
The CNN found price patterns that beat the classic trend signals.

Here every candidate gets a picture of the last WINDOW bars before the order is placed:
  channel 0: the price bars
  channel 1: horizontal lines at the entry, the stop and the swept level
Shorts are drawn on the flipped chart, so every setup "looks like a long".
That doubles the examples of each pattern the network gets to see.

Nothing after the placement bar is ever drawn.
"""
import numpy as np

from .detectors import mirror


def render(o, h, l, c, t, levels=(), window=48, height=48):
    """Picture of bars t-window+1..t as a (2, height, 3*window) uint8 array (0 or 255)."""
    s = max(0, t - window + 1)
    O, H, L, C = o[s:t + 1], h[s:t + 1], l[s:t + 1], c[s:t + 1]
    lev = [x for x in levels if np.isfinite(x)]
    lo = min(L.min(), *lev) if lev else L.min()
    hi = max(H.max(), *lev) if lev else H.max()
    span = hi - lo if hi > lo else 1.0

    def row(p):
        return (height - 1) - np.round((p - lo) / span * (height - 1)).astype(int)

    img = np.zeros((2, height, 3 * window), dtype=np.uint8)
    off = window - len(O)                     # right-align if there is less history
    for j in range(len(O)):
        x = 3 * (off + j)
        img[0, row(O[j]), x] = 255
        img[0, row(H[j]):row(L[j]) + 1, x + 1] = 255
        img[0, row(C[j]), x + 2] = 255
    for p in lev:
        img[1, row(p), :] = 255
    return img


def images_for(cands, df, window=48, height=48):
    """One picture per candidate, shape (n, 2, height, 3*window)."""
    arrs = {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    flipped = {d: mirror(arrs["open"], arrs["high"], arrs["low"], arrs["close"], d) for d in (+1, -1)}
    out = np.zeros((len(cands), 2, height, 3 * window), dtype=np.uint8)
    for i, r in enumerate(cands.itertuples(index=False)):
        o, h, l, c = flipped[r.direction]
        out[i] = render(o, h, l, c, r.t_place,
                        levels=(r.entry_f, r.stop_f, r.direction * r.swept_level),
                        window=window, height=height)
    return out
