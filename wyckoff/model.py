"""
THE RANGE MODEL: a small image model that looks at a chart and marks, bar by bar,
which candles are part of a range.

Input:  a picture of the last WINDOW candles on RANGE_TF, ending at the current candle
        (nothing after it is drawn). 2 pixel columns per candle: the high-low line and
        the close tick, scaled to the picture's own high and low.
Output: for every candle in the picture, the probability that it belongs to a range.
        The last one is "are we in a range right now?", which is what the bot needs.
Target: your range boxes (from the labelling page), or the starter boxes.

The network: a few 2D convolutions that read the shape of each candle and its
neighbours, then 1D convolutions along time (wide, dilated) so each candle's answer
can use a couple of weeks of context on both sides inside the picture.
"""
import numpy as np

WINDOW, HEIGHT = 120, 64


def render(h, l, c, t, window=WINDOW, height=HEIGHT):
    """(height, 2*window) uint8 picture of candles t-window+1..t (right-aligned)."""
    s = max(0, t - window + 1)
    H, L, C = h[s:t + 1], l[s:t + 1], c[s:t + 1]
    lo, hi = L.min(), H.max()
    span = hi - lo if hi > lo else 1.0
    row = lambda p: ((height - 1) - np.round((p - lo) / span * (height - 1))).astype(int)
    img = np.zeros((height, 2 * window), dtype=np.uint8)
    off = window - len(H)
    rh, rl, rc = row(H), row(L), row(C)
    for j in range(len(H)):
        x = 2 * (off + j)
        img[rh[j]:rl[j] + 1, x] = 255
        img[rc[j], x + 1] = 255
    return img


def make_net(window=WINDOW, height=HEIGHT):
    import torch.nn as nn

    class RangeNet(nn.Module):
        def __init__(self):
            super().__init__()
            def c2(i, o, pool):
                return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool)]

            def c1(i, o, dil):
                return [nn.Conv1d(i, o, 5, padding=2 * dil, dilation=dil), nn.BatchNorm1d(o), nn.ReLU()]

            self.img = nn.Sequential(*c2(1, 16, (2, 1)), *c2(16, 32, (2, 1)), *c2(32, 32, (2, 1)),
                                     *c2(32, 32, (2, 2)))                  # last pool: 2 columns -> 1 candle
            ch = 32 * (height // 16)
            self.time = nn.Sequential(*c1(ch, 64, 1), *c1(64, 64, 2), *c1(64, 64, 4), *c1(64, 64, 8),
                                      nn.Conv1d(64, 1, 1))

        def forward(self, x):                      # x: (B, 1, height, 2*window) in 0..1
            z = self.img(x)                        # (B, 32, height/16, window)
            z = z.flatten(1, 2)                    # (B, ch, window)
            return self.time(z).squeeze(1)         # (B, window) logits

    return RangeNet()
