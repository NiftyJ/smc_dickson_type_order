"""
Train and test the range model.

    python -m wyckoff.train --data "data/V75_M15.csv"                       starter boxes
    python -m wyckoff.train --data "data/V75_M15.csv" --labels my_ranges.json   YOUR boxes

History split (same as the rest of the bot):
    first 60%   training
    next 20%    test: the model has never seen it; every number printed is from here
    last 20%    lockbox, untouched
What it prints:
  1. how well the model says "in a range right now" on the test period, next to the
     simple rule in wyckoff/ranges.py (range_now) that only uses closed candles
  2. whether that matters: SMC setups taken while the model says "range" vs the rest
Saves the model to wyckoff/range_model.pt and a few pictures to outputs/wyckoff/.
Needs PyTorch. A GPU is optional (it runs on CPU in minutes; use RunPod for big data).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import config as cfg
from smcml.bias import resample, bar_length
from smcml.data import load_mt5_csv, simulate
from smcml.labels import label
from smcml.setups import scan
from wyckoff.model import WINDOW, HEIGHT, render, make_net
from wyckoff.ranges import find_ranges, load_boxes, in_range_mask, range_now, value_at_decision


def build(bars, mask, idx):
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    X = np.zeros((len(idx), 1, HEIGHT, 2 * WINDOW), dtype=np.uint8)
    Y = np.zeros((len(idx), WINDOW), dtype=np.float32)
    for n, t in enumerate(idx):
        X[n, 0] = render(h, l, c, t)
        Y[n] = mask[t - WINDOW + 1:t + 1]
    return X, Y


def auc(score, y):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def perm_p(x, m, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    obs = x[~m].mean() - x[m].mean()
    cnt = sum((x[~mm].mean() - x[mm].mean()) >= obs for mm in (rng.permutation(m) for _ in range(n)))
    return (cnt + 1) / (n + 1)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.DATA_CSV)
    ap.add_argument("--sim", default=None, choices=[None, "random", "planted"])
    ap.add_argument("--labels", default=None, help="your boxes from label_tool.html (JSON)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2, help="one training picture every N candles")
    ap.add_argument("--cost", type=float, default=None)
    args = ap.parse_args()
    if args.cost is not None:
        cfg.COST_MODE, cfg.COST_PRICE = "price", args.cost
    torch.manual_seed(0)
    np.random.seed(0)

    df = load_mt5_csv(args.data) if args.data else simulate(cfg.SIM_BARS, mode=args.sim or "random", seed=cfg.SIM_SEED)
    if args.labels:
        tf, boxes, reviewed = load_boxes(args.labels, with_reviewed=True)
        cfg.RANGE_TF = tf
        src = f"your labels ({len(boxes)} boxes)"
        if reviewed:                         # only the part you checked counts
            df = df[df.index <= pd.Timestamp(reviewed)]
            src += f", checked up to {reviewed}"
    else:
        boxes = find_ranges(df, cfg)
        src = f"starter labels from wyckoff/ranges.py ({len(boxes)} boxes)"
    bars = resample(df, cfg.RANGE_TF)
    mask = in_range_mask(bars.index, boxes).astype(np.float32)
    n = len(bars)
    dev, lock = int(n * cfg.DEV_FRACTION), int(n * (1 - cfg.LOCKBOX_FRACTION))
    tr_idx = np.arange(WINDOW - 1, dev, args.stride)
    te_idx = np.arange(max(dev, WINDOW - 1), lock)
    print(f"{len(bars):,} {cfg.RANGE_TF} candles, {src}; {mask[:lock].mean():.0%} of candles are in a range")
    print(f"training pictures: {len(tr_idx):,} (up to {bars.index[dev - 1]:%Y-%m-%d}), "
          f"test pictures: {len(te_idx):,} ({bars.index[dev]:%Y-%m-%d} to {bars.index[lock - 1]:%Y-%m-%d})")
    Xtr, Ytr = build(bars, mask, tr_idx)
    Xte, Yte = build(bars, mask, te_idx)

    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_net().to(dev_)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pos = float(Ytr.mean())
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3), device=dev_))
    w_cols = torch.ones(WINDOW, device=dev_)
    w_cols[-20:] = 3.0                                     # "right now" matters most
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for b in range(0, len(perm), 128):
            j = perm[b:b + 128]
            x = torch.tensor(Xtr[j], dtype=torch.float32, device=dev_) / 255
            y = torch.tensor(Ytr[j], device=dev_)
            logit = net(x)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, y, pos_weight=lossf.pos_weight, reduction="none") * w_cols).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot / len(perm):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    net.eval()
    with torch.no_grad():
        P = np.concatenate([torch.sigmoid(net(torch.tensor(Xte[b:b + 512], dtype=torch.float32, device=dev_) / 255))
                            .cpu().numpy() for b in range(0, len(Xte), 512)])
    now_p, now_y = P[:, -1], Yte[:, -1]
    rule = range_now(df, cfg).to_numpy()[te_idx]
    whole = auc(P.ravel(), Yte.ravel())
    print("\n1. DOES IT SEE RANGES? (test period only)")
    print(f"   every candle in the picture:   AUC {whole:.3f}   (0.5 = guessing, 1.0 = perfect)")
    print(f"   'in a range RIGHT NOW':         AUC {auc(now_p, now_y):.3f}")
    k = max(int(rule.sum()), 1)
    top = now_p >= np.sort(now_p)[-k]
    prec = lambda m: now_y[m].mean() if m.any() else float("nan")
    print(f"   flagging the same number of candles as the simple rule ({k}): "
          f"right {prec(top):.0%} of the time, simple rule right {prec(rule.astype(bool)):.0%}, "
          f"base rate {now_y.mean():.0%}")

    # 2. does it matter for SMC trades?
    lab = label(scan(df, cfg), df, cfg)
    f = lab[(lab.filled == 1) & lab.complete]
    start, end = bars.index[dev] + pd.Timedelta(cfg.RANGE_TF), bars.index[lock - 1]
    f = f[(f.time >= start) & (f.time <= end)]
    series = pd.Series(np.nan, index=bars.index)
    series.iloc[te_idx] = now_p
    p_at = value_at_decision(series.fillna(0), f.time, bar_length(df.index), cfg.RANGE_TF)
    print(f"\n2. DOES IT MATTER FOR SMC TRADES? ({len(f)} filled setups in the test period)")
    if len(f) >= 20:
        for name, m in (("model says range (p > 0.5)", p_at > 0.5),
                        ("model's top 25% range score", p_at >= np.quantile(p_at, 0.75))):
            if m.sum() < 5 or (~m).sum() < 5:
                print(f"   {name}: too few setups on one side")
                continue
            for T in sorted({cfg.TARGET_R_IN_DRAWDOWN, cfg.TARGET_R}):
                x, h = f[f"R_{T}"].to_numpy(), f[f"hit_{T}"].to_numpy()
                print(f"   {name}, {T}R: {m.sum()} setups in range hit {h[m].mean():.1%}, avg {x[m].mean():+.2f}R | "
                      f"{(~m).sum()} others hit {h[~m].mean():.1%}, avg {x[~m].mean():+.2f}R | "
                      f"chance the others look this much better by luck: p = {perm_p(x, m):.3f}")
        print("   Useful only if the 'in range' setups are clearly worse (p below 0.05) on data it never saw.")
    torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), "range_model.pt"))

    out = os.path.join(cfg.OUTPUT_DIR, "wyckoff")
    os.makedirs(out, exist_ok=True)
    pictures(bars, P, Yte, te_idx, out)
    print(f"\nSaved the model (wyckoff/range_model.pt) and example pictures in {out}/")


def pictures(bars, P, Y, te_idx, out, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smcml.plot_trades import _candles
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    picks = np.linspace(0, len(te_idx) - 1, n + 2).astype(int)[1:-1]
    for i in picks:
        t = te_idx[i]
        s = t - WINDOW + 1
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        _candles(a1, o, h, l, c, s, t)
        for j in range(WINDOW):
            if Y[i, j] > 0.5:
                a1.axvspan(s + j - 0.5, s + j + 0.5, color="#f4a261", alpha=0.18, lw=0)
        a2.plot(np.arange(s, t + 1), P[i], color="#1d3557")
        a2.fill_between(np.arange(s, t + 1), 0, Y[i], color="#f4a261", alpha=0.35, step="mid", label="labelled range")
        a2.axhline(0.5, color="#999", lw=0.8, ls="--")
        a2.set_ylim(0, 1)
        a2.set_ylabel("model: range?")
        a2.legend(loc="upper left", fontsize=8)
        a1.set_title(f"{bars.index[s]:%Y-%m-%d} to {bars.index[t]:%Y-%m-%d} ({cfg.RANGE_TF}): orange = labelled range, "
                     f"blue line below = model's range probability (last point = 'right now')", fontsize=10, loc="left")
        ticks = np.linspace(s, t, 6).astype(int)
        a2.set_xticks(ticks)
        a2.set_xticklabels([f"{bars.index[x]:%b %d}" for x in ticks], fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"range_{bars.index[t]:%Y%m%d_%H%M}.png"), dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    main()
