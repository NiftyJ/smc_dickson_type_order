"""
YOUR EYE, MEASURED.

Idea from NVIDIA's self-driving car (Bojarski et al., 2016): they never wrote rules
for "lane lines". They recorded a human driving and trained a network to copy the
steering. The trader version: you mark setups TAKE or SKIP, and a model learns to
copy your judgement ("my_style" judge in walkforward.py).

It also gives you a FAIR TEST of your own hit rate, because:
  * each chart stops at the moment the order would be placed (the future is hidden,
    so hindsight cannot help you),
  * the code, not your memory, decides what happened next, spread included.

Workflow:
  1. python run_all.py --export-review 150
       -> review/ folder with 150 chart images and review/my_calls.csv
  2. Open each image. In my_calls.csv write TAKE or SKIP in the my_call column.
     Decide from the picture only. Do not look the date up on a chart.
  3. python run_all.py --my-calls review/my_calls.csv
       -> your blind hit rate vs the random-walk rate, and the my_style judge.
"""
import os

import numpy as np
import pandas as pd

from .labels import chance_rate


def export_review(cands, df, out_dir, n=150, seed=0, bars_before=120):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(out_dir, exist_ok=True)
    pool = cands[cands["complete"]] if "complete" in cands else cands
    pick = pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("t_place")
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rows = []
    for r in pick.itertuples(index=False):
        t = int(r.t_place)
        s = max(0, t - bars_before + 1)
        x = np.arange(s, t + 1)
        fig, ax = plt.subplots(figsize=(11, 5))
        up = c[s:t + 1] >= o[s:t + 1]
        ax.vlines(x, l[s:t + 1], h[s:t + 1], color="#555555", linewidth=0.8)
        ax.vlines(x[up], o[s:t + 1][up], c[s:t + 1][up], color="#2a9d8f", linewidth=3)
        ax.vlines(x[~up], c[s:t + 1][~up], o[s:t + 1][~up], color="#e76f51", linewidth=3)
        ax.add_patch(Rectangle((r.ob_bar - 0.5, r.ob_low), t + 12 - r.ob_bar, r.ob_high - r.ob_low,
                               color="#8ab6d6", alpha=0.55, zorder=0, label="order block"))
        ax.hlines(r.entry, t - 20, t + 12, color="#1d3557", linestyles="--", label="entry (limit)")
        ax.hlines(r.stop, t - 20, t + 12, color="#c1121f", label="stop")
        ax.hlines(r.swept_level, r.sweep_bar - 15, r.sweep_bar + 2, color="#999999",
                  linestyles=":", label="swept level")
        ax.axvline(t + 0.5, color="black", linewidth=1)
        ax.text(t + 1, ax.get_ylim()[1], " future hidden", va="top", fontsize=9)
        ax.set_xlim(s - 1, t + 13)
        side = "LONG" if r.direction == 1 else "SHORT"
        ax.set_title(f"Setup {r.candidate_id}: {side}. Would you take this trade?")
        ax.set_xticks([])
        ax.legend(loc="upper left", fontsize=8)
        fname = f"setup_{r.candidate_id}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=90)
        plt.close(fig)
        rows.append({"candidate_id": r.candidate_id, "bar": t, "direction": side, "image": fname,
                     "my_call": "", "notes": ""})
    path = os.path.join(out_dir, "my_calls.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_calls(path, labeled):
    """Returns a Series candidate_id -> 1 (TAKE) / 0 (SKIP). Blank rows are ignored.

    Setups are matched by bar number and direction, so the file still works if the
    candidate numbering changes. (It will NOT match if you change the detector
    settings in config.py, because then different setups are found.)
    """
    t = pd.read_csv(path, dtype={"my_call": str})
    call = t["my_call"].fillna("").str.strip().str.upper().map({"TAKE": 1, "SKIP": 0, "1": 1, "0": 0})
    side = t["direction"].map({"LONG": 1, "SHORT": -1})
    key = pd.MultiIndex.from_arrays([t["bar"], side])
    ids = labeled.set_index(["t_place", "direction"])["candidate_id"].reindex(key)
    s = pd.Series(call.to_numpy(), index=ids.to_numpy())
    missing = int(s.index.isna().sum())
    if missing:
        print(f"Warning: {missing} rows in {path} did not match a setup (settings changed?)")
    s = s[s.index.notna() & s.notna()]
    s.index = s.index.astype(int)
    return s


def pretend_calls(cands, seed=0, share=0.4):
    """FOR THE DEMO ONLY: a pretend trader who likes big displacement candles in the
    London morning, and is right about as often as a real person (not always)."""
    rng = np.random.default_rng(seed)
    liking = (cands["displacement_atr"].to_numpy() / 2.0
              + ((cands["hour"] >= 7) & (cands["hour"] < 11)).to_numpy() * 0.6
              + rng.normal(0, 0.5, len(cands)))
    pick = cands.sample(frac=share, random_state=seed).index
    calls = (liking > np.quantile(liking, 0.6)).astype(int)
    return pd.Series(calls, index=cands["candidate_id"]).loc[cands.loc[pick, "candidate_id"]]


def blind_score(labeled, calls, cfg):
    """Your hit rate on TAKE calls vs SKIP calls vs a random walk, with a p-value."""
    from scipy.stats import binomtest
    d = labeled.set_index("candidate_id").loc[calls.index]
    d = d[d["complete"]]
    call = calls.loc[d.index]
    out = {}
    for name, flag in (("TAKE", 1), ("SKIP", 0)):
        g = d[(call == flag).to_numpy()]
        f = g[g["filled"] == 1]
        wins = int(f["y"].sum())
        entry = {"calls": len(g), "filled": len(f), "wins": wins,
                 "win_rate": wins / len(f) if len(f) else np.nan,
                 "avg_R": g["outcome_R"].mean() if len(g) else np.nan}
        if len(f):
            ci = binomtest(wins, len(f)).proportion_ci(0.95)
            entry["win_rate_95pct_range"] = (round(ci.low, 3), round(ci.high, 3))
            entry["p_value_vs_random_walk"] = binomtest(wins, len(f), chance_rate(cfg.TARGET_R, cfg),
                                                        alternative="greater").pvalue
        out[name] = entry
    out["random_walk_win_rate"] = chance_rate(cfg.TARGET_R, cfg)
    return out
