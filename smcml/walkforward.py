"""
THE EXAM: walk-forward testing. This is the part that stops you fooling yourself.

The idea: only ever test a model on a period it has never seen, the way it would
really have been used.

    history:  |---- chunk 0 ----|-- chunk 1 --|-- chunk 2 --| ... |-- chunk 5 --|
    fold 1:    train on chunk 0             -> test on chunk 1
    fold 2:    train on chunks 0-1          -> test on chunk 2
    ...
    fold 5:    train on chunks 0-4          -> test on chunk 5

Inside each training period:
  * the last VALID_FRACTION is held back to choose the score THRESHOLD
    ("take the setups scoring above X"). The threshold is picked to maximise total R
    on that held-back part. If no threshold makes money there, the judge takes NO
    trades in the next test chunk. That is the correct answer on a market with no edge.
  * PURGING: a training setup whose trade was still open when the test chunk started
    is thrown away, because its result would leak information from the test period.

Several "judges" are examined in exactly the same way, so the comparison is fair:
  all        every setup the state machine finds (no filter at all)
  rules      count of rule votes from rule_votes.py (your rules, weighed equally)
  model      the trained model from model.py
  my_style   (optional) a model trained to copy YOUR take/skip calls
  placebo    the same model trained on SHUFFLED results. It has learned nothing.
             If "model" is not clearly better than "placebo", the model has found nothing.
"""
import numpy as np
import pandas as pd

from .model import make_model, TreeModel


def choose_threshold(scores, R, min_trades):
    """Score cut-off that maximises total R. Returns +inf (= take nothing) if none is profitable."""
    if len(scores) == 0:
        return np.inf
    order = np.argsort(-scores, kind="stable")
    cum = np.cumsum(R[order])
    counts = np.arange(1, len(cum) + 1)
    cum_ok = np.where(counts >= min_trades, cum, -np.inf)
    best = int(np.argmax(cum_ok))
    if not np.isfinite(cum_ok[best]) or cum_ok[best] <= 0:
        return np.inf
    return scores[order][best]


def walk_forward(data, cfg, feature_cols, images=None, my_calls=None, seed=0, test_from=None):
    """Returns one row per TEST setup with each judge's score and decision.

    test_from=None: the usual N_FOLDS walk-forward.
    test_from=bar:  one single test period from that bar to the end (the lockbox exam).
    """
    data = data.sort_values(["t_place", "direction"], kind="stable").reset_index(drop=True)
    tp = data["t_place"].to_numpy()
    t_end = data["t_exit"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=int)
    R = data["outcome_R"].to_numpy(dtype=float)
    X = data[feature_cols]
    vote_cols = [c for c in data.columns if c.startswith("vote_")]
    rule_score = data[vote_cols].sum(axis=1).to_numpy(dtype=float) if vote_cols else np.zeros(len(data))
    rng = np.random.default_rng(seed)

    if test_from is None:
        edges = np.linspace(tp.min(), tp.max() + 1, cfg.N_FOLDS + 2).astype(int)
    else:
        edges = np.array([tp.min(), test_from, tp.max() + 1])
    out = []
    for k in range(len(edges) - 2):
        test = (tp >= edges[k + 1]) & (tp < edges[k + 2])
        past = t_end < edges[k + 1]                          # purged: finished before the test chunk
        if test.sum() == 0 or past.sum() < 50:
            continue
        val_start = np.quantile(tp[past], 1 - cfg.VALID_FRACTION)
        fit = past & (t_end < val_start)                     # purged against the validation part too
        val = past & (tp >= val_start)
        if fit.sum() < 30 or val.sum() < 10:
            continue

        def img(mask):
            return None if images is None else images[mask]

        rows = pd.DataFrame({"fold": k + 1, "idx": np.flatnonzero(test)})

        # take_all and rules
        rows["take_all"] = True                          # the "all" judge
        thr = choose_threshold(rule_score[val], R[val], cfg.MIN_TRADES_FOR_THRESHOLD)
        rows["score_rules"] = rule_score[test]
        rows["take_rules"] = rule_score[test] >= thr

        # model, and the placebo (same model, shuffled results)
        for name, labels in (("model", y[fit]), ("placebo", rng.permutation(y[fit]))):
            m = make_model(cfg.MODEL, seed).fit(X[fit], labels, img(fit))
            s_val, s_test = m.score(X[val], img(val)), m.score(X[test], img(test))
            thr = choose_threshold(s_val, R[val], cfg.MIN_TRADES_FOR_THRESHOLD)
            rows[f"score_{name}"] = s_test
            rows[f"take_{name}"] = s_test >= thr
            if name == "model":
                rows.attrs["model"] = m

        # my_style: copy YOUR calls (behaviour cloning), judged on real results
        if my_calls is not None:
            calls = my_calls.reindex(data["candidate_id"]).to_numpy(dtype=float)
            have = fit & np.isfinite(calls)
            if have.sum() >= 30 and len(np.unique(calls[have])) == 2:
                m = TreeModel(seed).fit(X[have], calls[have].astype(int))
                s_val, s_test = m.score(X[val]), m.score(X[test])
                thr = choose_threshold(s_val, R[val], cfg.MIN_TRADES_FOR_THRESHOLD)
                rows["score_my_style"] = s_test
                rows["take_my_style"] = s_test >= thr
            else:
                rows["score_my_style"] = np.nan
                rows["take_my_style"] = False
        out.append(rows)
        last_model = rows.attrs.get("model")

    if not out:
        raise RuntimeError("Not enough setups for walk-forward testing. Use more history.")
    res = pd.concat(out, ignore_index=True)
    res = pd.concat([data.loc[res["idx"]].reset_index(drop=True), res.drop(columns="idx")], axis=1)
    res.attrs["last_model"] = last_model
    return res


def summarize(R, y, filled):
    """Plain numbers for a set of orders (unfilled orders count as 0R)."""
    R, y, filled = np.asarray(R, float), np.asarray(y, float), np.asarray(filled, bool)
    n = len(R)
    if n == 0:
        return {"orders": 0, "filled": 0, "wins": 0, "win_rate": np.nan, "avg_R": np.nan,
                "total_R": 0.0, "profit_factor": np.nan, "max_drawdown_R": 0.0, "worst_losing_streak": 0}
    cum = np.cumsum(R)
    dd = np.max(np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum])
    streak = worst = 0
    for r, f in zip(R, filled):
        if not f:
            continue
        streak = streak + 1 if r < 0 else 0
        worst = max(worst, streak)
    gains, losses = R[R > 0].sum(), -R[R < 0].sum()
    return {
        "orders": n,
        "filled": int(filled.sum()),
        "wins": int(np.nansum(y[filled])),
        "win_rate": float(np.nanmean(y[filled])) if filled.any() else np.nan,
        "avg_R": float(R.mean()),
        "total_R": float(R.sum()),
        "profit_factor": float(gains / losses) if losses > 0 else np.nan,
        "max_drawdown_R": float(dd),
        "worst_losing_streak": int(worst),
    }


def judge_table(res, judges):
    """One line per judge. p_luck = chance of getting at least this many wins if the
    judge were no better than taking every setup. Below 0.05 = probably not luck."""
    from scipy.stats import binomtest
    every = res[res["filled"] == 1]
    base = every["y"].mean() if len(every) else np.nan
    rows = []
    for j in judges:
        col = f"take_{j}"
        if col not in res:
            continue
        sel = res[res[col]]
        s = summarize(sel["outcome_R"], sel["y"], sel["filled"] == 1)
        s["p_luck"] = (binomtest(s["wins"], s["filled"], base, alternative="greater").pvalue
                       if s["filled"] and 0 < base < 1 else np.nan)
        rows.append({"judge": j, **s})
    return pd.DataFrame(rows)


def fold_table(res, judge="model"):
    rows = []
    for k, g in res.groupby("fold"):
        sel = g[g[f"take_{judge}"]]
        s = summarize(sel["outcome_R"], sel["y"], sel["filled"] == 1)
        rows.append({"fold": k, "test_from": str(g["time"].min())[:10], "test_to": str(g["time"].max())[:10],
                     "setups": len(g), "taken": s["orders"], "win_rate": s["win_rate"],
                     "avg_R": s["avg_R"], "total_R": s["total_R"]})
    return pd.DataFrame(rows)


def ranking_check(res, judge="model", n_perm=2000, seed=0):
    """Threshold-free test: does the judge's TOP third of setups beat its BOTTOM third?

    Setups are split into thirds by score inside each test period (scores from
    different periods are not comparable). p = share of 2,000 random shuffles of
    the results that produce a gap at least this big. Below 0.05 = real ranking skill.
    """
    col = f"score_{judge}"
    if col not in res or res[col].isna().all():
        return np.nan, np.nan, pd.DataFrame()

    def thirds(s):
        if len(s) < 3:
            return pd.Series(1, index=s.index)
        return pd.qcut(s.rank(method="first"), 3, labels=False)

    third = res.groupby("fold")[col].transform(thirds).to_numpy()
    R = res["outcome_R"].to_numpy(float)
    folds = res["fold"].to_numpy()
    gap = R[third == 2].mean() - R[third == 0].mean()
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    beat = 0
    for _ in range(n_perm):
        Rp = R.copy()
        for g in groups:
            Rp[g] = Rp[rng.permutation(g)]
        beat += (Rp[third == 2].mean() - Rp[third == 0].mean()) >= gap
    p = (beat + 1) / (n_perm + 1)
    rows = []
    for k, name in ((0, "bottom third"), (1, "middle third"), (2, "top third")):
        sel = res[third == k]
        f = sel[sel["filled"] == 1]
        rows.append({"group": name, "setups": len(sel),
                     "win_rate": f["y"].mean() if len(f) else np.nan,
                     "avg_R": sel["outcome_R"].mean() if len(sel) else np.nan})
    return gap, p, pd.DataFrame(rows)
