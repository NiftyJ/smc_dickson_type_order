"""
Runs the whole pipeline and writes a plain-language report.

    python run_all.py                           # V75-style random simulator (no edge in it)
    python run_all.py --sim planted             # simulator with a hidden edge
    python run_all.py --data data/V75_M15.csv   # your own MT5 export
    python run_all.py --model cnn               # chart-picture CNN with focal loss (needs PyTorch)
    python run_all.py --export-review 150       # charts for you to mark TAKE / SKIP
    python run_all.py --my-calls review/my_calls.csv
    python run_all.py --pretend-calls           # demo of the my_style judge with a pretend trader

Everything printed is also saved to outputs/<run name>/report.txt
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import config as cfg
from smcml.data import load_mt5_csv, simulate
from smcml.setups import scan, feature_columns
from smcml.labels import label, target_table, chance_rate
from smcml.rule_votes import votes, rule_report
from smcml.walkforward import walk_forward, judge_table, fold_table, ranking_check
from smcml.risk import simulate_account, losing_streaks, drawdown_target
from smcml.reentry import find_reentries
from smcml.live import cfg_copy
from smcml import my_calls as mc

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


class Report:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def table(self, df, floatfmt=3):
        self(df.round(floatfmt).to_string(index=False))

    def h(self, title):
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.DATA_CSV)
    ap.add_argument("--sim", default=cfg.SIM_MODE, choices=["random", "planted"])
    ap.add_argument("--bars", type=int, default=cfg.SIM_BARS)
    ap.add_argument("--seed", type=int, default=cfg.SIM_SEED)
    ap.add_argument("--model", default=cfg.MODEL, choices=["gbm", "cnn", "cnn_deep"])
    ap.add_argument("--range-features", action="store_true",
                    help="also give the model the three range types (pause, directionless, staircase)")
    ap.add_argument("--export-review", type=int, default=0, metavar="N")
    ap.add_argument("--my-calls", default=None, metavar="CSV")
    ap.add_argument("--pretend-calls", action="store_true")
    ap.add_argument("--name", default=None, help="name of the output folder")
    ap.add_argument("--final", action="store_true", help="the one-time exam on the locked last 20%%")
    ap.add_argument("--no-bias-filter", action="store_true",
                    help="take setups regardless of D1/H4/H1 bias (biases are still recorded)")
    ap.add_argument("--cost", type=float, default=None, metavar="PRICE",
                    help="spread + commission + slippage per trade, in the data's price units")
    ap.add_argument("--plot-trades", type=int, default=0, metavar="N",
                    help="save pictures of N trades taken by the 'every setup' judge")
    args = ap.parse_args()
    cfg.MODEL = args.model
    if args.range_features:
        cfg.RANGE_FEATURES = True
    if args.no_bias_filter:
        cfg.BIAS_FILTER = False
    if args.cost is not None:
        cfg.COST_MODE, cfg.COST_PRICE = "price", args.cost

    # ------------------------------------------------------------------ data
    if args.data:
        df = load_mt5_csv(args.data)
        source = os.path.basename(args.data)
    else:
        df = simulate(args.bars, mode=args.sim, seed=args.seed)
        source = f"simulator ({args.sim}, V75-style, seed {args.seed})"
    name = args.name or ((os.path.splitext(source)[0] if args.data else f"sim_{args.sim}")
                         + f"_{cfg.MODEL}" + ("_FINAL" if args.final else ""))
    out_dir = os.path.join(cfg.OUTPUT_DIR, name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    rep = Report(os.path.join(out_dir, "report.txt"))
    started = time.time()

    rep.h("1. DATA")
    rep(f"Source: {source}")
    rep(f"Bars: {len(df):,} from {df.index[0]} to {df.index[-1]}")
    rep("Cost per trade: " + (f"{cfg.COST_PRICE} price units" if cfg.COST_MODE == "price"
                              else f"{cfg.COST_ATR} x ATR"))
    rep("Top-down bias filter: " + ("ON (" + " -> ".join(cfg.BIAS_TIMEFRAMES) + " must all agree with the trade)"
                                    if cfg.BIAS_FILTER else "OFF (biases recorded as features only)"))
    lock_bar = int(len(df) * (1 - cfg.LOCKBOX_FRACTION))
    rep(f"Lockbox (hidden unless --final): {df.index[lock_bar]} onwards")
    if args.final:
        rep("*** FINAL EXAM: testing ONCE on the locked data. If you change anything after")
        rep("*** seeing this and run --final again, this result no longer means anything.")

    # ------------------------------------------------------------------ setups + outcomes
    rep.h("2. SETUPS FOUND BY THE STATE MACHINE")
    cands = scan(df, cfg)
    if cands.empty:
        rep("No setups found. Check the data, or loosen the settings in config.py.")
        return
    lab_all = label(cands, df, cfg)
    lab_all = pd.concat([lab_all, votes(lab_all, cfg)], axis=1)
    range_cols = []
    if getattr(cfg, "RANGE_FEATURES", False):
        from wyckoff.range_types import range_features, RANGE_FEATURES as range_cols
        lab_all = pd.concat([lab_all, range_features(lab_all, df)], axis=1)
    # Everything except the final exam only sees setups that finished before the lockbox.
    lab = lab_all[lab_all["t_exit"] < lock_bar].copy() if cfg.LOCKBOX_FRACTION > 0 else lab_all
    lab.to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
    done = lab[lab["complete"]]
    rep(f"Setups: {len(lab):,}  (long {int((lab.direction == 1).sum()):,}, short {int((lab.direction == -1).sum()):,})")
    rep(f"Limit orders that got filled: {done['filled'].mean():.0%}")
    rep(f"Setups too close to the end of the data to grade (ignored): {int((~lab['complete']).sum())}")

    rep.h("3. RAW RESULTS: EVERY FILLED SETUP, NO FILTER")
    tt = target_table(lab, cfg)
    rep.table(tt)
    rep("")
    rep("How to read this: 'random_walk_win_rate' is what ANY entry rule gets on a pure")
    rep("random walk with no time limit (1 in target+1, a bit less with break-even on). Winning more often than that,")
    rep("after costs, over hundreds of trades, is what an edge looks like.")

    # ------------------------------------------------------------------ rule report
    rep.h("4. RULE REPORT (first part of history only)")
    cut = lab["t_place"].quantile(cfg.RULE_REPORT_FRACTION)
    rr = rule_report(lab[lab["t_place"] <= cut], cfg)
    rr.to_csv(os.path.join(out_dir, "rule_report.csv"), index=False)
    rep.table(rr)
    rep("")
    rep(f"Base {cfg.TARGET_R}R win rate in this period: {rr.attrs['base_win_rate']:.1%}")
    rep(f"Setups where at least one rule says YES and another says NO: {rr.attrs['share_with_conflict']:.0%}")
    rep("A useful rule has win_if_yes clearly above win_if_no. Rules where they are about")
    rep("equal are doing nothing; delete them. Contradicting rules are fine here: the")
    rep("model decides how much each one is worth.")

    # ------------------------------------------------------------------ your calls
    calls = None
    if args.my_calls:
        calls = mc.load_calls(args.my_calls, lab)
    elif args.pretend_calls:
        calls = mc.pretend_calls(done, seed=args.seed)
    if calls is not None:
        rep.h("5a. YOUR CALLS, BLIND-SCORED" + ("  (PRETEND TRADER - demo only)" if args.pretend_calls else ""))
        bs = mc.blind_score(lab, calls, cfg)
        rw = bs["random_walk_win_rate"]
        for k in ("TAKE", "SKIP"):
            b = bs[k]
            if not b["filled"]:
                rep(f"{k}: {b['calls']} calls, none filled")
                continue
            lo, hi = b["win_rate_95pct_range"]
            rep(f"{k}: {b['calls']} calls, {b['filled']} filled, {b['wins']} reached {cfg.TARGET_R}R "
                f"= {b['win_rate']:.1%} (about 1 in {1 / max(b['win_rate'], 1e-9):.0f}; "
                f"95% range {lo:.1%} to {hi:.1%}), average {b['avg_R']:+.2f}R per call")
            rep(f"      chance of doing at least this well on a random walk: p = {b['p_value_vs_random_walk']:.3g}")
        rep(f"Random walk: {rw:.1%} (1 in {cfg.TARGET_R + 1}). p below 0.01 = very unlikely to be luck.")
        rep("Your TAKE calls should clearly beat your SKIP calls AND the random walk.")

    # ------------------------------------------------------------------ walk-forward
    if args.final:
        rep.h(f"5. FINAL EXAM ON THE LOCKBOX (trained on everything before it, model = {cfg.MODEL})")
        exam_set, test_from = lab_all[lab_all["complete"]], lock_bar
    else:
        rep.h(f"5. WALK-FORWARD EXAM ({cfg.N_FOLDS} test periods, model = {cfg.MODEL})")
        exam_set, test_from = done, None
    feature_cols = feature_columns(cfg) + [c for c in lab.columns if c.startswith("vote_")] + list(range_cols)
    images = None
    if cfg.MODEL in ("cnn", "cnn_deep"):
        from smcml.chart_images import images_for
        images = images_for(exam_set.sort_values(["t_place", "direction"], kind="stable").reset_index(drop=True), df)
    res = walk_forward(exam_set, cfg, feature_cols, images=images, my_calls=calls, seed=args.seed,
                       test_from=test_from)
    res.to_csv(os.path.join(out_dir, "walkforward_rows.csv"), index=False)
    judges = ["all", "rules", "model", "placebo"] + (["my_style"] if calls is not None else [])
    jt = judge_table(res, judges)
    rep("All numbers below are OUT-OF-SAMPLE: each setup was judged by a model that")
    rep("had never seen its period. Unfilled orders count as 0R. Costs are included.")
    rep("")
    rep.table(jt)
    rep("")
    rep("p_luck = chance of at least this many wins if the judge were no better than")
    rep("taking every setup. With 20R targets, luck is HUGE: a few lucky winners swing")
    rep("the total by 100R. Look at p_luck and the placebo row before believing total_R.")
    rep("")
    rep("Model, test period by test period:")
    ft = fold_table(res, "model")
    rep.table(ft)
    rep("")
    rep("Does the model RANK setups well? (no threshold involved)")
    ranks = {}
    for j in ("model", "placebo"):
        gap, p, tab = ranking_check(res, j, seed=args.seed)
        ranks[j] = (gap, p)
        rep(f"  {j}: top third minus bottom third = {gap:+.2f}R per setup, p = {p:.3f}")
        rep.table(tab.assign(judge=j)[["judge", "group", "setups", "win_rate", "avg_R"]])
    rep("A model with real skill has a clearly positive gap with p below 0.05,")
    rep("and the placebo does not.")
    imp = res.attrs.get("last_model")
    if imp is not None and hasattr(imp, "importance"):
        top = imp.importance(feature_cols)
        if top:
            rep("")
            rep("What the last model paid most attention to: " +
                ", ".join(f"{k} ({v})" for k, v in list(top.items())[:8]))

    # ------------------------------------------------------------------ account
    rep.h("6. ACCOUNT SIMULATION WITH THE RISK GUARD")
    bar_day = df.index.normalize().to_numpy()
    rep(f"Risk per trade {cfg.RISK_PER_TRADE_PCT}% | max daily loss {cfg.MAX_DAILY_LOSS_PCT}% | "
        f"max open {cfg.MAX_OPEN_TRADES}")
    rep(describe_rules(cfg))
    tables = {}

    def tables_for(c):
        """Setups graded with c's break-even rule, plus their possible re-entries."""
        k = (getattr(c, "BREAKEVEN_AT_R", None), c.REENTRY, getattr(c, "REENTRY_MAX_SHOTS", 2), c.TARGET_R,
             c.TARGET_R_IN_DRAWDOWN)
        if k not in tables:
            same_be = getattr(c, "BREAKEVEN_AT_R", None) == getattr(cfg, "BREAKEVEN_AT_R", None)
            base = exam_set if same_be else label(cands.loc[exam_set.index], df, c)
            re = None
            if c.REENTRY:
                re = find_reentries(base, df, c)
                if len(re) and not args.final:          # normal runs must not use results from the lockbox
                    ends = re[[f"t_exit_{t}" for t in sorted({c.TARGET_R, c.TARGET_R_IN_DRAWDOWN})]].max(axis=1)
                    re = re[ends < lock_bar]
            tables[k] = (base, re)
        return tables[k]

    def run_rules(c, judge):
        base, re = tables_for(c)
        rows = base[base["candidate_id"].isin(res.loc[res[f"take_{judge}"], "candidate_id"])]
        fn = drawdown_target(c) if c.DYNAMIC_TARGET else (lambda dd, t=c.TARGET_R: t)
        return simulate_account(rows, bar_day, c, target_for=fn, reentries=re, prices=df)

    curves = {}
    for j in ("all", "model", "placebo"):
        trades, blocked = run_rules(cfg, j)
        if trades.empty:
            rep(f"{j:9s}: no trades taken")
            continue
        trades["time"] = df.index[trades["t_place"].to_numpy()]
        eq = trades["equity_pct"].to_numpy()
        dd = np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq])
        rep(f"{j:9s}: {len(trades)} orders, result {eq[-1]:+.1f}% of starting balance, "
            f"worst drawdown {dd:.1f}%, blocked by guard {({k: v for k, v in blocked.items() if v})}")
        curves[j] = trades[["time", "equity_pct"]]
        trades.to_csv(os.path.join(out_dir, f"trades_{j}.csv"), index=False)
        if j == "all" and args.plot_trades:
            from smcml.plot_trades import plot_sample
            first = trades[~trades["is_reentry"]].copy()
            first["outcome_R"], first["t_exit"] = first["R_used"], first["t_exit_used"]
            first["y"] = [getattr(r, f"hit_{int(r.target_R_used)}") for r in first.itertuples(index=False)]
            paths = plot_sample(first, df, os.path.join(out_dir, "trade_charts"), cfg,
                                n=args.plot_trades, seed=args.seed)
            rep(f"           saved {len(paths)} trade pictures in {os.path.join(out_dir, 'trade_charts')}")
    _plot(curves, os.path.join(out_dir, "equity.png"), source)

    rep("")
    rep("Each rule switched off one at a time (same setups, same test periods):")
    rows = []
    for name, changes in rule_variants(cfg):
        c = cfg_copy(cfg, **changes)
        for j in ("all", "model"):
            tr, blocked = run_rules(c, j)
            if tr.empty:
                continue
            eq = tr["equity_pct"].to_numpy()
            f = tr[tr["filled"] == 1]
            hit = [getattr(r, f"hit_{int(r.target_R_used)}") == 1 for r in f.itertuples(index=False)]
            rows.append({"judge": j, "rules": name, "orders": len(tr), "result_pct": eq[-1],
                         "worst_drawdown_pct": np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]),
                         "targets_hit": int(sum(hit)), "stopped": int((f["R_used"] < -0.99).sum()),
                         "break_even_exits": int(((f["R_used"].abs() < 0.5) & ~np.array(hit, bool)).sum()),
                         "reentries": int(tr["is_reentry"].sum()),
                         "locked_out": int(sum(v for k, v in blocked.items() if k not in ("daily_loss", "too_many_open")))})
    if rows:
        rep.table(pd.DataFrame(rows), 1)
    rep("If switching a rule OFF gives a better result than 'your settings', that rule is costing money on")
    rep("this history. One history can flatter or punish any rule by luck; check it on more data / iterate.py.")

    # ------------------------------------------------------------------ verdict
    rep.h("7. VERDICT")
    m = jt.set_index("judge")
    ls = losing_streaks(max(tt.loc[tt.target_R == cfg.TARGET_R, "win_rate"].iloc[0], 0.01))
    verdict(rep, m, ft, cfg, ranks)
    rep("")
    rep(f"Losing streaks to expect at the raw {cfg.TARGET_R}R win rate over 1,000 trades: "
        f"typically {ls['typical']} in a row, {ls['bad_1_in_10']} in a bad case.")
    rep(f"Finished in {time.time() - started:.0f}s. Files are in {out_dir}/")


def describe_rules(cfg):
    parts = [f"targets {cfg.TARGET_R}R" + (f", {cfg.TARGET_R_IN_DRAWDOWN}R when more than {cfg.DD_SWITCH_PCT:g}% "
                                           f"below the peak" if cfg.DYNAMIC_TARGET else "")]
    if getattr(cfg, "BREAKEVEN_AT_R", None):
        parts.append(f"stop to break-even at +{cfg.BREAKEVEN_AT_R}R")
    parts.append(f"up to {cfg.REENTRY_MAX_SHOTS} shots per setup" if cfg.REENTRY else "1 shot per setup")
    if cfg.COOLDOWN:
        why = (f"{cfg.COOLDOWN_LOSING_DAYS} losing days within {cfg.COOLDOWN_WINDOW_DAYS}"
               if cfg.COOLDOWN_TRIGGER == "losing_days" else f"{cfg.COOLDOWN_AFTER_STOPS} stop-outs in a row")
        parts.append(f"range lock {cfg.COOLDOWN_DAYS} days after {why}"
                     + (" (or until price breaks the range)" if cfg.COOLDOWN_ENDS_ON_BREAKOUT else ""))
    if getattr(cfg, "LOSS_BLOCK_COUNT", 0):
        parts.append(f"{cfg.LOSS_BLOCK_HOURS}h block after {cfg.LOSS_BLOCK_COUNT} losses in "
                     f"{cfg.LOSS_BLOCK_WINDOW_HOURS}h")
    if getattr(cfg, "MAX_ORDERS_PER_24H", 0):
        parts.append(f"max {cfg.MAX_ORDERS_PER_24H} orders per 24h")
    return "Your rules: " + "; ".join(parts)


def rule_variants(cfg):
    """'Your settings', then each journal rule switched off, then the old bot."""
    out = [("your settings", {})]
    if getattr(cfg, "BREAKEVEN_AT_R", None):
        out.append(("without break-even", {"BREAKEVEN_AT_R": None}))
    if cfg.REENTRY and cfg.REENTRY_MAX_SHOTS > 2:
        out.append(("2 shots max", {"REENTRY_MAX_SHOTS": 2}))
    if cfg.REENTRY:
        out.append(("1 shot (no re-entries)", {"REENTRY": False}))
    if cfg.COOLDOWN:
        out.append(("without range lock", {"COOLDOWN": False}))
    if getattr(cfg, "LOSS_BLOCK_COUNT", 0) or getattr(cfg, "MAX_ORDERS_PER_24H", 0):
        out.append(("without loss block / 24h cap", {"LOSS_BLOCK_COUNT": 0, "MAX_ORDERS_PER_24H": 0}))
    if cfg.DYNAMIC_TARGET:
        out.append((f"always {cfg.TARGET_R}R", {"DYNAMIC_TARGET": False}))
    if {10, 20} <= set(cfg.EXTRA_TARGETS) | {cfg.TARGET_R}:
        out.append(("old bot (20R/10R, 2 shots, no journal rules)",
                    {"TARGET_R": 20, "TARGET_R_IN_DRAWDOWN": 10, "BREAKEVEN_AT_R": None, "REENTRY": True,
                     "REENTRY_MAX_SHOTS": 2, "COOLDOWN": False, "LOSS_BLOCK_COUNT": 0, "MAX_ORDERS_PER_24H": 0,
                     "DYNAMIC_TARGET": True}))
    return out


def verdict(rep, m, ft, cfg, ranks):
    """Two separate questions, answered separately."""
    from scipy.stats import binomtest
    rw = chance_rate(cfg.TARGET_R, cfg)

    # A. Do the setups themselves beat a random walk?
    a = m.loc["all"]
    p_setups = binomtest(int(a["wins"]), int(a["filled"]), rw, alternative="greater").pvalue if a["filled"] else np.nan
    rep(f"A. DO THE SETUPS THEMSELVES HAVE AN EDGE? (every setup, test periods only)")
    rep(f"   {int(a['wins'])} wins in {int(a['filled'])} filled trades = {a['win_rate']:.1%} "
        f"vs {rw:.1%} on a random walk; average {a['avg_R']:+.2f}R per order.")
    rep(f"   Chance of doing this well on a pure random walk: p = {p_setups:.3g}")
    setups_edge = np.isfinite(p_setups) and p_setups < 0.01 and a["avg_R"] > 0
    rep("   -> YES, well beyond luck." if setups_edge else
        "   -> NOT SHOWN. This is within what a random walk produces by luck.")
    rep("   (The random-walk rate is exact for V75-style indices. On real markets, which")
    rep("   can trend, treat it as a rough yardstick.)")
    rep("")

    # B. Does the model pick the better setups?
    model_R = m.loc["model", "total_R"]
    (gap, p_rank), (gap_pl, _) = ranks["model"], ranks["placebo"]
    good_folds = int((ft["total_R"] > 0).sum())
    mm = m.loc["model"]
    p_rw = (binomtest(int(mm["wins"]), int(mm["filled"]), rw, alternative="greater").pvalue
            if mm["filled"] else np.nan)
    checks = [
        (f"the setups it chose beat the random-walk win rate beyond luck (p = {p_rw:.3g}, need < 0.01)",
         np.isfinite(p_rw) and p_rw < 0.01),
        ("its top third of setups beats its bottom third, beyond luck (p < 0.05)",
         np.isfinite(p_rank) and p_rank < 0.05 and gap > 0),
        ("it ranks better than the placebo model trained on shuffled results",
         np.isfinite(gap) and gap > (gap_pl if np.isfinite(gap_pl) else -np.inf)),
        ("the setups it chose made money out of sample", model_R > 0),
        (f"profitable in most test periods ({good_folds}/{len(ft)})", good_folds > len(ft) / 2),
    ]
    rep("B. DOES THE MODEL PICK THE BETTER SETUPS?")
    for text, ok in checks:
        rep(f"   [{'x' if ok else ' '}] {text}")
    model_adds = all(ok for _, ok in checks)
    rep("   -> YES." if model_adds else "   -> NOT SHOWN. Its choices are not clearly better than taking every setup.")
    rep("")

    if model_adds:
        rep("SUMMARY: POSSIBLE EDGE, AND THE MODEL ADDS TO IT. Next: run it on a demo")
        rep("account for a few months and check live results match before real money.")
    elif setups_edge:
        rep("SUMMARY: THE SETUPS LOOK GOOD ON THEIR OWN, THE MODEL ADDS NOTHING YET.")
        rep("Try better features, more history, or skip the model. Confirm on demo first.")
    else:
        rep("SUMMARY: NO RELIABLE EDGE FOUND. Do not trade this with real money.")


def _plot(curves, path, title):
    if not curves:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    styles = {"all": ("#999999", "every setup"), "model": ("#1d3557", "model"),
              "placebo": ("#e76f51", "placebo (learned nothing)")}
    for j, d in curves.items():
        color, lab = styles[j]
        ax.plot(pd.to_datetime(d["time"]), d["equity_pct"], color=color, label=lab, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("% of starting balance")
    ax.set_title(f"Out-of-sample account result with the risk guard: {title}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def export_only():
    """--export-review N: make the review folder and stop."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.DATA_CSV)
    ap.add_argument("--sim", default=cfg.SIM_MODE, choices=["random", "planted"])
    ap.add_argument("--bars", type=int, default=cfg.SIM_BARS)
    ap.add_argument("--seed", type=int, default=cfg.SIM_SEED)
    ap.add_argument("--export-review", type=int, default=150)
    ap.add_argument("--no-bias-filter", action="store_true")
    args, _ = ap.parse_known_args()
    if args.no_bias_filter:
        cfg.BIAS_FILTER = False
    df = load_mt5_csv(args.data) if args.data else simulate(args.bars, mode=args.sim, seed=args.seed)
    lab = label(scan(df, cfg), df, cfg)
    lab = lab[lab["t_exit"] < int(len(df) * (1 - cfg.LOCKBOX_FRACTION))]   # keep the lockbox unseen
    path = mc.export_review(lab, df, "review", n=args.export_review, seed=args.seed)
    print(f"Saved {args.export_review} charts and {path}.")
    print("Write TAKE or SKIP in the my_call column, then run:")
    same = f' --data "{args.data}"' if args.data else f" --sim {args.sim} --seed {args.seed}"
    same += " --no-bias-filter" if args.no_bias_filter else ""
    print("  python run_all.py --my-calls review/my_calls.csv" + same)


if __name__ == "__main__":
    if any(a.startswith("--export-review") for a in sys.argv[1:]):
        export_only()
    else:
        main()
