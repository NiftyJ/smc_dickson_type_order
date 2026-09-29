"""
THE LOSS-REVIEW LOOP: look at the losses, change the code, and find out honestly whether
the change helped.

    python iterate.py start --data "data/V75_M15.csv"      once: the starting point + loss charts
        (look at experiments/000_baseline/losses/, write why each one lost in
         loss_notes.csv, spot a pattern, change the code: setups.py, detectors.py,
         rule_votes.py or config.py)
    python iterate.py check "skip setups that sweep into an H4 order block"
    python iterate.py log                                   every attempt so far
    python iterate.py restore 3                             put the code of attempt 3 back

Your history is split in three:
    DEVELOPMENT  first 60%   the losses you look at and design changes from
    VALIDATION   next 20%    never charted; every change is scored here
    LOCKBOX      last 20%    not touched here at all (run_all.py --final, once, at the end)

A change is KEPT only if
    1. all the tests still pass (nothing uses the future),
    2. it doesn't make the DEVELOPMENT period worse (the trades it was designed on),
    3. it makes more R on the VALIDATION period, which it wasn't designed on, and
    4. it doesn't make RANDOM prices look profitable. On a random walk nothing can beat
       chance, so a change that seems to is using the future or fitting noise.

Why so strict: removing the losses you are looking at is always possible (add a rule
that happens to exclude them). The question is whether the rule also works on trades
you haven't looked at. Every change that doesn't is a way to lose money live.
"""
import argparse
import datetime as dt
import json
import os
import re as regex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import config as cfg
from smcml.bias import bar_length
from smcml.data import load_mt5_csv, simulate
from smcml.labels import label, chance_rate
from smcml.live import cfg_copy
from smcml.reentry import find_reentries
from smcml.risk import simulate_account, drawdown_target
from smcml.setups import scan

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
META = os.path.join(EXP, "meta.json")
LOG = os.path.join(EXP, "log.csv")
RANDOM_SEEDS, RANDOM_BARS = 6, 40_000


# ----------------------------------------------------------------------------- the data
def load_data(meta):
    if meta.get("data"):
        return load_mt5_csv(meta["data"])
    return simulate(meta["bars"], mode=meta["sim"], seed=meta["seed"])


def apply_cost(meta):
    if meta.get("cost") is not None:
        cfg.COST_MODE, cfg.COST_PRICE = "price", meta["cost"]


def periods(n):
    dev_end = int(n * getattr(cfg, "DEV_FRACTION", 0.6))
    lock = int(n * (1 - cfg.LOCKBOX_FRACTION))
    return {"development": (0, dev_end), "validation": (dev_end, lock)}


# ----------------------------------------------------------------------------- scoring
def run_history(df):
    """The bot's trades in each period, exactly as the risk guard would take them."""
    cands = scan(df, cfg)
    if cands.empty:
        return cands, {p: pd.DataFrame() for p in periods(len(df))}
    lab = label(cands, df, cfg)
    re = find_reentries(lab, df, cfg) if cfg.REENTRY else None
    bar_day = df.index.normalize().to_numpy()
    target_for = drawdown_target(cfg) if cfg.DYNAMIC_TARGET else None
    tcols = [f"t_exit_{t}" for t in sorted({cfg.TARGET_R, cfg.TARGET_R_IN_DRAWDOWN})]
    out = {}
    for name, (a, b) in periods(len(df)).items():
        sel = lab[lab.complete & (lab.t_place >= a) & (lab.t_exit < b)]
        r = None
        if re is not None and len(re):
            r = re[(re.t_place >= a) & (re[tcols].max(axis=1) < b)]
        tr, _ = simulate_account(sel, bar_day, cfg, target_for=target_for, reentries=r, prices=df)
        if not tr.empty:
            tr["time"] = df.index[tr.t_place.to_numpy()]
            tr["key"] = [f"{t:%Y-%m-%d %H:%M}|{'L' if d == 1 else 'S'}|{'re' if x else 'setup'}"
                         for t, d, x in zip(tr.time, tr.direction, tr.is_reentry)]
            tr["hit"] = [int(getattr(x, f"hit_{int(x.target_R_used)}") == 1) for x in tr.itertuples(index=False)]
        out[name] = tr
    return lab, out


def _tail_prob(ps, k):
    """P(at least k wins) when trade i wins with probability ps[i] (exact)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def summarize(tr):
    if tr is None or tr.empty:
        return dict(orders=0, filled=0, targets_hit=0, stopped=0, total_R=0.0, result_pct=0.0,
                    worst_dd_pct=0.0, hit_rate=None, chance_rate=None, p_luck=None)
    f = tr[tr.filled == 1]
    ps = np.array([chance_rate(t, cfg) for t in f.target_R_used.to_numpy(float)])
    wins = int(f.hit.sum())
    eq = np.r_[0, np.cumsum(tr.R_used.to_numpy(float) * cfg.RISK_PER_TRADE_PCT)]
    return dict(orders=int(len(tr)), filled=int(len(f)), targets_hit=wins, stopped=int((f.R_used < -0.99).sum()),
                total_R=round(float(tr.R_used.sum()), 1), result_pct=round(float(eq[-1]), 1),
                worst_dd_pct=round(float(np.max(np.maximum.accumulate(eq) - eq)), 1),
                hit_rate=round(wins / len(f), 4) if len(f) else None,
                chance_rate=round(float(ps.mean()), 4) if len(f) else None,
                p_luck=round(_tail_prob(ps, wins), 4) if len(f) else None)


def random_check(bar_minutes=15):
    """Every setup on random prices. Nothing may beat chance here."""
    from scipy.stats import binom
    n = hits = 0
    Rs = []
    for s in range(RANDOM_SEEDS):
        df = simulate(RANDOM_BARS, mode="random", seed=9000 + s, bar_minutes=bar_minutes)
        cands = scan(df, cfg)
        if cands.empty:
            continue
        lab = label(cands, df, cfg)
        f = lab[(lab.filled == 1) & lab.complete]
        n += len(f)
        hits += int((f[f"hit_{cfg.TARGET_R}"] == 1).sum())
        Rs.extend(f.outcome_R.tolist())
    chance = chance_rate(cfg.TARGET_R, cfg)
    p = float(binom.sf(hits - 1, n, chance)) if n else 1.0
    return dict(trades=n, targets_hit=hits, hit_rate=round(hits / n, 4) if n else None, chance_rate=round(chance, 4),
                p=round(p, 4), avg_R=round(float(np.mean(Rs)), 3) if Rs else None, ok=bool(p >= 0.01))


def run_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-x"], cwd=HERE, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    return r.returncode == 0, last, r.stdout[-3000:]


# ----------------------------------------------------------------------------- pictures
def loss_charts(tr, lab, df, folder, n_losses=30, n_wins=6):
    from smcml.plot_trades import plot_trade, plot_simple
    os.makedirs(folder, exist_ok=True)
    notes = []
    losses = tr[(tr.filled == 1) & (tr.R_used < -0.99)]
    wins = tr[tr.hit == 1]
    pick_l = losses.iloc[np.unique(np.linspace(0, len(losses) - 1, min(n_losses, len(losses))).astype(int))] \
        if len(losses) else losses
    pick_w = wins.iloc[np.unique(np.linspace(0, len(wins) - 1, min(n_wins, len(wins))).astype(int))] \
        if len(wins) else wins
    for kind, rows in (("loss", pick_l), ("WIN", pick_w)):
        for r in rows.itertuples(index=False):
            tr_R = int(r.target_R_used)
            name = f"{kind}_{r.time:%Y%m%d_%H%M}_{'long' if r.direction == 1 else 'short'}" + \
                   ("_reentry" if r.is_reentry else "") + ".png"
            path = os.path.join(folder, name)
            try:
                if r.is_reentry:
                    o = dict(key=r.key, direction=r.direction, placed_time=r.time, entry=r.entry, stop=r.stop,
                             tp=r.entry + r.direction * tr_R * r.risk, target_R=tr_R, is_reentry=True,
                             fill_time=df.index[int(r.t_fill)] if r.filled == 1 else None,
                             exit_time=df.index[int(r.t_exit_used)], R=r.R_used,
                             exit_reason="sl" if r.R_used < -0.99 else "tp" if r.hit else "time")
                    plot_simple(df, o, path)
                else:
                    row = r._replace(outcome_R=r.R_used, y=r.hit, t_exit=r.t_exit_used)
                    plot_trade(row, df, path, cfg_copy(cfg, TARGET_R=tr_R))
            except Exception as e:
                print(f"  (chart for {r.key} failed: {e})")
                continue
            if kind == "loss":
                notes.append(dict(key=r.key, chart=name, kind="re-entry" if r.is_reentry else "setup",
                                  target_R=tr_R, R=round(r.R_used, 2), why_it_lost=""))
    pd.DataFrame(notes, columns=["key", "chart", "kind", "target_R", "R", "why_it_lost"]).to_csv(
        os.path.join(folder, "loss_notes.csv"), index=False)
    return len(pick_l), len(pick_w)


# ----------------------------------------------------------------------------- bookkeeping
def read_meta():
    if not os.path.exists(META):
        sys.exit("No experiments yet. Start with:  python iterate.py start --data \"data/your_export.csv\"")
    with open(META) as fh:
        return json.load(fh)


def write_meta(meta):
    with open(META, "w") as fh:
        json.dump(meta, fh, indent=1)


def snapshot(folder):
    code = os.path.join(folder, "code")
    os.makedirs(os.path.join(code, "smcml"), exist_ok=True)
    shutil.copy2(os.path.join(HERE, "config.py"), code)
    for f in os.listdir(os.path.join(HERE, "smcml")):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(HERE, "smcml", f), os.path.join(code, "smcml"))


def exp_folder(n, note):
    slug = regex.sub(r"[^a-z0-9]+", "_", note.lower()).strip("_")[:40] or "change"
    return os.path.join(EXP, f"{n:03d}_{slug}")


def find_folder(n):
    for f in sorted(os.listdir(EXP)):
        if f.startswith(f"{n:03d}_"):
            return os.path.join(EXP, f)
    sys.exit(f"No attempt number {n}. See: python iterate.py log")


def save_result(folder, trades, summ, rnd):
    os.makedirs(folder, exist_ok=True)
    for p, tr in trades.items():
        cols = [c for c in ("key", "time", "direction", "is_reentry", "target_R_used", "R_used", "hit", "filled")
                if c in tr.columns]
        (tr[cols] if len(tr) else pd.DataFrame(columns=cols)).to_csv(os.path.join(folder, f"{p}_trades.csv"),
                                                                     index=False)
    with open(os.path.join(folder, "summary.json"), "w") as fh:
        json.dump(dict(periods=summ, random=rnd), fh, indent=1)


def load_result(folder):
    with open(os.path.join(folder, "summary.json")) as fh:
        s = json.load(fh)
    tr = {p: pd.read_csv(os.path.join(folder, f"{p}_trades.csv")) for p in s["periods"]}
    return s, tr


def fmt_period(s):
    if not s["orders"]:
        return "no trades"
    hr = f"{100 * s['hit_rate']:.1f}% vs chance {100 * s['chance_rate']:.1f}%" if s["filled"] else "-"
    return (f"{s['orders']} orders, {s['targets_hit']} targets hit, {s['stopped']} stopped, "
            f"{s['total_R']:+.1f}R ({s['result_pct']:+.1f}%), worst drawdown {s['worst_dd_pct']:.1f}%, "
            f"hit rate {hr}")


# ----------------------------------------------------------------------------- commands
def cmd_start(args):
    if os.path.exists(EXP):
        if not args.fresh:
            sys.exit("experiments/ already exists. Keep going with 'check', or add --fresh to start over "
                     "(the old folder is kept as experiments_old_<date>).")
        shutil.move(EXP, EXP + f"_old_{dt.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(EXP)
    meta = dict(data=os.path.abspath(args.data) if args.data else None, sim=args.sim, seed=args.seed,
                bars=args.bars, cost=args.cost, created=f"{dt.datetime.now():%Y-%m-%d %H:%M}", kept=0, checks=0)
    write_meta(meta)
    apply_cost(meta)
    df = load_data(meta)
    per = periods(len(df))
    print(f"Data: {len(df):,} bars, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
    for p, (a, b) in per.items():
        print(f"  {p:<12} {df.index[a]:%Y-%m-%d} to {df.index[b - 1]:%Y-%m-%d}")
    print(f"  {'lockbox':<12} {df.index[per['validation'][1]]:%Y-%m-%d} onwards (not used here)")
    print("Running the tests ...", flush=True)
    ok, last, out = run_tests()
    print(f"  {last}")
    if not ok:
        print(out)
        sys.exit("The tests fail on the starting code. Fix that first.")
    print("Scoring the starting code ...", flush=True)
    lab, trades = run_history(df)
    summ = {p: summarize(tr) for p, tr in trades.items()}
    rnd = random_check(int(bar_length(df.index).total_seconds() // 60))
    folder = exp_folder(0, "baseline")
    save_result(folder, trades, summ, rnd)
    snapshot(folder)
    nl, nw = loss_charts(trades["development"], lab, df, os.path.join(folder, "losses"), args.charts)
    for p, s in summ.items():
        print(f"  {p:<12} {fmt_period(s)}")
    print(f"  random prices: {rnd['targets_hit']} of {rnd['trades']} hit {cfg.TARGET_R}R "
          f"({100 * (rnd['hit_rate'] or 0):.1f}% vs chance {100 * rnd['chance_rate']:.1f}%)")
    pd.DataFrame([dict(n=0, when=meta["created"], note="baseline", tests=last, verdict="START",
                       dev_R=summ["development"]["total_R"], val_R=summ["validation"]["total_R"],
                       random_p=rnd["p"])]).to_csv(LOG, index=False)
    print(f"\nSaved {nl} loss charts and {nw} winners (for comparison) from the DEVELOPMENT period in\n"
          f"  {os.path.relpath(os.path.join(folder, 'losses'), HERE)}/\n"
          f"Write why each one lost in loss_notes.csv there, look for a pattern, change the code, then run:\n"
          f"  python iterate.py check \"what you changed\"")


def cmd_check(args):
    meta = read_meta()
    apply_cost(meta)
    n = meta["checks"] + 1
    kept_folder = find_folder(meta["kept"])
    kept, kept_tr = load_result(kept_folder)
    print(f"Attempt {n}: {args.note}")
    print("Running the tests ...", flush=True)
    ok, last, out = run_tests()
    print(f"  {last}")
    folder = exp_folder(n, args.note)
    snapshot(folder)
    df = load_data(meta)
    summ, rnd, trades, lab = None, None, None, None
    if ok:
        print("Scoring the change ...", flush=True)
        lab, trades = run_history(df)
        summ = {p: summarize(tr) for p, tr in trades.items()}
        rnd = random_check(int(bar_length(df.index).total_seconds() // 60))
        save_result(folder, trades, summ, rnd)

    kp = kept["periods"]
    print(f"\n{'':<12} KEPT VERSION (attempt {meta['kept']})")
    for p in kp:
        print(f"  {p:<12} {fmt_period(kp[p])}")
    if ok:
        print(f"{'':<12} THIS CHANGE")
        for p in summ:
            print(f"  {p:<12} {fmt_period(summ[p])}")
        print()
        for p in summ:
            old, new = kept_tr[p], trades[p]
            old_keys, new_keys = set(old.key) if len(old) else set(), set(new.key) if len(new) else set()
            old_loss = set(old[old.R_used < -0.99].key) if len(old) else set()
            old_win = set(old[old.hit == 1].key) if len(old) else set()
            print(f"  {p:<12} losses removed {len(old_loss - new_keys)}, winners removed {len(old_win - new_keys)}, "
                  f"new trades {len(new_keys - old_keys)}")
        print(f"  random prices: {rnd['targets_hit']} of {rnd['trades']} hit {cfg.TARGET_R}R "
              f"({100 * (rnd['hit_rate'] or 0):.1f}% vs chance {100 * rnd['chance_rate']:.1f}%, p = {rnd['p']})")

    if not ok:
        verdict, why = "REVERT", "a test failed. The change may be using future prices, or broke something."
        print(out)
    elif not rnd["ok"]:
        verdict, why = "REVERT", ("it makes RANDOM prices look profitable. Nothing can beat chance on a random "
                                  "walk, so the change is using the future somewhere, or it's a bug.")
    elif summ["validation"]["orders"] == 0:
        verdict, why = "REVERT", "it takes no trades at all in the validation period."
    elif summ["development"]["total_R"] < kp["development"]["total_R"]:
        verdict, why = "REVERT", ("it made the development period worse, the very trades it was designed on. "
                                  "Any gain on validation is then most likely luck.")
    elif summ["validation"]["total_R"] > kp["validation"]["total_R"]:
        verdict, why = "KEEP", ("it helped on the development period AND made more R on the validation period, "
                                "which it wasn't designed on.")
    else:
        verdict, why = "REVERT", "it didn't make more R on the validation period, where it wasn't designed."
    print(f"\nVERDICT: {verdict}, because {why}")
    if ok and verdict == "KEEP":
        diff = summ["validation"]["total_R"] - kp["validation"]["total_R"]
        if diff < cfg.TARGET_R:
            print(f"  Careful: the gain ({diff:+.1f}R) is less than one winning trade. It could easily be luck.")
    if verdict == "KEEP":
        meta["kept"] = n
        nl, nw = loss_charts(trades["development"], lab, df, os.path.join(folder, "losses"), args.charts)
        print(f"  New loss charts: {os.path.relpath(os.path.join(folder, 'losses'), HERE)}/")
    else:
        print(f"  To put the kept code back:  python iterate.py restore {meta['kept']}")
    meta["checks"] = n
    write_meta(meta)
    row = dict(n=n, when=f"{dt.datetime.now():%Y-%m-%d %H:%M}", note=args.note, tests=last, verdict=verdict,
               dev_R=summ["development"]["total_R"] if ok else None,
               val_R=summ["validation"]["total_R"] if ok else None, random_p=rnd["p"] if ok else None)
    pd.concat([pd.read_csv(LOG), pd.DataFrame([row])]).to_csv(LOG, index=False)
    if n >= 10:
        print(f"\nThis is attempt {n} against the same validation period. Each attempt is another roll of the "
              f"dice, so some KEEPs are now luck. Get fresh data (newer history, or trades from the demo bot), "
              f"then 'python iterate.py start --fresh' on it.")


def cmd_log(args):
    read_meta()
    t = pd.read_csv(LOG)
    with pd.option_context("display.width", 200, "display.max_colwidth", 50):
        print(t.to_string(index=False))


def cmd_restore(args):
    meta = read_meta()
    src = os.path.join(find_folder(args.n), "code")
    backup = os.path.join(EXP, f"_code_before_restore_{dt.datetime.now():%Y%m%d_%H%M%S}")
    snapshot(backup)
    shutil.copy2(os.path.join(src, "config.py"), HERE)
    for f in os.listdir(os.path.join(src, "smcml")):
        shutil.copy2(os.path.join(src, "smcml", f), os.path.join(HERE, "smcml"))
    print(f"Code of attempt {args.n} is back (your current code was saved in {os.path.relpath(backup, HERE)}).")
    if args.n != meta["kept"]:
        print(f"Note: the kept version is attempt {meta['kept']}; checks are always compared against that one.")


def main():
    ap = argparse.ArgumentParser(description="Look at losses, change the code, score the change honestly.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="score the current code and chart its losses (once)")
    s.add_argument("--data", default=cfg.DATA_CSV, help="your MT5 export (CSV)")
    s.add_argument("--sim", default="random", choices=["random", "planted"], help="simulator, if no --data")
    s.add_argument("--seed", type=int, default=cfg.SIM_SEED)
    s.add_argument("--bars", type=int, default=cfg.SIM_BARS)
    s.add_argument("--cost", type=float, default=None, metavar="PRICE", help="cost per trade in price units")
    s.add_argument("--charts", type=int, default=30, help="how many loss charts to save")
    s.add_argument("--fresh", action="store_true", help="start over (keeps the old folder)")
    c = sub.add_parser("check", help="score your code change against the kept version")
    c.add_argument("note", help='what you changed, e.g. "skip setups in the Asian session"')
    c.add_argument("--charts", type=int, default=30)
    sub.add_parser("log", help="every attempt so far")
    r = sub.add_parser("restore", help="put the code of an earlier attempt back")
    r.add_argument("n", type=int)
    args = ap.parse_args()
    {"start": cmd_start, "check": cmd_check, "log": cmd_log, "restore": cmd_restore}[args.cmd](args)


if __name__ == "__main__":
    main()
