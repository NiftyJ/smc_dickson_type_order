"""
THE RUNNING VERSION: the backtest's decisions, made one closed bar at a time.

The Engine is the only place trading decisions are made. It is used in two ways:
  * live:   Engine + MT5Broker (live_mt5.py) + LiveSignals (runs scan() on the bars MT5 sends)
  * replay: Engine + PaperBroker (fills on historical bars exactly like labels.py)
            + ReplaySignals (scan() run once over the whole history)
Because both use the same Engine, a replay of your history must give the same trades as
the backtest. tests/test_live.py checks that it does.

What the Engine does at the close of every bar:
  1. record fills and closed trades reported by the broker (and their result in R)
  2. cancel limit orders older than MAX_BARS_WAIT_FILL bars, close trades held longer
     than MAX_HOLD_BARS bars
  3. new setups on this bar, then re-entries on this bar (one per stopped trade)
  4. the risk guard: daily loss limit, max open trades, target switching in drawdown,
     and lot size from the stop distance so a stop-out loses RISK_PER_TRADE_PCT
"""
import json
import math
import os
import types
from collections import defaultdict

import numpy as np
import pandas as pd

from .labels import label
from .reentry import find_reentries, reentry_round
from .risk import drawdown_target, JournalLocks, locks_on
from .setups import scan

ORDER_COLS = ["key", "status", "shot", "is_reentry", "parent_key", "direction", "placed_time", "entry", "stop",
              "tp", "target_R", "risk", "risk_money", "lots", "fill_time", "exit_time", "exit_reason",
              "profit", "R", "backtest_R", "be_moved", "note", "ticket"]


def cfg_copy(cfg, **changes):
    c = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    for k, v in changes.items():
        setattr(c, k, v)
    return c


def key_of(time, direction, shot=1):
    return f"{pd.Timestamp(time):%Y%m%d-%H%M}{'L' if int(direction) == 1 else 'S'}{f'-shot{shot}' if shot > 1 else ''}"


def _plain(x):
    """Make numpy / pandas values JSON-friendly."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    return x


# ----------------------------------------------------------------------------- signals
class LiveSignals:
    """Runs the backtest's own scan() on the bars it is given and returns only the
    decisions made on the LAST closed bar."""

    def __init__(self, cfg):
        self.cfg = cfg

    def new_setups(self, df):
        cands = scan(df, self.cfg)
        if cands.empty:
            return []
        return cands[cands["time"] == df.index[-1]].to_dict("records")

    def reentry(self, df, parent, tr, stop_time):
        s = df.index.get_indexer([pd.Timestamp(stop_time)])[0]
        if s < 0:
            return None
        row = {"candidate_id": 0, "direction": int(parent["direction"]), "filled": 1}
        for t in sorted({self.cfg.TARGET_R, self.cfg.TARGET_R_IN_DRAWDOWN}):
            row[f"R_{t}"] = -1.0 if t == tr else np.nan
            row[f"t_exit_{t}"] = float(s) if t == tr else np.nan
        re = reentry_round(pd.DataFrame([row]), df, self.cfg, shot=int(parent.get("shot") or 1) + 1)
        if re.empty or int(re.iloc[0]["t_place"]) != len(df) - 1:
            return None
        return re.iloc[0].to_dict()


class ReplaySignals:
    """The same answers as LiveSignals, worked out once for a whole history. This is valid
    because the no-lookahead tests prove scan() and find_reentries() never use later bars."""

    def __init__(self, df, cfg):
        self.cfg = cfg
        self.index = df.index
        self.cands = scan(df, cfg)
        self.lab = label(self.cands, df, cfg) if len(self.cands) else self.cands
        self.re = find_reentries(self.lab, df, cfg) if (cfg.REENTRY and len(self.lab)) else pd.DataFrame()
        self.by_time = {t: g.to_dict("records") for t, g in self.cands.groupby("time")} if len(self.cands) else {}
        self.re_map = {(int(x["parent_id"]), int(x["parent_target"])): x for x in self.re.to_dict("records")}

    def new_setups(self, df):
        return self.by_time.get(df.index[-1], [])

    def reentry(self, df, parent, tr, stop_time):
        if parent.get("candidate_id") is None:
            return None
        x = self.re_map.get((int(parent["candidate_id"]), int(tr)))
        if x is None or self.index[int(x["t_place"])] != df.index[-1]:
            return None
        return x


# ----------------------------------------------------------------------------- paper broker
class PaperBroker:
    """Trades on historical bars with exactly the backtest's fill rules (labels.py):
    a limit fills when a later bar reaches it; stop and target in the same bar = stop;
    on the fill bar only the close counts towards the target; costs as in config."""

    def __init__(self, df, balance=10_000.0):
        self.df = df
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.i = -1
        self._balance = float(balance)
        self.orders = {}
        self.events = []
        self._next = 1

    # --- what the Engine calls
    def is_demo(self):
        return True

    def balance(self):
        return self._balance

    def closed_bars(self, n=None):
        s = 0 if n is None else max(0, self.i - n + 1)
        return self.df.iloc[s:self.i + 1]

    def place(self, direction, entry, stop, tp, risk_money, cost_price=0.0, comment=""):
        tk = self._next
        self._next += 1
        risk = abs(entry - stop)
        self.orders[tk] = dict(d=int(direction), entry=entry, stop=stop, tp=tp, risk=risk,
                               tr=round(abs(tp - entry) / risk, 6), risk_money=risk_money,
                               cost_R=cost_price / risk, placed=self.i, filled=None, moved=False)
        return dict(ticket=tk, lots=round(risk_money / risk, 6), price=entry), None

    def cancel(self, ticket):
        od = self.orders.get(ticket)
        if od is not None and od["filled"] is None:
            del self.orders[ticket]

    def close(self, ticket):
        od = self.orders.get(ticket)
        if od is not None and od["filled"] is not None:
            self._close(ticket, od, (od["d"] * self.c[self.i] - od["d"] * od["entry"]) / od["risk"], "time", self.c[self.i])

    def move_stop(self, ticket, price):
        od = self.orders.get(ticket)
        if od is not None and od["filled"] is not None:
            od["stop"], od["moved"] = price, True       # takes effect from the next bar

    def open_tickets(self):
        return set(self.orders)

    def poll(self):
        ev, self.events = self.events, []
        return ev

    # --- moving through history
    def advance(self, i):
        self.i = i
        for tk, od in list(self.orders.items()):
            self._bar(tk, od, i)

    def _bar(self, tk, od, i):
        d = od["d"]
        fh, fl, fc = (self.h[i], self.l[i], self.c[i]) if d == 1 else (-self.l[i], -self.h[i], -self.c[i])
        entry, stop, risk, tr = d * od["entry"], d * od["stop"], od["risk"], od["tr"]
        if od["filled"] is None:
            if i <= od["placed"] or fl > entry:
                return
            od["filled"] = i
            self.events.append(dict(kind="fill", ticket=tk, time=self.df.index[i], price=od["entry"]))
            if fl <= stop:
                return self._close(tk, od, -1.0, "sl")
            if (fc - entry) / risk >= tr:
                return self._close(tk, od, tr, "tp")
            return
        if fl <= stop:
            return self._close(tk, od, (stop - entry) / risk, "be" if od["moved"] else "sl")
        if (fh - entry) / risk >= tr:
            return self._close(tk, od, tr, "tp")

    def _close(self, tk, od, R, reason, price=None):
        if price is None:
            price = od["entry"] + od["d"] * R * od["risk"]
        R = R - od["cost_R"]
        profit = R * od["risk_money"]
        self._balance += profit
        self.events.append(dict(kind="close", ticket=tk, time=self.df.index[self.i], profit=profit, reason=reason,
                                price=price))
        del self.orders[tk]


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, broker, signals, cfg, out_dir, symbol="", charts=True, max_charts=None, verbose=True,
                 autosave=True):
        self.b, self.sig, self.cfg = broker, signals, cfg
        self.autosave, self.dirty = autosave, False
        self.out_dir, self.symbol, self.verbose = out_dir, symbol, verbose
        self.charts, self.max_charts, self.n_charts = charts, max_charts, 0
        os.makedirs(out_dir, exist_ok=True)
        self.state_path = os.path.join(out_dir, "state.json")
        self.trades_path = os.path.join(out_dir, "trades.csv")
        self.target_for = drawdown_target(cfg) if getattr(cfg, "DYNAMIC_TARGET", False) else None
        if os.path.exists(self.state_path):
            with open(self.state_path) as fh:
                self.st = json.load(fh)
        else:
            self.st = dict(last_bar=None, equity_pct=0.0, peak_pct=0.0, realized={}, orders={},
                           stopouts=[], log=[])
        self.locks = JournalLocks(cfg, self.st.get("locks"))
        self.st["locks"] = self.locks.s

    # --- helpers
    def say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _pos(df, t):
        """Position of bar time t in df, or -1 (searchsorted: fast on long histories)."""
        t = pd.Timestamp(t)
        k = int(df.index.searchsorted(t))
        return k if k < len(df) and df.index[k] == t else -1

    def _bars_since(self, df, t):
        k = self._pos(df, t)
        return len(df) - 1 - k if k >= 0 else 10 ** 9

    def _bar_of(self, df, t):
        return int(df.index.searchsorted(pd.Timestamp(t), side="right") - 1)

    def _open(self):
        return {tk: o for tk, o in self.st["orders"].items() if o["status"] in ("pending", "filled")}

    def _save(self):
        with open(self.state_path, "w") as fh:
            json.dump(self.st, fh, indent=1, default=str)
        rows = [{k: o.get(k) for k in ORDER_COLS} for o in self.st["orders"].values()] + self.st["log"]
        pd.DataFrame(rows, columns=ORDER_COLS).sort_values("placed_time").to_csv(self.trades_path, index=False)

    # --- the bar
    def on_bar(self, df):
        now = df.index[-1]
        if self.st["last_bar"] is not None and pd.Timestamp(self.st["last_bar"]) >= now:
            return
        self._sync(df)
        self._housekeeping(df)
        for row in sorted(self.sig.new_setups(df), key=lambda r: r["direction"]):
            self._try(df, row, reentry=False, parent=None)
        if getattr(self.cfg, "REENTRY", False):
            self._reentries(df)
        self.st["last_bar"] = str(now)
        if self.autosave and self.dirty:
            self._save()
            self.dirty = False

    def _sync(self, df):
        self._events(df)
        live = {str(t) for t in self.b.open_tickets()}
        for tk, o in self._open().items():
            if tk in live:
                o.pop("missing", None)
            elif o["status"] == "pending":            # missing twice in a row = really gone
                o["missing"] = o.get("missing", 0) + 1
                if o["missing"] >= 2:
                    o.update(status="cancelled", note="order disappeared at the broker")
                    self.dirty = True

    def _events(self, df):
        for ev in self.b.poll():
            o = self.st["orders"].get(str(ev["ticket"]))
            if o is None:
                continue
            if ev["kind"] == "fill" and o["status"] == "pending":
                o.update(status="filled", fill_time=str(df.index[max(self._bar_of(df, ev["time"]), 0)]))
                self.dirty = True
                self.say(f"{ev['time']}  filled   {o['key']} at {ev.get('price', o['entry'])}")
            elif ev["kind"] == "close" and o["status"] in ("pending", "filled"):
                self._closed(df, o, ev)

    def _closed(self, df, o, ev):
        R = float(ev["profit"]) / o["risk_money"]
        k = self._bar_of(df, ev["time"])
        exit_bar_time = df.index[max(k, 0)]
        o.update(status="closed", exit_time=str(exit_bar_time), exit_reason=ev["reason"],
                 profit=round(float(ev["profit"]), 2), R=round(R, 3))
        self.dirty = True
        if o.get("fill_time") is None:
            o["fill_time"] = str(exit_bar_time)
        day = str(exit_bar_time.normalize().date())
        self.st["realized"][day] = self.st["realized"].get(day, 0.0) + R
        self.st["equity_pct"] += R * self.cfg.RISK_PER_TRADE_PCT
        self.st["peak_pct"] = max(self.st["peak_pct"], self.st["equity_pct"])
        o["backtest_R"] = self._backtest_R(df, o)
        self.say(f"{exit_bar_time}  closed   {o['key']}  {ev['reason']}  {R:+.2f}R  ({o['profit']:+.2f})")
        stopped = R < -0.99 or (ev["reason"] == "sl" and not o.get("be_moved"))
        max_shots = int(getattr(self.cfg, "REENTRY_MAX_SHOTS", 2))
        if stopped and int(o.get("shot") or 1) < max_shots:
            self.st["stopouts"].append(dict(ticket=str(o["ticket"]), tr=o["target_R"], stop_time=str(exit_bar_time),
                                            done=False))
        if locks_on(self.cfg):
            self.locks.closed(exit_bar_time, R, o["placed_time"], df.index,
                              df["high"].to_numpy(float), df["low"].to_numpy(float))
        if self.charts and stopped and (self.max_charts is None or self.n_charts < self.max_charts):
            self._chart(df, o)

    def _backtest_R(self, df, o):
        """What the backtest says this same order did, to spot live/backtest differences."""
        t = self._pos(df, o["placed_time"])
        if t < 0:
            return None
        d = o["direction"]
        row = pd.DataFrame([dict(t_place=t, direction=d, entry_f=d * o["entry"], stop_f=d * o["stop"],
                                 risk=o["risk"], cost_price=o.get("cost_price", 0.0))])
        lab = label(row, df, cfg_copy(self.cfg, EXTRA_TARGETS=tuple(set(self.cfg.EXTRA_TARGETS) | {o["target_R"]})))
        v = lab.iloc[0].get(f"R_{int(o['target_R'])}")
        return None if v is None or not np.isfinite(v) else round(float(v), 3)

    def _housekeeping(self, df):
        for tk, o in self._open().items():
            if o["status"] == "pending" and self._bars_since(df, o["placed_time"]) >= self.cfg.MAX_BARS_WAIT_FILL:
                self.b.cancel(o["ticket"])
                o.update(status="expired", R=0.0, exit_time=str(df.index[-1]), note="not filled in time")
                self.dirty = True
                self.say(f"{df.index[-1]}  expired  {o['key']} (not filled in {self.cfg.MAX_BARS_WAIT_FILL} bars)")
            elif o["status"] == "filled" and self._bars_since(df, o["fill_time"]) >= self.cfg.MAX_HOLD_BARS - 1:
                self.b.close(o["ticket"])
            elif o["status"] == "filled" and not o.get("be_moved") and self._reached_break_even(df, o):
                self.b.move_stop(o["ticket"], o["entry"])
                o["be_moved"] = True
                self.dirty = True
                self.say(f"{df.index[-1]}  stop to break-even  {o['key']} (+{self.cfg.BREAKEVEN_AT_R}R reached)")
        self._events(df)                                 # a time exit closes immediately

    def _reached_break_even(self, df, o):
        """Has price gone BREAKEVEN_AT_R in our favour since the fill? (fill bar: close only,
        exactly like labels.py). The stop then moves from the next bar on."""
        be = getattr(self.cfg, "BREAKEVEN_AT_R", None)
        if be is None or be >= o["target_R"]:
            return False
        k = self._pos(df, o["fill_time"])
        if k < 0:
            return False
        d = o["direction"]
        ext = df["high"].to_numpy()[k + 1:] if d == 1 else df["low"].to_numpy()[k + 1:]
        best = d * df["close"].to_numpy()[k]
        if len(ext):
            best = max(best, float((d * ext).max()))
        return (best - d * o["entry"]) / o["risk"] >= be

    def _reentries(self, df):
        for s in self.st["stopouts"]:
            if s["done"]:
                continue
            if self._bars_since(df, s["stop_time"]) > self.cfg.REENTRY_WINDOW_BARS:
                s["done"] = True
                self.dirty = True
                continue
            parent = self.st["orders"][s["ticket"]]
            row = self.sig.reentry(df, parent, s["tr"], s["stop_time"])
            if row is not None:
                s["done"] = True
                self.dirty = True
                self._try(df, row, reentry=True, parent=parent)

    def _try(self, df, row, reentry, parent):
        cfg, now = self.cfg, df.index[-1]
        d = int(row["direction"])
        shot = int(parent.get("shot") or 1) + 1 if reentry else 1
        key = key_of(now, d, shot)
        base = dict(key=key, shot=shot, is_reentry=reentry, parent_key=parent["key"] if parent else None, direction=d,
                    placed_time=str(now), entry=_plain(row["entry"]), stop=_plain(row["stop"]),
                    risk=_plain(row["risk"]), cost_price=_plain(row.get("cost_price", 0.0)),
                    candidate_id=_plain(row.get("candidate_id")))
        day = str(now.normalize().date())
        if locks_on(cfg):
            why = self.locks.blocked(now, df.index, df["close"].to_numpy(float))
            if why:
                return self._skip(base, why)
        if self.st["realized"].get(day, 0.0) * cfg.RISK_PER_TRADE_PCT <= -cfg.MAX_DAILY_LOSS_PCT:
            return self._skip(base, "daily loss limit reached")
        if len(self._open()) >= cfg.MAX_OPEN_TRADES:
            return self._skip(base, "max open trades")
        if self.target_for is None:
            tr = cfg.TARGET_R
        else:
            dd = (self.st["peak_pct"] - self.st["equity_pct"]) / (100 + self.st["peak_pct"]) * 100
            tr = self.target_for(dd)
        entry, stop, risk = row["entry"], row["stop"], row["risk"]
        tp = entry + d * tr * risk
        risk_money = self.b.balance() * cfg.RISK_PER_TRADE_PCT / 100
        res, why = self.b.place(d, float(entry), float(stop), float(tp), risk_money,
                                cost_price=float(row.get("cost_price", 0.0)), comment=key)
        if res is None:
            return self._skip(base, why)
        o = dict(base, status="pending", ticket=res["ticket"], tp=float(tp), target_R=int(tr),
                 risk_money=round(risk_money, 2), lots=res["lots"])
        self.st["orders"][str(res["ticket"])] = o
        if locks_on(cfg):
            self.locks.placed(now)
        self.dirty = True
        side = "BUY" if d == 1 else "SELL"
        self.say(f"{now}  placed   {key}  {side} {res['lots']} lots  entry {entry:.5g}  stop {stop:.5g}  "
                 f"target {tp:.5g} ({tr}R)  risking {risk_money:.2f}")
        return True

    def _skip(self, base, why):
        self.st["log"].append(dict(base, status="skipped", note=why))
        self.dirty = True
        return False

    def _chart(self, df, o):
        from .plot_trades import plot_simple
        folder = os.path.join(self.out_dir, "charts")
        os.makedirs(folder, exist_ok=True)
        try:
            plot_simple(df, o, os.path.join(folder, f"loss_{o['key']}.png"))
            self.n_charts += 1
        except Exception as e:  # a chart must never stop the bot
            self.say(f"(chart failed: {e})")

    # --- summary
    def summary(self):
        t = pd.read_csv(self.trades_path) if os.path.exists(self.trades_path) else pd.DataFrame(columns=ORDER_COLS)
        closed = t[t.status == "closed"]
        return dict(orders=int(t.status.isin(["pending", "filled", "closed", "expired"]).sum()),
                    filled=int(len(closed)), targets_hit=int((closed.exit_reason == "tp").sum()),
                    stopped=int((closed.R < -0.99).sum()), total_R=round(float(closed.R.sum()), 2),
                    result_pct=round(float(closed.R.sum()) * self.cfg.RISK_PER_TRADE_PCT, 2),
                    skipped=int((t.status == "skipped").sum()))


def replay(df, cfg, out_dir, balance=10_000.0, max_charts=30, verbose=False):
    """Run the Engine over a history with the PaperBroker. Returns (engine, signals)."""
    if os.path.exists(os.path.join(out_dir, "state.json")):
        os.remove(os.path.join(out_dir, "state.json"))
    sig = ReplaySignals(df, cfg)
    broker = PaperBroker(df, balance)
    eng = Engine(broker, sig, cfg, out_dir, charts=max_charts > 0, max_charts=max_charts, verbose=verbose,
                 autosave=False)
    start = int(sig.cands["t_place"].min()) if len(sig.cands) else len(df)
    for i in range(start, len(df)):
        broker.advance(i)
        eng.on_bar(broker.closed_bars(None))
    eng._save()
    return eng, sig
