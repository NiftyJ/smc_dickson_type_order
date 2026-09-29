"""
THE RISK GUARD: the part that protects the account from you (and from the model).

Rules, all set once in config.py:
  * every trade risks RISK_PER_TRADE_PCT of the starting balance (1R)
  * no more than MAX_OPEN_TRADES orders/trades at the same time
  * after losing MAX_DAILY_LOSS_PCT in a day, no new orders until the next day
  * the locks from your journal (class JournalLocks below):
      - range lock: after COOLDOWN_LOSING_DAYS losing days within COOLDOWN_WINDOW_DAYS
        (or COOLDOWN_AFTER_STOPS stop-outs in a row), no new orders for COOLDOWN_DAYS,
        or until a candle closes outside the range price was stuck in
      - loss block: LOSS_BLOCK_COUNT stop-outs within 24 hours -> no orders for
        LOSS_BLOCK_HOURS
      - at most MAX_ORDERS_PER_24H orders in any 24 hours

In the live bot these must be enforced in code with no override button.
Emotional decisions blow accounts; a limit you can switch off is not a limit.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def drawdown_target(cfg):
    """The target rule: TARGET_R normally, TARGET_R_IN_DRAWDOWN while the account is
    more than DD_SWITCH_PCT below its highest point."""
    return lambda dd_pct: cfg.TARGET_R_IN_DRAWDOWN if dd_pct > cfg.DD_SWITCH_PCT else cfg.TARGET_R


def locks_on(cfg):
    return bool(getattr(cfg, "COOLDOWN", False) or getattr(cfg, "LOSS_BLOCK_COUNT", 0)
                or getattr(cfg, "MAX_ORDERS_PER_24H", 0))


class JournalLocks:
    """The journal's locks. The backtest (simulate_account) and the running version
    (live.Engine) both use this one class, so they can't drift apart.

    Everything is in bar times: a trade's exit time is the open time of the bar it
    closed in, an order's time is the bar it was placed at. The state is a plain dict
    so the running version can save it."""

    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.s = state if state is not None else dict(streak=[], pause=None, last_trigger_day=None, day_R={},
                                                      losses=[], block_until=None, orders=[])

    # --- things that happened
    def closed(self, exit_time, R, placed_time, idx, H, L):
        """A filled trade closed with result R (in R). idx/H/L: bar times and highs/lows
        (only bars up to exit_time are used)."""
        cfg, s = self.cfg, self.s
        exit_time = pd.Timestamp(exit_time)
        day = exit_time.normalize()
        key = str(day.date())
        s["day_R"][key] = s["day_R"].get(key, 0.0) + R
        stopped = R < -0.99
        n_block = int(getattr(cfg, "LOSS_BLOCK_COUNT", 0) or 0)
        if n_block and stopped:
            window = pd.Timedelta(hours=float(getattr(cfg, "LOSS_BLOCK_WINDOW_HOURS", 24)))
            s["losses"] = [t for t in s["losses"] if pd.Timestamp(t) > exit_time - window] + [str(exit_time)]
            if len(s["losses"]) >= n_block:
                s["block_until"] = str(exit_time + pd.Timedelta(hours=float(getattr(cfg, "LOSS_BLOCK_HOURS", 24))))
                s["losses"] = []
        if not getattr(cfg, "COOLDOWN", False):
            return
        if getattr(cfg, "COOLDOWN_TRIGGER", "stops_in_a_row") == "losing_days":
            first_day = day - (int(cfg.COOLDOWN_WINDOW_DAYS) - 1) * DAY
            if s["last_trigger_day"] is not None:
                first_day = max(first_day, pd.Timestamp(s["last_trigger_day"]) + DAY)
            n_days = int((day - first_day) / DAY) + 1
            losing = [d for d in (first_day + i * DAY for i in range(max(n_days, 0)))
                      if s["day_R"].get(str(d.date()), 0.0) < 0]
            if len(losing) >= int(cfg.COOLDOWN_LOSING_DAYS):
                self._pause(exit_time, losing[0], idx, H, L)
                s["last_trigger_day"] = key
        else:
            if not stopped:
                s["streak"] = []
                return
            s["streak"].append(str(pd.Timestamp(placed_time)))
            if len(s["streak"]) >= int(cfg.COOLDOWN_AFTER_STOPS):
                self._pause(exit_time, pd.Timestamp(s["streak"][0]), idx, H, L)
                s["streak"] = []

    def _pause(self, exit_time, start_time, idx, H, L):
        a = int(idx.searchsorted(start_time))
        b = int(idx.searchsorted(exit_time, side="right"))
        resume = exit_time.normalize() + int(self.cfg.COOLDOWN_DAYS) * DAY
        self.s["pause"] = dict(resume=str(resume), hi=float(np.max(H[a:b])), lo=float(np.min(L[a:b])),
                               since=str(exit_time))

    def placed(self, now):
        now = pd.Timestamp(now)
        self.s["orders"] = [t for t in self.s["orders"] if pd.Timestamp(t) > now - DAY] + [str(now)]

    # --- the check before every new order
    def blocked(self, now, idx, C):
        """None, or the reason no order may be placed at bar time `now`."""
        cfg, s = self.cfg, self.s
        now = pd.Timestamp(now)
        p = s["pause"]
        if p:
            over = now.normalize() >= pd.Timestamp(p["resume"])
            if not over and getattr(cfg, "COOLDOWN_ENDS_ON_BREAKOUT", True):
                k = int(idx.searchsorted(pd.Timestamp(p["since"]), side="right"))
                e = int(idx.searchsorted(now, side="right"))
                later = C[k:e]
                over = bool(((later > p["hi"]) | (later < p["lo"])).any())
            if over:
                s["pause"] = None
            else:
                return "range lock"
        if s["block_until"] is not None:
            if now < pd.Timestamp(s["block_until"]):
                return "loss block"
            s["block_until"] = None
        cap = int(getattr(cfg, "MAX_ORDERS_PER_24H", 0) or 0)
        if cap and sum(pd.Timestamp(t) > now - DAY for t in s["orders"]) >= cap:
            return "max orders per 24h"
        return None


def simulate_account(selected, bar_day, cfg, target_for=None, reentries=None, prices=None):
    """Walk through the chosen setups in time order and apply the guard.

    selected:   rows with t_place, t_exit, outcome_R (unfilled orders have 0R),
                plus R_<target> / t_exit_<target> columns if target_for is used
    bar_day:    the calendar day of every bar (to apply the daily loss limit)
    target_for: optional function(drawdown %) -> target in R, decided when each
                order is placed, using only trades that have already closed
    reentries:  optional table from reentry.find_reentries(). When a trade the guard
                took is stopped out, its next shot (if one formed) is queued, up to
                REENTRY_MAX_SHOTS trades per setup.
    prices:     the price DataFrame. Needed for the journal locks (times, and the range
                price was stuck in).
    Returns (trades actually taken, count of setups the guard blocked, by reason).
    """
    import heapq

    re_map = {}
    if reentries is not None and len(reentries):
        re_map = {(int(x.parent_id), int(x.parent_target)): x for x in reentries.itertuples(index=False)}
    queue = [(int(r.t_place), i, r, False) for i, r in
             enumerate(selected.sort_values("t_place").itertuples(index=False))]
    heapq.heapify(queue)
    counter = len(queue)

    max_shots = int(getattr(cfg, "REENTRY_MAX_SHOTS", 2))
    locks = None
    if locks_on(cfg):
        if prices is None:
            raise ValueError("The journal locks are switched on: pass prices=<the price DataFrame>.")
        locks = JournalLocks(cfg)
        idx = prices.index
        H, L, CL = (prices[k].to_numpy(dtype=float) for k in ("high", "low", "close"))

    open_trades = []                         # (exit bar, R, day of exit, filled, placement bar)
    realized = defaultdict(float)            # R realised per day
    equity = peak = 0.0                      # closed-trade equity, % of starting balance
    taken = []
    blocked = {"daily_loss": 0, "too_many_open": 0, "range lock": 0, "loss block": 0, "max orders per 24h": 0}
    while queue:
        t_place, _, r, is_re = heapq.heappop(queue)
        still = []
        for te, R, d, filled, tp0 in sorted(open_trades, key=lambda x: (x[0], x[4])):
            if te <= t_place:
                realized[d] += R
                equity += R * cfg.RISK_PER_TRADE_PCT
                peak = max(peak, equity)
                if locks is not None and filled:
                    locks.closed(idx[te], R, idx[tp0], idx, H, L)
            else:
                still.append((te, R, d, filled, tp0))
        open_trades = still
        today = bar_day[t_place]
        if locks is not None:
            why = locks.blocked(idx[t_place], idx, CL)
            if why:
                blocked[why] += 1
                continue
        if realized[today] * cfg.RISK_PER_TRADE_PCT <= -cfg.MAX_DAILY_LOSS_PCT:
            blocked["daily_loss"] += 1
            continue
        if len(open_trades) >= cfg.MAX_OPEN_TRADES:
            blocked["too_many_open"] += 1
            continue
        if target_for is None:
            tr, R, te = cfg.TARGET_R, r.outcome_R, r.t_exit
        else:
            dd_pct = (peak - equity) / (100 + peak) * 100      # % below the highest balance
            tr = target_for(dd_pct)
            R, te = getattr(r, f"R_{tr}"), getattr(r, f"t_exit_{tr}")
        if not (np.isfinite(R) and np.isfinite(te)):
            continue
        te = int(te)
        open_trades.append((te, R, bar_day[te], int(getattr(r, "filled", 1)) == 1, t_place))
        if locks is not None:
            locks.placed(idx[t_place])
        rec = r._asdict()
        shot = int(getattr(r, "shot", 1)) if is_re else 1
        rec.update(is_reentry=is_re, shot=shot, target_R_used=tr, R_used=R, t_exit_used=te)
        taken.append(rec)
        stopped = R < -0.99
        if stopped and shot < max_shots and (int(r.candidate_id), tr) in re_map:
            x = re_map[(int(r.candidate_id), tr)]
            heapq.heappush(queue, (int(x.t_place), counter, x, True))
            counter += 1
    trades = pd.DataFrame(taken)
    if not trades.empty:
        trades["pct"] = trades["R_used"] * cfg.RISK_PER_TRADE_PCT
        trades["equity_pct"] = trades["pct"].cumsum()
    return trades, blocked


def losing_streaks(win_rate, n_trades=1000, sims=5000, seed=0):
    """How long the worst losing run usually is, for a given win rate.

    Know this number BEFORE you trade. At a 5% win rate a run of 80+ losses in
    1,000 trades is normal, not a sign the system broke.
    """
    rng = np.random.default_rng(seed)
    worst = np.empty(sims, dtype=int)
    for i in range(sims):
        win_at = np.flatnonzero(rng.random(n_trades) < win_rate)
        worst[i] = (np.diff(np.r_[-1, win_at, n_trades]) - 1).max()
    return {"typical": int(np.median(worst)), "bad_1_in_10": int(np.percentile(worst, 90))}
