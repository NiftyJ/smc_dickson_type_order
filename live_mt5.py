"""
THE RUNNING VERSION on MetaTrader 5, or a paper replay of your history.

Live (on the Windows machine / VPS where MT5 is installed and logged in):
    pip install MetaTrader5
    python live_mt5.py --symbol "Volatility 75 Index" --timeframe M15

Paper replay (no MT5 needed; same decisions, fills like the backtest):
    python live_mt5.py --replay "data/V75_M15.csv"

What it does at every closed bar: exactly what the backtest does (smcml/live.py).
Everything is written to live/<symbol>_<timeframe>/:
    trades.csv     every order: placed, filled, closed, expired or skipped (and why),
                   the result in R, and "backtest_R" = what the backtest says the same
                   order did. If R and backtest_R keep disagreeing, live differs from the
                   backtest (spread, slippage, broker rules) and that has to be fixed first.
    charts/        a chart of every losing trade, for the loss review
    state.json     what the bot remembers if it is restarted

Safety, not optional:
  * DEMO ONLY unless you add --allow-real
  * risk per trade, daily loss limit and max open trades come from config.py; there is
    no command-line switch to change them
  * lot size comes from the stop distance so a stop-out loses RISK_PER_TRADE_PCT; if even
    the broker's smallest lot would lose more than that, the trade is skipped
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

import config as cfg
from smcml.live import Engine, LiveSignals, replay

MAGIC = 20_261_000


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        sys.exit("The MetaTrader5 package isn't installed. On the Windows machine with MT5: pip install MetaTrader5")
    return mt5


class MT5Broker:
    """Talks to the MT5 terminal. Only this class touches MT5; the Engine makes the decisions."""

    def __init__(self, symbol, timeframe, magic=MAGIC, login=None, password=None, server=None):
        mt5 = self.mt5 = _mt5()
        kw = {k: v for k, v in dict(login=login, password=password, server=server).items() if v}
        if not mt5.initialize(**kw):
            sys.exit(f"Couldn't connect to MT5: {mt5.last_error()}. Is the terminal open and logged in?")
        if mt5.symbol_info(symbol) is None or not mt5.symbol_select(symbol, True):
            sys.exit(f"Symbol '{symbol}' not found. Use the exact name from MT5's Market Watch.")
        self.symbol, self.magic = symbol, magic
        self.tf = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}")
        self.since = datetime.now() - timedelta(days=7)
        self.seen = set()
        self.commission = {}

    # --- account and prices
    def is_demo(self):
        return self.mt5.account_info().trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO

    def balance(self):
        return float(self.mt5.account_info().balance)

    def closed_bars(self, n):
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.tf, 1, n)      # start at 1 = closed bars only
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no bars from MT5: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df.index = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"), name="time")
        return df[["open", "high", "low", "close"]].astype(float)

    # --- orders
    def _filling(self, info):
        m = self.mt5
        flags = int(getattr(info, "filling_mode", 0))
        return m.ORDER_FILLING_FOK if flags & 1 else m.ORDER_FILLING_IOC if flags & 2 else m.ORDER_FILLING_RETURN

    def place(self, direction, entry, stop, tp, risk_money, cost_price=0.0, comment=""):
        m, sym = self.mt5, self.symbol
        info, tick = m.symbol_info(sym), m.symbol_info_tick(sym)
        rnd = lambda x: round(x, info.digits)
        entry, stop, tp = rnd(entry), rnd(stop), rnd(tp)
        risk = abs(entry - stop)
        loss_1lot = m.order_calc_profit(m.ORDER_TYPE_BUY if direction == 1 else m.ORDER_TYPE_SELL, sym, 1.0, entry, stop)
        if loss_1lot is None or loss_1lot >= 0:
            return None, f"couldn't work out the lot size ({m.last_error()})"
        loss_1lot = -loss_1lot
        step = info.volume_step
        dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        lots = math.floor(risk_money / loss_1lot / step + 1e-9) * step
        if lots < info.volume_min:
            return None, (f"smallest lot {info.volume_min} would lose {info.volume_min * loss_1lot:.2f} at the stop, "
                          f"more than the {risk_money:.2f} allowed")
        lots = round(min(lots, info.volume_max), dec)

        price_now = tick.ask if direction == 1 else tick.bid
        min_dist = info.trade_stops_level * info.point
        at_or_past = (price_now <= entry) if direction == 1 else (price_now >= entry)
        too_close = abs(price_now - entry) < max(min_dist, info.point)
        if at_or_past or too_close:
            if abs(price_now - entry) > 0.05 * risk and not at_or_past:
                return None, "limit too close to price for the broker, and a market order would cost > 0.05R"
            req = dict(action=m.TRADE_ACTION_DEAL, type=m.ORDER_TYPE_BUY if direction == 1 else m.ORDER_TYPE_SELL,
                       price=price_now)
        else:
            req = dict(action=m.TRADE_ACTION_PENDING,
                       type=m.ORDER_TYPE_BUY_LIMIT if direction == 1 else m.ORDER_TYPE_SELL_LIMIT, price=entry)
        if abs(req["price"] - stop) < min_dist or abs(tp - req["price"]) < min_dist:
            return None, "stop or target closer than the broker allows"
        req.update(symbol=sym, volume=lots, sl=stop, tp=tp, deviation=50, magic=self.magic,
                   comment=comment[:31], type_time=m.ORDER_TIME_GTC, type_filling=self._filling(info))
        res = m.order_send(req)
        ok = (m.TRADE_RETCODE_DONE, m.TRADE_RETCODE_PLACED, getattr(m, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        if res is None or res.retcode not in ok:
            why = f"{res.retcode} {res.comment}" if res is not None else str(m.last_error())
            return None, f"MT5 rejected the order: {why}"
        return dict(ticket=int(res.order), lots=lots, price=req["price"]), None

    def cancel(self, ticket):
        self.mt5.order_send(dict(action=self.mt5.TRADE_ACTION_REMOVE, order=int(ticket)))

    def close(self, ticket):
        m = self.mt5
        for p in m.positions_get(symbol=self.symbol) or ():
            if p.magic == self.magic and int(p.identifier) == int(ticket):
                tick = m.symbol_info_tick(self.symbol)
                buy = p.type == m.POSITION_TYPE_BUY
                m.order_send(dict(action=m.TRADE_ACTION_DEAL, symbol=self.symbol, volume=p.volume, position=p.ticket,
                                  type=m.ORDER_TYPE_SELL if buy else m.ORDER_TYPE_BUY,
                                  price=tick.bid if buy else tick.ask, deviation=50, magic=self.magic,
                                  comment="time exit", type_time=m.ORDER_TIME_GTC,
                                  type_filling=self._filling(m.symbol_info(self.symbol))))

    def move_stop(self, ticket, price):
        """Stop to break-even. If price is already back at the entry, close the trade now."""
        m = self.mt5
        for p in m.positions_get(symbol=self.symbol) or ():
            if p.magic == self.magic and int(p.identifier) == int(ticket):
                price = round(price, m.symbol_info(self.symbol).digits)
                tick = m.symbol_info_tick(self.symbol)
                buy = p.type == m.POSITION_TYPE_BUY
                now = tick.bid if buy else tick.ask
                if (buy and now <= price) or (not buy and now >= price):
                    return self.close(ticket)
                res = m.order_send(dict(action=m.TRADE_ACTION_SLTP, symbol=self.symbol, position=p.ticket,
                                        sl=price, tp=p.tp, magic=self.magic))
                if res is None or res.retcode != m.TRADE_RETCODE_DONE:
                    print(f"could not move the stop of {ticket}: {res.retcode if res else m.last_error()}", flush=True)

    def open_tickets(self):
        m = self.mt5
        out = {int(o.ticket) for o in (m.orders_get(symbol=self.symbol) or ()) if o.magic == self.magic}
        out |= {int(p.identifier) for p in (m.positions_get(symbol=self.symbol) or ()) if p.magic == self.magic}
        return out

    def poll(self):
        """Fills and closes since the last call, from MT5's deal history."""
        m = self.mt5
        deals = m.history_deals_get(self.since, datetime.now() + timedelta(days=2)) or ()
        out = []
        for dl in sorted(deals, key=lambda d: d.time_msc):
            if dl.ticket in self.seen or dl.magic != self.magic or dl.symbol != self.symbol:
                continue
            self.seen.add(dl.ticket)
            t = pd.Timestamp(dl.time, unit="s")
            costs = float(dl.commission) + float(dl.swap) + float(getattr(dl, "fee", 0.0))
            if dl.entry == m.DEAL_ENTRY_IN:
                self.commission[dl.position_id] = self.commission.get(dl.position_id, 0.0) + costs
                out.append(dict(kind="fill", ticket=int(dl.position_id), time=t, price=dl.price))
            elif dl.entry in (m.DEAL_ENTRY_OUT, getattr(m, "DEAL_ENTRY_OUT_BY", 3)):
                profit = float(dl.profit) + costs + self.commission.pop(dl.position_id, 0.0)
                reason = {m.DEAL_REASON_SL: "sl", m.DEAL_REASON_TP: "tp"}.get(dl.reason, "other")
                out.append(dict(kind="close", ticket=int(dl.position_id), time=t, price=dl.price,
                                profit=profit, reason=reason))
        return out


def run_live(args):
    broker = MT5Broker(args.symbol, args.timeframe, args.magic, args.login, args.password, args.server)
    if not broker.is_demo() and not args.allow_real:
        sys.exit("This MT5 account is a REAL account. The bot only trades demo accounts unless you add "
                 "--allow-real. Run it on demo first, for 100+ trades.")
    safe = "".join(ch if ch.isalnum() else "_" for ch in args.symbol)
    out = os.path.join("live", f"{safe}_{args.timeframe.upper()}")
    eng = Engine(broker, LiveSignals(cfg), cfg, out, symbol=args.symbol)
    print(f"Running {args.symbol} {args.timeframe} on a {'DEMO' if broker.is_demo() else 'REAL'} account. "
          f"Risk {cfg.RISK_PER_TRADE_PCT}% per trade, max {cfg.MAX_OPEN_TRADES} open, "
          f"daily loss limit {cfg.MAX_DAILY_LOSS_PCT}%. Log: {out}/trades.csv   (Ctrl+C to stop)", flush=True)
    last = None
    while True:
        try:
            df = broker.closed_bars(args.bars)
            if df.index[-1] != last:
                eng.on_bar(df)
                last = df.index[-1]
        except KeyboardInterrupt:
            raise
        except Exception as e:                    # never die silently; log and keep going
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  error: {e!r} (retrying)", flush=True)
            time.sleep(10)
            if not broker.mt5.initialize():
                print("MT5 reconnect failed; will retry", flush=True)
        time.sleep(args.poll)


def run_replay(args):
    from smcml.data import load_mt5_csv
    from smcml.risk import simulate_account, drawdown_target

    if args.cost is not None:
        cfg.COST_MODE, cfg.COST_PRICE = "price", args.cost
    df = load_mt5_csv(args.replay)
    name = os.path.splitext(os.path.basename(args.replay))[0]
    out = os.path.join("live", f"replay_{name}")
    print(f"Paper replay of {len(df):,} bars from {args.replay} ...", flush=True)
    eng, sig = replay(df, cfg, out, balance=args.balance, max_charts=args.charts, verbose=args.verbose)
    s = eng.summary()
    print(f"\nPaper replay: {s['orders']} orders, {s['filled']} filled and closed, {s['targets_hit']} targets hit, "
          f"{s['stopped']} stopped, total {s['total_R']:+.1f}R = {s['result_pct']:+.1f}% "
          f"({s['skipped']} setups skipped by the risk guard or broker rules)")
    done = sig.lab[sig.lab.complete] if len(sig.lab) else sig.lab
    if len(done):
        bt, _ = simulate_account(done, df.index.normalize().to_numpy(), cfg,
                                 target_for=drawdown_target(cfg) if cfg.DYNAMIC_TARGET else None,
                                 reentries=sig.re if cfg.REENTRY else None, prices=df)
        # compare up to the first setup the backtest can't grade yet (end of the data)
        pending = sig.lab.loc[~sig.lab.complete, "time"]
        cut = pending.min() if len(pending) else df.index[-1] + pd.Timedelta(days=1)
        t = pd.read_csv(os.path.join(out, "trades.csv"))
        t = t[(t.status != "skipped") & (pd.to_datetime(t.placed_time) < cut)]
        b = bt[df.index[bt.t_place.to_numpy()] < cut]
        a_rows = list(zip(pd.to_datetime(t.placed_time), t.direction, t.is_reentry, t.R.fillna(0)))
        b_rows = list(zip(df.index[b.t_place.to_numpy()], b.direction, b.is_reentry, b.R_used))
        same = sum(x[:3] == y[:3] and abs(x[3] - y[3]) < 0.01 for x, y in zip(a_rows, b_rows))
        print(f"Check against the backtest: {same} of {len(b_rows)} trades identical "
              f"(up to {cut:%Y-%m-%d}; after that the backtest can't grade trades yet).")
        if same != len(b_rows) or len(a_rows) != len(b_rows):
            print("WARNING: the replay and the backtest disagree. Don't trade this until that's fixed.")
    print(f"Log: {out}/trades.csv   Loss charts: {out}/charts/ (first {args.charts})")


def main():
    ap = argparse.ArgumentParser(description="Run the SMC bot on MT5 (demo) or replay it on history.")
    ap.add_argument("--symbol", help='exact MT5 symbol name, e.g. "Volatility 75 Index"')
    ap.add_argument("--timeframe", default="M15", help="M1, M5, M15, M30, H1 (the entry chart)")
    ap.add_argument("--bars", type=int, default=30_000, help="closed bars of history the bot looks at (must cover 100+ days for the D1 bias)")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between checks for a new bar")
    ap.add_argument("--magic", type=int, default=MAGIC, help="tag on the bot's orders")
    ap.add_argument("--login", type=int, default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--server", default=None)
    ap.add_argument("--allow-real", action="store_true", help="allow a REAL account (default: demo only)")
    ap.add_argument("--replay", metavar="CSV", help="paper-trade this MT5 export instead of connecting to MT5")
    ap.add_argument("--balance", type=float, default=10_000.0, help="starting balance for --replay")
    ap.add_argument("--cost", type=float, default=None, metavar="PRICE", help="cost per trade for --replay")
    ap.add_argument("--charts", type=int, default=30, help="loss charts to save in --replay")
    ap.add_argument("--verbose", action="store_true", help="print every order in --replay")
    args = ap.parse_args()
    if args.replay:
        run_replay(args)
    elif args.symbol:
        run_live(args)
    else:
        ap.error("give --symbol for live trading, or --replay CSV for a paper replay")


if __name__ == "__main__":
    main()
