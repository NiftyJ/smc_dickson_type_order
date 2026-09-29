"""
A pretend MetaTrader5 module, so live_mt5.MT5Broker can be tested without a terminal.

It replays a price history bar by bar, fills limit orders when a bar reaches them,
closes positions at stop (checked first) or target, and keeps a deal history, using the
names of the real MetaTrader5 Python package. It is only as faithful as its author's
reading of the MT5 docs, so always run the real bot on a demo account first.
"""
import types

import numpy as np
import pandas as pd

ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30, TIMEFRAME_H1 = 1, 5, 15, 30, 16385
ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT = 0, 1, 2, 3
POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
TRADE_ACTION_DEAL, TRADE_ACTION_PENDING, TRADE_ACTION_SLTP, TRADE_ACTION_REMOVE = 1, 5, 6, 8
ORDER_TIME_GTC = 0
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE, TRADE_RETCODE_INVALID_STOPS = 10008, 10009, 10016
DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_OUT_BY = 0, 1, 3
DEAL_REASON_EXPERT, DEAL_REASON_SL, DEAL_REASON_TP = 3, 4, 5

NS = types.SimpleNamespace


class Market:
    def __init__(self, df, balance=10_000.0, demo=True, contract=1.0, volume_min=0.001, volume_step=0.001,
                 digits=2):
        self.df = df
        self.t = np.asarray((df.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1), dtype="int64")
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.i = 0
        self.balance = balance
        self.demo = demo
        self.info = NS(digits=digits, volume_step=volume_step, volume_min=volume_min, volume_max=1000.0,
                       trade_stops_level=0, point=10 ** -digits, filling_mode=1, trade_contract_size=contract)
        self.contract = contract
        self.pending, self.positions, self.deals = {}, {}, []
        self.next_ticket = 1000

    def _tk(self):
        self.next_ticket += 1
        return self.next_ticket

    def _deal(self, t, entry, pos, price, profit=0.0, reason=DEAL_REASON_EXPERT, magic=0):
        self.deals.append(NS(ticket=self._tk(), time=int(t), time_msc=int(t) * 1000 + len(self.deals), magic=magic,
                             symbol="TEST", entry=entry, position_id=pos, price=price, profit=profit,
                             commission=0.0, swap=0.0, fee=0.0, reason=reason))

    def _open(self, ticket, p, t):
        self.positions[ticket] = dict(p, open_price=p["price"])
        self._deal(t, DEAL_ENTRY_IN, ticket, p["price"], magic=p["magic"])

    def _close(self, ticket, price, reason, t):
        p = self.positions.pop(ticket)
        d = 1 if p["type"] in (ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT) else -1
        profit = d * (price - p["open_price"]) * p["volume"] * self.contract
        self.balance += profit
        self._deal(t, DEAL_ENTRY_OUT, ticket, price, profit, reason, p["magic"])

    def advance(self, i):
        """Bar i happens: fills (limit orders from earlier bars) then stops / targets."""
        self.i = i
        t, h, l, c = self.t[i], self.h[i], self.l[i], self.c[i]
        filled_now = set()
        for tk, p in list(self.pending.items()):
            buy = p["type"] == ORDER_TYPE_BUY_LIMIT
            if p["bar"] < i and ((buy and l <= p["price"]) or (not buy and h >= p["price"])):
                del self.pending[tk]
                self._open(tk, p, t)
                filled_now.add(tk)
        for tk, p in list(self.positions.items()):
            buy = p["type"] in (ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT)
            hit_sl = (l <= p["sl"]) if buy else (h >= p["sl"])
            reach = c if tk in filled_now else (h if buy else l)          # fill bar: only the close counts
            hit_tp = (reach >= p["tp"]) if buy else (reach <= p["tp"])
            if hit_sl:
                self._close(tk, p["sl"], DEAL_REASON_SL, t)
            elif hit_tp:
                self._close(tk, p["tp"], DEAL_REASON_TP, t)


def make_module(market):
    m = types.ModuleType("MetaTrader5")
    for k, v in globals().items():
        if k.isupper():
            setattr(m, k, v)
    mk = market
    m.initialize = lambda **kw: True
    m.shutdown = lambda: None
    m.last_error = lambda: (1, "Success")
    m.account_info = lambda: NS(trade_mode=ACCOUNT_TRADE_MODE_DEMO if mk.demo else ACCOUNT_TRADE_MODE_REAL,
                                balance=mk.balance)
    m.symbol_info = lambda s: mk.info
    m.symbol_select = lambda s, on=True: True
    m.symbol_info_tick = lambda s: NS(bid=mk.c[mk.i], ask=mk.c[mk.i])

    def copy_rates_from_pos(sym, tf, start, n):
        end = mk.i + 1 - (start - 1)                  # start=1: bars up to the last CLOSED one
        s = max(0, end - n)
        arr = np.zeros(end - s, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")])
        arr["time"], arr["open"], arr["high"], arr["low"], arr["close"] = (
            mk.t[s:end], mk.o[s:end], mk.h[s:end], mk.l[s:end], mk.c[s:end])
        return arr
    m.copy_rates_from_pos = copy_rates_from_pos

    def order_calc_profit(typ, sym, vol, p_open, p_close):
        d = 1 if typ == ORDER_TYPE_BUY else -1
        return d * (p_close - p_open) * vol * mk.contract
    m.order_calc_profit = order_calc_profit

    def order_send(req):
        a = req["action"]
        if a == TRADE_ACTION_REMOVE:
            mk.pending.pop(req["order"], None)
            return NS(retcode=TRADE_RETCODE_DONE, order=req["order"], comment="removed")
        if a == TRADE_ACTION_SLTP:
            pos = next(k for k, p in mk.positions.items() if p.get("ticket", k) == req["position"])
            mk.positions[pos]["sl"] = req["sl"]
            return NS(retcode=TRADE_RETCODE_DONE, order=0, comment="modified")
        if a == TRADE_ACTION_PENDING:
            tk = mk._tk()
            mk.pending[tk] = dict(req, bar=mk.i)
            return NS(retcode=TRADE_RETCODE_PLACED, order=tk, comment="placed")
        if a == TRADE_ACTION_DEAL and "position" in req:
            pos = next(k for k, p in mk.positions.items() if p["ticket"] == req["position"])
            mk._close(pos, mk.c[mk.i], DEAL_REASON_EXPERT, mk.t[mk.i] + 1)
            return NS(retcode=TRADE_RETCODE_DONE, order=mk._tk(), comment="closed")
        if a == TRADE_ACTION_DEAL:
            tk = mk._tk()
            nxt = mk.t[mk.i + 1] if mk.i + 1 < len(mk.t) else mk.t[mk.i] + 1
            mk._open(tk, dict(req, ticket=tk, bar=mk.i), nxt)
            return NS(retcode=TRADE_RETCODE_DONE, order=tk, comment="done")
        return NS(retcode=10013, order=0, comment="invalid request")
    m.order_send = order_send

    m.orders_get = lambda symbol=None: tuple(NS(ticket=tk, magic=p["magic"]) for tk, p in mk.pending.items())

    def positions_get(symbol=None):
        out = []
        for tk, p in mk.positions.items():
            p.setdefault("ticket", tk)
            out.append(NS(ticket=p["ticket"], identifier=tk, magic=p["magic"], volume=p["volume"], sl=p["sl"], tp=p["tp"],
                          type=POSITION_TYPE_BUY if p["type"] in (ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT)
                          else POSITION_TYPE_SELL))
        return tuple(out)
    m.positions_get = positions_get
    m.history_deals_get = lambda a, b, group=None: tuple(mk.deals)
    return m
