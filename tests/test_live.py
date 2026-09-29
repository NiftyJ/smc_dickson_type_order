"""
Checks for the running version (smcml/live.py, live_mt5.py).

  * a paper replay of a history gives the SAME trades as the backtest
  * the live signal code, looking only at the last N bars, finds the same setups
  * the MT5 connection code works against a pretend MT5 terminal (tests/fake_mt5.py):
    orders, fills, stops, targets, lot sizes, demo-only lock

Run with:  python -m pytest tests
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config as cfg  # noqa: E402
from smcml.data import simulate  # noqa: E402
from smcml.live import Engine, LiveSignals, ReplaySignals, cfg_copy, replay  # noqa: E402
from smcml.risk import simulate_account, drawdown_target  # noqa: E402
import fake_mt5  # noqa: E402

# your setup code, with fixed account settings so the counts below stay meaningful
C = cfg_copy(cfg, BIAS_FILTER=False, COOLDOWN=False, REENTRY=True, DYNAMIC_TARGET=True, MAX_OPEN_TRADES=1)
DF = simulate(20000, mode="random", seed=31)
TMP = tempfile.mkdtemp()
ENG, SIG = replay(DF, C, os.path.join(TMP, "paper"), max_charts=2)
PAPER = pd.read_csv(os.path.join(TMP, "paper", "trades.csv"))


def _backtest():
    done = SIG.lab[SIG.lab.complete]
    bt, _ = simulate_account(done, DF.index.normalize().to_numpy(), C, target_for=drawdown_target(C),
                             reentries=SIG.re, prices=DF)
    bt["time"] = DF.index[bt.t_place.to_numpy()]
    return bt


def test_replay_gives_the_backtest_trades():
    bt = _backtest()
    incomplete = SIG.lab.loc[~SIG.lab.complete, "time"]
    cut = incomplete.min() if len(incomplete) else DF.index[-1]
    a = bt[bt.time < cut]
    b = PAPER[(PAPER.status != "skipped") & (pd.to_datetime(PAPER.placed_time) < cut)]
    assert len(a) > 10 and len(a) == len(b)
    a_keys = list(zip(a.time.astype(str), a.direction, a.is_reentry))
    b_keys = list(zip(pd.to_datetime(b.placed_time).astype(str), b.direction, b.is_reentry))
    assert a_keys == b_keys
    assert np.allclose(a.R_used.to_numpy(), b.R.to_numpy(), atol=2e-3)
    assert (a.target_R_used.to_numpy() == b.target_R.to_numpy()).all()
    assert a.is_reentry.any() or len(SIG.re) == 0               # re-entries were exercised too


def test_replay_logs_what_the_backtest_says():
    closed = PAPER[PAPER.status == "closed"]
    assert np.allclose(closed.R, closed.backtest_R, atol=2e-3)
    assert set(PAPER.status) >= {"closed", "expired", "skipped"}
    assert len(os.listdir(os.path.join(TMP, "paper", "charts"))) == 2


def test_live_signals_on_the_last_bars_match():
    """Live, the bot only sees the last N bars. With enough bars it must find exactly the
    setups the backtest finds on the full history."""
    c = cfg_copy(cfg)                                       # bias filter ON: it needs the most history
    df = simulate(26000, mode="random", seed=5)
    full = ReplaySignals(df, c)
    live = LiveSignals(c)
    bars = sorted(set(full.cands.t_place.to_numpy()))
    rng = np.random.default_rng(0)
    pick = [b for b in rng.choice(bars, 30, replace=False) if b > 21000][:6]
    assert len(pick) >= 3
    for i in pick:
        key = lambda rows: sorted((r["direction"], round(r["entry"], 6), round(r["stop"], 6)) for r in rows)
        assert key(full.new_setups(df.iloc[:i + 1])) == key(live.new_setups(df.iloc[i - 20000 + 1:i + 1]))


def _fake_run(demo=True, n=20000, volume_min=0.001):
    mk = fake_mt5.Market(DF.iloc[:n], demo=demo, volume_min=volume_min)
    sys.modules["MetaTrader5"] = fake_mt5.make_module(mk)
    import live_mt5
    broker = live_mt5.MT5Broker("TEST", "M15")
    broker.symbol = "TEST"
    out = tempfile.mkdtemp()
    sig = ReplaySignals(DF.iloc[:n], C)
    eng = Engine(broker, sig, C, out, charts=False, verbose=False, autosave=False)
    for i in range(int(sig.cands.t_place.min()), n):
        mk.advance(i)
        eng.on_bar(broker.closed_bars(30000))
    eng._save()
    return mk, pd.read_csv(os.path.join(out, "trades.csv"))


def test_mt5_broker_against_a_pretend_terminal():
    mk, t = _fake_run(n=8000)
    closed = t[t.status == "closed"]
    assert len(closed) > 3
    moved = closed.be_moved.fillna(False).astype(bool)
    stops = closed[(closed.exit_reason == "sl") & ~moved]
    be_exits = closed[(closed.exit_reason == "sl") & moved]
    assert (be_exits.R.abs() < 0.05).all()                  # stop moved to entry: out at about 0R
    # lot size from the stop distance: a stop-out loses about RISK_PER_TRADE_PCT, never more
    assert (stops.R > -1.0 - 1e-6).all() and (stops.R < -0.9).all()
    # results agree with what the backtest says for the same orders (fills can differ
    # slightly: a market order at the close instead of a limit filled on the next bar)
    agree = (closed.R - closed.backtest_R).abs() < 0.15
    assert agree.mean() > 0.8
    # the first trades are the same ones the paper replay took
    paper = PAPER[PAPER.status.isin(["closed", "expired"])].head(8)
    assert list(t[t.status.isin(["closed", "expired"])].head(8).key) == list(paper.key)
    assert abs(mk.balance - (10_000 + closed.profit.sum())) < 0.01 * len(closed)     # profits logged to the cent


def test_smallest_lot_too_big_means_no_trade():
    _, t = _fake_run(n=6000, volume_min=100.0)
    assert (t.status == "skipped").all()
    assert t.note.str.contains("smallest lot").all()


def test_real_account_is_refused_without_allow_real():
    mk = fake_mt5.Market(DF.iloc[:3000], demo=False)
    sys.modules["MetaTrader5"] = fake_mt5.make_module(mk)
    import live_mt5
    args = type("A", (), dict(symbol="TEST", timeframe="M15", magic=1, login=None, password=None, server=None,
                              allow_real=False, bars=1000, poll=0))
    with pytest.raises(SystemExit) as e:
        live_mt5.run_live(args)
    assert "REAL" in str(e.value)


@pytest.mark.parametrize("trigger", ["stops_in_a_row", "losing_days"])
def test_range_lock_replay_matches_backtest_and_really_locks(trigger):
    """The journal's range lock, both triggers: replay = backtest, and it really blocks orders."""
    c = cfg_copy(cfg, BIAS_FILTER=False, COOLDOWN=True, COOLDOWN_TRIGGER=trigger, COOLDOWN_AFTER_STOPS=3,
                 COOLDOWN_LOSING_DAYS=3, COOLDOWN_WINDOW_DAYS=10, COOLDOWN_DAYS=15, COOLDOWN_ENDS_ON_BREAKOUT=True,
                 LOSS_BLOCK_COUNT=3, MAX_ORDERS_PER_24H=5, REENTRY=True, DYNAMIC_TARGET=True, MAX_OPEN_TRADES=1)
    df = DF.iloc[:12000]
    out = os.path.join(TMP, f"lock_{trigger}")
    eng, sig = replay(df, c, out, max_charts=0)
    t = pd.read_csv(os.path.join(out, "trades.csv"))
    assert (t.note == "range lock").sum() >= 1
    done = sig.lab[sig.lab.complete]
    bt, blocked = simulate_account(done, df.index.normalize().to_numpy(), c, target_for=drawdown_target(c),
                                   reentries=sig.re, prices=df)
    assert blocked["range lock"] >= 1
    bt["time"] = df.index[bt.t_place.to_numpy()]
    incomplete = sig.lab.loc[~sig.lab.complete, "time"]
    cut = incomplete.min() if len(incomplete) else df.index[-1]
    a, b = bt[bt.time < cut], t[(t.status != "skipped") & (pd.to_datetime(t.placed_time) < cut)]
    assert list(a.time.astype(str)) == list(pd.to_datetime(b.placed_time).astype(str))
    assert np.allclose(a.R_used.to_numpy(), b.R.fillna(0).to_numpy(), atol=2e-3)


def test_loss_block_and_order_cap_hold_in_the_backtest():
    c = cfg_copy(cfg, BIAS_FILTER=False, COOLDOWN=False, LOSS_BLOCK_COUNT=2, LOSS_BLOCK_WINDOW_HOURS=24,
                 LOSS_BLOCK_HOURS=24, MAX_ORDERS_PER_24H=2, MAX_OPEN_TRADES=1, REENTRY=True, DYNAMIC_TARGET=True)
    done = SIG.lab[SIG.lab.complete]
    bt, blocked = simulate_account(done, DF.index.normalize().to_numpy(), c, target_for=drawdown_target(c),
                                   reentries=SIG.re, prices=DF)
    assert blocked["loss block"] > 0 and blocked["max orders per 24h"] > 0
    times = DF.index[bt.t_place.to_numpy()]
    for i, t in enumerate(times):                                   # never more than 2 orders in 24h
        assert ((times[:i + 1] > t - pd.Timedelta(days=1)).sum()) <= 2
    exits = DF.index[bt.t_exit_used.to_numpy()]
    stops = [(e, i) for i, (e, r) in enumerate(zip(exits, bt.R_used)) if r < -0.99]
    for k in range(1, len(stops)):                                  # 2 losses within 24h -> 24h without orders
        (e1, _), (e2, i2) = stops[k - 1], stops[k]
        if e2 - e1 < pd.Timedelta(hours=24):
            later = times[(times > e2) & (times < e2 + pd.Timedelta(hours=24))]
            assert len(later) == 0 or k >= 2
