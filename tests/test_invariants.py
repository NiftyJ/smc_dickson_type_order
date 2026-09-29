"""
Things that must ALWAYS be true. If one of these fails after you change something,
the change has a bug.

Run with:  python -m pytest tests      (or)      python tests/test_invariants.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from smcml.data import simulate  # noqa: E402
from smcml.setups import scan  # noqa: E402
from smcml.labels import label  # noqa: E402
from smcml.risk import simulate_account, drawdown_target  # noqa: E402
from smcml.walkforward import choose_threshold  # noqa: E402

_DF = simulate(20000, mode="random", seed=21)
_old, cfg.BIAS_FILTER = cfg.BIAS_FILTER, False        # more setups to check
_LAB = label(scan(_DF, cfg), _DF, cfg)
cfg.BIAS_FILTER = True
_FILTERED = scan(_DF, cfg)
cfg.BIAS_FILTER = _old


def test_bias_filter_only_trades_with_all_timeframes():
    assert len(_FILTERED) > 0
    for col in ("bias_d1", "bias_h4", "bias_h1"):
        assert (_FILTERED[col] == 1).all(), col
    longs, shorts = _FILTERED[_FILTERED.direction == 1], _FILTERED[_FILTERED.direction == -1]
    for tf in ("d1", "h4", "h1"):
        assert (longs[f"bias_{tf}_trend"] == 1).all() and (shorts[f"bias_{tf}_trend"] == -1).all()


def test_bias_filter_keeps_a_subset():
    key = ["t_place", "direction"]
    allowed = _FILTERED.set_index(key).index
    everything = _LAB.set_index(key).index
    assert allowed.isin(everything).all()
    assert len(allowed) < len(everything)


def test_stop_is_on_the_right_side():
    longs, shorts = _LAB[_LAB.direction == 1], _LAB[_LAB.direction == -1]
    assert (longs.stop < longs.entry).all()
    assert (shorts.stop > shorts.entry).all()


def test_order_comes_after_the_sweep_and_the_break():
    assert (_LAB.sweep_bar < _LAB.t_place).all()
    assert (_LAB.ext_bar <= _LAB.t_place).all()
    assert (_LAB.bars_sweep_to_shift <= cfg.MAX_BARS_SWEEP_TO_SHIFT).all()


def test_stop_is_not_too_tight():
    assert (_LAB.risk_atr >= cfg.MIN_RISK_ATR - 1e-12).all()


def test_fill_happens_after_placement_and_at_the_entry_price():
    f = _LAB[_LAB.filled == 1]
    assert (f.t_fill > f.t_place).all()
    assert (f.t_fill - f.t_place <= cfg.MAX_BARS_WAIT_FILL).all()
    lows, highs = _DF.low.to_numpy(), _DF.high.to_numpy()
    fl = f[f.direction == 1]
    assert (lows[fl.t_fill.to_numpy()] <= fl.entry.to_numpy()).all()
    fs = f[f.direction == -1]
    assert (highs[fs.t_fill.to_numpy()] >= fs.entry.to_numpy()).all()


def test_results_are_consistent():
    d = _LAB[_LAB.complete]
    win = d[d.y == 1]
    assert np.allclose(win.outcome_R, cfg.TARGET_R - win.cost_price / win.risk)
    assert (d[d.filled == 0].outcome_R == 0).all()
    assert (d.outcome_R >= -1 - d.cost_price / d.risk - 1e-9).all()   # never lose more than 1R + costs
    assert (d.t_exit >= d.t_place).all()


def test_random_walk_gives_random_walk_results():
    """On a random walk, a 3R target is hit first about 1 time in 4. If the labeller
    reports something very different, the labeller is wrong."""
    f = _LAB[(_LAB.filled == 1) & _LAB.hit_3.notna()]
    rate = f.hit_3.mean()
    assert 0.17 < rate < 0.33, rate


def test_risk_guard_limits_open_trades():
    d = _LAB[_LAB.complete].copy()
    bar_day = _DF.index.normalize().to_numpy()
    trades, _ = simulate_account(d, bar_day, cfg, prices=_DF)
    events = sorted([(t, +1) for t in trades.t_place] + [(t, -1) for t in trades.t_exit], key=lambda e: (e[0], e[1]))
    open_now = peak = 0
    for _, step in events:
        open_now += step
        peak = max(peak, open_now)
    assert peak <= cfg.MAX_OPEN_TRADES


def test_target_switches_only_in_deep_drawdown():
    """10R only while CLOSED trades have the account >20% below its peak; 20R otherwise."""
    d = _LAB[_LAB.complete].copy()
    bar_day = _DF.index.normalize().to_numpy()
    old = cfg.DD_SWITCH_PCT, getattr(cfg, "COOLDOWN", False)
    cfg.DD_SWITCH_PCT, cfg.COOLDOWN = 3.0, False        # low switch point so both modes get used
    try:
        trades, _ = simulate_account(d, bar_day, cfg, target_for=drawdown_target(cfg), prices=_DF)
        _check_switching(trades)
    finally:
        cfg.DD_SWITCH_PCT, cfg.COOLDOWN = old


def _check_switching(trades):
    assert set(trades.target_R_used) == {cfg.TARGET_R, cfg.TARGET_R_IN_DRAWDOWN}
    rows = list(trades.itertuples(index=False))
    for i, r in enumerate(rows):
        closed = [x for x in rows[:i] if x.t_exit_used <= r.t_place]
        eq = np.cumsum([x.R_used * cfg.RISK_PER_TRADE_PCT for x in closed]) if closed else np.array([0.0])
        equity, peak = eq[-1], max(0.0, eq.max())
        dd = (peak - equity) / (100 + peak) * 100
        expected = cfg.TARGET_R_IN_DRAWDOWN if dd > cfg.DD_SWITCH_PCT else cfg.TARGET_R
        assert r.target_R_used == expected, (i, dd, r.target_R_used)
        assert r.R_used == getattr(r, f"R_{r.target_R_used}")


def test_break_even_moves_the_stop_after_7R():
    """A long runs +8R, comes back to the entry, then runs to the target. With break-even at
    7R it closes at 0R; without it, it reaches the 25R target."""
    from smcml.live import cfg_copy
    from smcml.labels import chance_rate
    idx = pd.date_range("2024-01-01", periods=12, freq="15min")
    bars = pd.DataFrame({"open": [100.5, 100.2, 104, 107.5, 106, 102, 101, 110, 120, 128, 129, 131],
                         "high": [100.6, 100.3, 105, 108.2, 106.5, 102.5, 101.5, 112, 122, 129, 130, 132],
                         "low": [100.4, 99.9, 103, 107, 104, 100.8, 99.95, 101, 110, 120, 127, 128],
                         "close": [100.5, 100.1, 104.5, 107.8, 104.5, 101.0, 100.8, 111, 121, 128.5, 129.5, 131]},
                        index=idx)
    cand = pd.DataFrame([dict(t_place=0, direction=1, entry_f=100.0, stop_f=99.0, risk=1.0, cost_price=0.0)])
    with_be = label(cand, bars, cfg_copy(cfg, TARGET_R=25, EXTRA_TARGETS=(25,), BREAKEVEN_AT_R=7)).iloc[0]
    without = label(cand, bars, cfg_copy(cfg, TARGET_R=25, EXTRA_TARGETS=(25,), BREAKEVEN_AT_R=None)).iloc[0]
    assert with_be.t_fill == 1 and with_be.R_25 == 0.0 and with_be.hit_25 == 0 and with_be.t_exit_25 == 6
    assert without.R_25 == 25 and without.hit_25 == 1 and without.t_exit_25 == 9
    assert abs(chance_rate(25, cfg_copy(cfg, BREAKEVEN_AT_R=7)) - 7 / 200) < 1e-12
    assert abs(chance_rate(25, cfg_copy(cfg, BREAKEVEN_AT_R=None)) - 1 / 26) < 1e-12


def test_threshold_refuses_to_trade_without_edge():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    R = np.full(500, -0.05)                       # everything loses a little
    assert choose_threshold(scores, R, 30) == np.inf
    R2 = np.where(scores > 0.8, 2.0, -1.0)        # top scores win
    assert 0.75 < choose_threshold(scores, R2, 30) < 0.85


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
