"""
Checks for the re-entry rule (smcml/reentry.py): up to REENTRY_MAX_SHOTS trades per setup.

Run with:  python -m pytest tests      (or)      python tests/test_reentry.py
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
from smcml.reentry import find_reentries  # noqa: E402
from smcml.risk import simulate_account, drawdown_target  # noqa: E402

_old, cfg.BIAS_FILTER = cfg.BIAS_FILTER, False          # more setups to check
_DF = simulate(20000, mode="random", seed=31)
_LAB = label(scan(_DF, cfg), _DF, cfg)
_RE = find_reentries(_LAB, _DF, cfg)
cfg.BIAS_FILTER = _old
KEY = ["parent_id", "parent_target"]


def test_reentries_exist_and_are_unique():
    assert len(_RE) > 3
    assert not _RE.duplicated(KEY).any()                 # max ONE next shot per stopped trade
    assert _RE["shot"].between(2, cfg.REENTRY_MAX_SHOTS).all()
    assert not _RE["candidate_id"].isin(_LAB["candidate_id"]).any()   # ids never clash with setups
    assert not _RE["candidate_id"].duplicated().any()


def test_reentry_comes_after_the_stop_and_inside_the_window():
    parent = pd.concat([_LAB, _RE]).set_index("candidate_id")
    for r in _RE.itertuples(index=False):
        p = parent.loc[r.parent_id]
        assert r.shot == (2 if r.parent_id in set(_LAB.candidate_id) else int(p.shot) + 1)
        assert getattr(p, f"R_{r.parent_target}") < -0.99            # parent really was stopped out
        assert int(getattr(p, f"t_exit_{r.parent_target}")) == r.stopout_bar
        assert r.stopout_bar < r.t_place <= r.stopout_bar + cfg.REENTRY_WINDOW_BARS
        assert r.direction == p.direction                             # same direction as the failed trade
        assert (r.stop < r.entry) if r.direction == 1 else (r.stop > r.entry)


def test_reentry_has_no_lookahead():
    """Cut the data right after some re-entry decisions: they must come out identical."""
    old, cfg.BIAS_FILTER = cfg.BIAS_FILTER, False
    try:
        df = _DF.iloc[:8000]
        lab = label(scan(df, cfg), df, cfg)
        full = find_reentries(lab, df, cfg)
        cols = ["parent_id", "parent_target", "t_place", "direction", "entry", "stop", "stopout_bar"]
        rng = np.random.default_rng(0)
        for T in sorted(rng.choice(full["t_place"].to_numpy(), size=min(8, len(full)), replace=False) + 1):
            part_df = df.iloc[:T]
            part = find_reentries(label(scan(part_df, cfg), part_df, cfg), part_df, cfg)
            a = full[full["t_place"] < T][cols].sort_values(KEY).reset_index(drop=True)
            b = part[cols].sort_values(KEY).reset_index(drop=True) if len(part) else part
            pd.testing.assert_frame_equal(a, b[cols] if len(b) else a.iloc[:0], check_dtype=False)
    finally:
        cfg.BIAS_FILTER = old


def test_reentry_ignores_future_prices():
    """Crash every price after the re-entry bar: the re-entry itself must not change.
    (A version that peeks even a few bars ahead picks up the crash and fails this.)"""
    old, cfg.BIAS_FILTER = cfg.BIAS_FILTER, False
    try:
        df = _DF.iloc[:6000]
        full = find_reentries(label(scan(df, cfg), df, cfg), df, cfg)
        cols = ["parent_id", "parent_target", "t_place", "direction", "entry", "stop", "stopout_bar"]
        rng = np.random.default_rng(1)
        for k in sorted(rng.choice(full["t_place"].to_numpy(), size=min(8, len(full)), replace=False)):
            junk = df.copy()
            junk.iloc[k + 1:] = junk.iloc[k + 1:].to_numpy() * 0.5          # everything after k collapses
            got = find_reentries(label(scan(junk, cfg), junk, cfg), junk, cfg)
            a = full[full["t_place"] <= k][cols].sort_values(KEY).reset_index(drop=True)
            b = got[got["t_place"] <= k][cols].sort_values(KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
    finally:
        cfg.BIAS_FILTER = old


def test_break_entry_uses_the_break_candle_close_and_no_future():
    """REENTRY_ENTRY = "break": entry is the close of the break candle, and future prices
    still change nothing."""
    old = cfg.BIAS_FILTER, getattr(cfg, "REENTRY_ENTRY", "order_block")
    cfg.BIAS_FILTER, cfg.REENTRY_ENTRY = False, "break"
    try:
        df = _DF.iloc[:6000]
        lab = label(scan(df, cfg), df, cfg)
        full = find_reentries(lab, df, cfg)
        assert len(full) > 2
        assert np.allclose(full["entry"].to_numpy(), df["close"].to_numpy()[full["t_place"].to_numpy()])
        k = int(full["t_place"].iloc[len(full) // 2])
        junk = df.copy()
        junk.iloc[k + 1:] = junk.iloc[k + 1:].to_numpy() * 0.5
        got = find_reentries(label(scan(junk, cfg), junk, cfg), junk, cfg)
        cols = ["parent_id", "parent_target", "t_place", "entry", "stop"]
        a = full[full["t_place"] <= k][cols].sort_values(KEY).reset_index(drop=True)
        b = got[got["t_place"] <= k][cols].sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    finally:
        cfg.BIAS_FILTER, cfg.REENTRY_ENTRY = old


def test_account_takes_reentries_only_after_its_own_stopped_trade():
    d = _LAB[_LAB.complete]
    bar_day = _DF.index.normalize().to_numpy()
    old = getattr(cfg, "COOLDOWN", False), cfg.LOSS_BLOCK_COUNT, cfg.MAX_ORDERS_PER_24H
    cfg.COOLDOWN, cfg.LOSS_BLOCK_COUNT, cfg.MAX_ORDERS_PER_24H = False, 0, 0     # re-entries only, no locks
    try:
        trades, _ = simulate_account(d, bar_day, cfg, target_for=drawdown_target(cfg), reentries=_RE, prices=_DF)
    finally:
        cfg.COOLDOWN, cfg.LOSS_BLOCK_COUNT, cfg.MAX_ORDERS_PER_24H = old
    re = trades[trades.is_reentry]
    taken = trades.set_index("candidate_id")
    assert len(re) > 0
    assert not re.duplicated("parent_id").any()
    assert trades.shot.max() <= cfg.REENTRY_MAX_SHOTS
    for r in re.itertuples(index=False):
        p = taken.loc[r.parent_id]                                   # the parent trade was taken ...
        assert p.R_used < -0.99 and p.target_R_used == r.parent_target   # ... and stopped out
        assert r.shot == p.shot + 1
        assert p.t_exit_used <= r.t_place
    events = sorted([(t, 1) for t in trades.t_place] + [(t, -1) for t in trades.t_exit_used], key=lambda e: (e[0], e[1]))
    now = peak = 0
    for _, step in events:
        now += step
        peak = max(peak, now)
    assert peak <= cfg.MAX_OPEN_TRADES


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
