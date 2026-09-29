"""
ALL SETTINGS LIVE HERE.
Change numbers in this file, not inside the code. Each setting says what it does.
"""

# ---------------------------------------------------------------- data
# Path to your MT5 export (see README, "Getting your data out of MT5").
# Leave as None to use the built-in simulator instead.
DATA_CSV = None

# Simulator used when DATA_CSV is None.
#   "random"  = a Volatility-75-style random walk. There is NO edge in it.
#               Your pipeline must NOT find one here. If it does, something is broken.
#   "planted" = the same random walk plus a hidden pattern that really does
#               produce 20R runners. Your pipeline SHOULD find this one.
SIM_MODE = "random"
SIM_BARS = 100_000         # 15-minute bars (100,000 is about 2.9 years of 24/7 data)
SIM_SEED = 7

# ---------------------------------------------------------------- detectors
ATR_N = 14                 # ATR length. Almost every size is measured in ATRs so that
                           # features mean the same thing in quiet and wild markets.
SWING_N = 5                # A swing high = highest high with 5 bars on each side.
                           # IMPORTANT: it is only KNOWN 5 bars later. The code respects that.
HTF_SWING_N = 25           # Bigger swings, used as the "higher timeframe" structure.

# ---------------------------------------------------------------- top-down bias
# Higher-timeframe structure, built from your bars. Bullish = the last structure break
# on that timeframe was a close above a swing high; bearish = a close below a swing low.
# Only candles that have already CLOSED are used (no peeking at the forming candle).
BIAS_TIMEFRAMES = ("1D", "4h", "1h")   # D1 -> H4 -> H1
BIAS_SWING_N = 3               # swing size on those timeframes (3 candles each side)
BIAS_FILTER = True             # True = only take longs when ALL are bullish, shorts when ALL bearish
                               # False = take everything, just record the biases as features

# ---------------------------------------------------------------- setup state machine
# Long setup = sweep of a swing low -> bullish structure break -> limit order at the
# order block. Shorts are the mirror image (the code literally flips the chart).
MAX_BARS_SWEEP_TO_SHIFT = 30   # the structure break must come within 30 bars of the sweep
MAX_BARS_WAIT_FILL = 40        # a limit order that is not filled in 40 bars is cancelled
STOP_BUFFER_ATR = 0.05         # stop goes this far beyond the sweep extreme
MIN_RISK_ATR = 0.25            # skip setups whose stop is tighter than this (spread would eat them)

# ---------------------------------------------------------------- trade outcome
TARGET_R = 25                  # the reward you aim for, in R (journal: "anything else go for 25R in profit")
EXTRA_TARGETS = (3, 5, 10, 12, 20, 25)   # also report what would happen at these targets
MAX_HOLD_BARS = 2000           # close the trade at market after this many bars (~3 weeks of M15)
# Smaller target while the account is deep in drawdown:
DYNAMIC_TARGET = True          # False = always use TARGET_R
DD_SWITCH_PCT = 20.0           # switch when the account is more than 20% below its highest point
TARGET_R_IN_DRAWDOWN = 12      # the target used during that drawdown (journal: "stick to 12R"; must be in EXTRA_TARGETS)
# Break-even (journal: "sl shift only after 7R", "break even if you hit 7R+"): once a trade
# is this many R in profit, the stop moves to the entry. None = never move the stop.
BREAKEVEN_AT_R = 7
# More shots after a stop-out: a re-entry in the same direction after a break of structure
# on the entry timeframe (your bars: M15 here, M5 if you load M5 data).
REENTRY = True
REENTRY_MAX_SHOTS = 3          # trades per setup in total (journal: "max 3 tries"); 2 = one re-entry
REENTRY_WINDOW_BARS = 96       # look for that break for up to 96 bars after the stop (1 day on M15)
REENTRY_ENTRY = "order_block"  # "order_block": limit order at the new order block (same as the first entry)
                               # "break": enter at the close of the break candle (always fills, wider stop)
# ---------------------------------------------------------------- journal locks
# Range lock (journal: "after 3 successive losing days on a symbol apply 15 day lock",
# "15-20 day range, condition is 3 losing days in a 10 day range", "wait for top or bottom to break")
COOLDOWN = True                # True = lock the symbol when it looks like a range
COOLDOWN_TRIGGER = "losing_days"   # "losing_days": COOLDOWN_LOSING_DAYS losing days within COOLDOWN_WINDOW_DAYS
                                   # "stops_in_a_row": COOLDOWN_AFTER_STOPS stop-outs in a row
COOLDOWN_LOSING_DAYS = 3
COOLDOWN_WINDOW_DAYS = 10
COOLDOWN_AFTER_STOPS = 3
COOLDOWN_DAYS = 15             # no new orders for up to this many days
COOLDOWN_ENDS_ON_BREAKOUT = True   # unlock early once a candle closes outside the range price was stuck in
# Loss block (journal: "block orders for 24 hours if you hit 3+ losses"). 0 = off.
LOSS_BLOCK_COUNT = 3           # this many stop-outs ...
LOSS_BLOCK_WINDOW_HOURS = 24   # ... within this many hours ...
LOSS_BLOCK_HOURS = 24          # ... = no new orders for this many hours
# Journal: "no more than 5 trades in total for a span of 24 hours". 0 = off.
MAX_ORDERS_PER_24H = 5
# Trading cost per trade (spread + commission + slippage), charged in R.
#   "atr":   cost = COST_ATR x ATR   (handy for the simulator)
#   "price": cost = COST_PRICE in price units (use your real spread for real data)
COST_MODE = "atr"
COST_ATR = 0.03
COST_PRICE = 0.0

# ---------------------------------------------------------------- model and testing
MODEL = "gbm"                  # "gbm" (gradient boosted trees, start here) or "cnn" (needs PyTorch)
N_FOLDS = 5                    # walk-forward: the test period is split into 5 chunks
VALID_FRACTION = 0.3           # last 30% of each training period is used to pick the threshold
MIN_TRADES_FOR_THRESHOLD = 30  # never pick a threshold that is based on fewer trades than this
RULE_REPORT_FRACTION = 0.4     # the rule report only looks at the first 40% of history
# THE LOCKBOX: the last 20% of your history is hidden from every normal run.
# Tinker, change features, rerun as often as you like on the other 80%. When you are
# finished, run ONCE with --final to test on the locked 20%. If you then keep changing
# things and re-running --final, the lockbox is spent and its result means nothing.
LOCKBOX_FRACTION = 0.2
# The loss-review loop (iterate.py) splits the rest again: you look at losses in the first
# 60% (development) and every change is scored on the next 20% (validation), unseen.
DEV_FRACTION = 0.6

# ---------------------------------------------------------------- risk guard
# Set these ONCE. The bot enforces them. Do not give yourself a way to override them.
RISK_PER_TRADE_PCT = 0.5       # % of account lost when a trade hits its stop (= 1R)
MAX_DAILY_LOSS_PCT = 2.0       # stop opening trades for the rest of the day after this loss
MAX_OPEN_TRADES = 1            # never hold more than this many trades/orders at once

OUTPUT_DIR = "outputs"
