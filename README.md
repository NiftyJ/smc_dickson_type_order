# SMC → Machine Learning template

A working starting point for turning an SMC bot into:
**rules that find setups → a model that judges them → a risk guard that protects the
account → an honest exam that tells you whether any of it actually works.**

You don't need to read the papers. Every idea borrowed from them is explained in plain
words at the top of the file that uses it, and in the table below.

---

## The pipeline

```
price bars
  │
  ├─ detectors.py   swings (only once CONFIRMED), BOS / CHoCH, order blocks, FVGs
  ├─ bias.py        top-down bias: D1 → H4 → H1 structure, from CLOSED candles only
  ├─ setups.py      the state machine: bias agrees → liquidity sweep → structure break → limit order
  │                 + a list of FEATURES describing each setup
  ├─ rule_votes.py  your judgement rules, turned into votes (+1 / -1 / no opinion)
  ├─ labels.py      what actually happened: filled? stop (-1R), break-even (0R) or target first?
  ├─ model.py       a model that scores every setup
  ├─ walkforward.py the exam: only ever tested on periods it never saw, plus a placebo
  ├─ reentry.py     one more shot after a stop-out, on a break of structure
  ├─ risk.py        the risk guard, and the account simulation
  └─ live.py        the same decisions made one closed bar at a time (the running version)
run_all.py   runs everything and writes a plain-English report
iterate.py   the loss-review loop: look at losses, change the code, score the change honestly
live_mt5.py  runs the bot on MT5 (demo), or paper-replays your history
```

The big design decision, and the fix for your "rules overlap" problem:
**the state machine only contains what MUST be true for a setup to exist** (sweep, then
break, then an order at the order block). Every "it's better if…" rule (OB inside an
FVG, discount zone, HTF agrees, killzone…) is **not an if-statement**. It is written
down as a feature or a vote, and the model learns how much each one matters. With no
competing if-branches left, there's nothing left to overlap.

Shorts are handled by flipping the chart upside down, so there is one copy of every
rule and long and short can never drift apart.

---

## Quick start (about 10 minutes)

1. Install Python 3.10 or newer, then in this folder:
   ```
   pip install -r requirements.txt
   ```
2. Run the tests. They must all pass:
   ```
   python -m pytest tests
   ```
3. Run the **random** market (a Volatility-75-style random walk, no edge in it by construction):
   ```
   python run_all.py --no-bias-filter
   ```
   Expected: `SUMMARY: NO RELIABLE EDGE FOUND.`
4. Run the **planted** market (same random walk plus a hidden pattern that really works):
   ```
   python run_all.py --sim planted --no-bias-filter
   ```
   Expected: `SUMMARY: POSSIBLE EDGE, AND THE MODEL ADDS TO IT.`

   (The smoke test runs without the D1/H4/H1 filter because the filter removes about
   80% of setups, which leaves too few examples to find the planted pattern reliably.
   For your own data, leave the filter on if that's how you trade.)

Steps 3 and 4 work like testing a smoke detector. If it beeps in clean air, or stays
silent in smoke, it's broken. I checked both on 6 different random seeds each:
the random market gave "no edge" 6 out of 6 times, and the planted market was
found 6 out of 6 times.

Each run writes a folder in `outputs/` with `report.txt`, `equity.png`,
`candidates.csv` (every setup with its features and result), `rule_report.csv`
and `trades_model.csv`.

### The most important thing the random run shows

In the random run, **taking every setup made +190R over the test periods, or +45% of
the starting balance after the risk guard**. That market is a coin-flip generator, so
there is nothing to find. The profit is pure luck.

With 20R targets, a handful of lucky winners swings the total by 100R or more. That's
why the report never trusts a profit on its own. It always asks: *could luck have done
this?* (the p-values), and *would a model that learned nothing have done just as well?*
(the placebo).

---

## Rules from your trading journal (the defaults)

These come straight from your journal. Each one is a setting in `config.py`, and each works
the same way in the backtest, the paper replay and the MT5 bot (the tests check they give
identical trades).

| Journal | Setting | Default |
|---|---|---|
| "go for 25R in profit", "take 12 in drawdown", "stick to 12R" | `TARGET_R`, `TARGET_R_IN_DRAWDOWN`, `DD_SWITCH_PCT` | 25R, and 12R while more than 20% below the peak |
| "sl shift only after 7R", "break even if you hit 7R+" | `BREAKEVEN_AT_R` | 7: the stop moves to the entry from the next bar |
| "max 3 tries", "2 shots per BOS" | `REENTRY_MAX_SHOTS` | 3 trades per setup, each after a new break of structure |
| "3 losing days in a 10 day range", "15 day lock", "wait for top or bottom to break" | `COOLDOWN_*` | 3 losing days within 10 days lock the symbol for 15 days, or until a candle closes outside the range |
| "block orders for 24 hours if you hit 3+ losses" | `LOSS_BLOCK_*` | 3 stop-outs within 24h, then no orders for 24h |
| "no more than 5 trades in total for a span of 24 hours" | `MAX_ORDERS_PER_24H` | 5 |
| "don't tamper with the EA" | (no override switches) | the bot has none |

Not in: "4% max per size". Risk stays at 0.5%, because at a 1:20 hit rate 20 losses in a row
is routine, and at 4% that's -56% of the account. Also not in: "home run at a high POI"
(needs an H4/D1 point-of-interest detector) and "fast markets only".

Section 6 of every report switches each rule off one at a time, so you can see what each
one does on your own data. On my data (test periods, every setup, 0.5% risk):

| | Random (median of 30) | Gold | USDJPY | GBPJPY | EURUSD |
|---|---|---|---|---|---|
| **Journal profile** (all the rules) | -0.8% | -60.9% | -20.4% | -23.3% | -36.8% |
| without break-even | -2.1% | -56.0% | -17.6% | -16.2% | -44.3% |
| 1 shot (no re-entries) | -2.1% | -61.2% | -11.9% | -24.3% | -35.9% |
| without the range lock | +1.4% | -54.4% | -19.8% | -31.0% | -38.0% |
| always 25R (no 12R in drawdown) | -0.8% | -43.3% | -23.7% | -32.1% | -66.7% |
| old bot (20R / 10R, 2 shots, none of the journal rules) | -1.9% | -30.7% | -27.7% | -32.5% | -26.1% |

What that shows:
- **Compared with the old bot**, the journal profile did better on USDJPY and GBPJPY and
  worse on gold and EURUSD. On random prices both are about zero. No market turned profitable.
- **Each rule helps on one market and hurts on another.** Break-even helped EURUSD and hurt
  the other three. The range lock helped GBPJPY and hurt gold. That's what rules that
  reshape the losses without changing the odds look like.
- **Some rules almost never came into play.** With one trade open at a time and targets that
  take days, the 24h loss block and the 5-orders cap changed only 1 or 2 orders in total, and
  a 3rd shot happened 4 times in 30 random histories. They're harmless protections, not
  profit makers.
- The quick-start smoke tests still pass with the journal profile: random gave "no edge" 6
  out of 6 times, and the planted pattern was found 6 out of 6 times.

---

## Top-down bias (D1 → H4 → H1)

By default a long is only allowed when **D1, H4 and H1 are all bullish**, and a short only
when all three are bearish (`BIAS_FILTER = True` in `config.py`).

- The D1, H4 and H1 candles are built from your M15 bars, and the same swing and
  structure logic is run on each timeframe (swings of 3 candles each side).
- **Bullish** = the last structure break on that timeframe was a close above a swing high.
  **Bearish** = a close below a swing low.
- **No peeking:** at each M15 bar, only higher-timeframe candles that have already
  closed count. The H4 candle that is still forming is ignored, just as it could still
  change on a live chart. The tests check this, and they catch a version that peeks at
  the forming candle.
- The trade pictures (`--plot-trades`) show the D1, H4 and H1 charts at the moment of
  entry, with the break that set each bias, so you can check it matches your reading.
- `--no-bias-filter` (or `BIAS_FILTER = False`) takes every setup but still records
  each timeframe's bias, so the report shows whether the filter actually helps.

What I found on 30 random V75-style histories with the filter on: it cut the trades
from about 600 to about 157 per run and narrowed the swings (worst −39%, best +53%,
instead of −114% to +109%). But it was still 14 runs up and 16 down, with a 20R hit
rate of 5.1% vs 4.8% for pure chance. On random prices no filter can create an edge.
Whether it helps on a real market is exactly what running it on your data will show.

---

## Target switching in drawdown

(The defaults are now 25R and 12R, from your journal; see "Rules from your trading journal".
The results below were measured with the earlier 20R / 10R.)

With `DYNAMIC_TARGET = True`, each new order gets a **20R** target normally and a **10R**
target while the account is more than **20%** below its highest point
(`TARGET_R`, `TARGET_R_IN_DRAWDOWN`, `DD_SWITCH_PCT`). The drawdown is worked out from
trades that have already closed, at the moment the order is placed. Section 6 of the
report compares "always 20R", "always 10R" and the switching rule side by side.

What I found (D1/H4/H1 filter on, test periods only, 0.5% risk per trade):

| | Always 20R | Always 10R | Switching rule |
|---|---|---|---|
| 30 random histories: median result | -1.0% | +0.2% | -2.3% |
| 30 random histories: average result | +2.8% | +1.8% | +3.1% |
| Gold | -37.8% | -22.5% | -31.5% |
| USDJPY | -24.0% | -8.0% | -19.2% |
| GBPJPY | -51.3% | -42.4% | -36.1% |
| EURUSD | -44.5% | -33.7% | -23.4% |

The switch cut the losses compared with always 20R on all four real markets, but it
didn't turn any of them profitable, and on random data all three average about zero.
Choosing the target from past results changes how the losses arrive, not the odds of
the next trade.

---

## One more shot after a stop-out (re-entry)

(Now up to `REENTRY_MAX_SHOTS` = 3 trades per setup in total, from your journal's "max 3 tries".
The results below were measured with one re-entry, i.e. 2 shots.)

With `REENTRY = True`, a setup that gets stopped out may get **one** more trade:

1. After the stop, watch the entry chart for up to `REENTRY_WINDOW_BARS` bars (96 = 1 day on M15).
2. Wait for a break of structure in the **same direction** as the failed trade (for a long,
   a candle closes above the last confirmed swing high).
3. D1, H4 and H1 must still all agree at that moment.
4. Place the order the same way as the first one: a limit at the top of the order block at
   the new low, the stop just under the new low, and a 20R target (10R in deep drawdown).
5. If the re-entry is stopped too, the setup is finished. A re-entry never gets a re-entry.

The break is looked for on whatever timeframe you load. With M15 data that's M15. For
M5, export M5 bars from MT5 and run on those. Every "bars" setting then counts M5 bars,
so 96 bars is 8 hours instead of a day.

Section 6 of the report adds a "switching + 1 re-entry" row. The tests check that a
re-entry only uses prices up to its own decision bar.

`example_reports/reentry_gold_setup239.png` shows the rule on the gold trade from
21 Jan 2016. The break of structure came at 22:45, and the order sat at the new order
block for 10 hours without being filled. It was cancelled, and price reached it 6 hours
later. A filled order would have made 20R.

What I found (D1/H4/H1 filter on, drawdown switching on, test periods only):

| | Switching rule | + 1 re-entry (order block) | + 1 re-entry (at the break) |
|---|---|---|---|
| 30 random histories: median result | -2.3% | -1.9% | +2.0% |
| 30 random histories: average result | +3.1% | +4.2% | +2.3% |
| Gold | -31.5% | -30.7% | -28.4% |
| USDJPY | -19.2% | -27.7% | -22.9% |
| GBPJPY | -36.1% | -32.5% | -21.0% |
| EURUSD | -23.4% | -26.1% | -39.4% |

- **Order block entry (the default):** on the four real markets the rule fired 146 times.
  Only 46 orders filled, because price usually left without coming back to the new order
  block, and 1 of those 46 reached its target. On random prices, 86 of 444 filled and 7
  reached target, which is the rate chance gives at 10R–20R.
- **Waiting a full day for the fill** instead of 10 hours: more fills on the real markets
  (64), still only 1 winner.
- **Entering at the close of the break candle** (`REENTRY_ENTRY = "break"`): every order
  fills, but the stop sits under the new low, far below the entry, so the target in R is
  a very long way off. On the real markets 2 of 118 reached target (both gold). Gold and
  USDJPY still came out ahead on slow time exits, while GBPJPY and EURUSD lost. On random
  prices it was 0 targets in 375, for -25.7R.

The re-entry changes which trades you take, not the odds of each one. It didn't make
any of these markets profitable. Trying several versions on the same history is also
how luck gets in, so pick one before your final run.

---

## Range lock after repeated losses

(Now **on by default** with the journal's trigger, `COOLDOWN_TRIGGER = "losing_days"`: 3 losing
days within 10 days lock the symbol for 15 days or until price breaks out of the range.
The results below were measured with the other trigger, `"stops_in_a_row"`.)

With `COOLDOWN = True`, after `COOLDOWN_AFTER_STOPS` stop-outs in a row (first entries and
re-entries all count) the bot places no new orders on that symbol for `COOLDOWN_DAYS`
days. If `COOLDOWN_ENDS_ON_BREAKOUT` is on, it resumes earlier once a candle closes
outside the high/low of the failed shots. It's in the backtest, the running version and
the paper replay (the tests check they agree). Section 6 of every report shows a
"+ range pause" row, even while it's switched off, so you can see what it would do.

On my data this trigger didn't reliably help:

| Test periods, D1/H4/H1 on | No pause | 3 stops, 15 days or breakout | 3 stops, 15 days | 2 stops, 15 days |
|---|---|---|---|---|
| 30 random histories: median | -1.9% | +0.2% | +3.3% | -2.6% |
| 30 random: target hit rate | 5.7% | 5.5% | 5.2% | 4.5% |
| Gold | -30.7% | -43.9% | -35.7% | -28.5% |
| USDJPY | -27.7% | -24.1% | -28.7% | +7.8% |
| GBPJPY | -32.5% | -39.9% | -18.9% | -60.5% |
| EURUSD | -26.1% | -54.4% | -68.6% | -33.2% |

If the pause really skipped ranges, the trades left over would hit their target more
often. They didn't. Two reasons:
- **At 1:20, losing streaks are the normal state.** At a 6% hit rate, 3 stop-outs in a
  row happen 83% of the time. Even at 4 wins in 21, it's 53%. So the rule fires most of
  the time, ranging or not.
- **"Went into profit, then came back to the stop" happens on random prices too.** Of
  the stopped trades, 46% went +1R first on random prices and 47% on real gold (+3R
  first: 20% vs 21%).

In the loss-review loop on gold, the pause (3 stops, 15 days or breakout) removed 34
losses and no winners, and the validation period went from +12.6R to +24.4R. But the
development period got slightly worse (-7.4R to -8.1R), so the loop said REVERT. Test it
on your own V75 or Spredix history: set `COOLDOWN = True`, then
`python iterate.py check "range pause"`.

---

## Real markets vs the simulator (what I found)

`python get_sample_data.py` downloads free real M15 history (2012-2022) for gold,
USDJPY, GBPJPY and EURUSD. On those, with rough raw-account costs, the same SMC rules
hit 20R on 3.4-3.9% of filled trades with the D1/H4/H1 filter and 3.8-4.8% without it,
against 4.8% for a random walk. No model found anything to pick. That is with my
version of the rules; yours may differ, so run your own.

The prices themselves look very different, though. Real markets have sudden huge
moves, quiet and wild periods that cluster, and busy trading sessions. The
simulator (like Deriv's description of V75) has none of those.

---

## Getting your data out of MT5

1. In MT5: **View → Symbols** (Ctrl+U) → **Bars** tab.
2. Pick the symbol (e.g. Volatility 75 Index) and timeframe (M15 to start with).
3. Set the date range as far back as it goes. **Aim for at least 3 years of M15.**
   In testing, with 1.7 years the model sometimes failed to learn the planted
   pattern (3 of 6 times); with 2.9 years it learned it every time (6 of 6).
4. Click **Request**, then **Export Bars**, and save the CSV in the `data/` folder.
   (`data/example_mt5_export.csv` shows the format it will look like.)
5. Run:
   ```
   python run_all.py --data "data/Volatility 75 Index_M15.csv"
   ```
6. **Set your real trading cost.** In `config.py` set `COST_MODE = "price"` and
   `COST_PRICE` to your spread (+ commission, + a little slippage) in price units.
   The simulator's default cost is just a guess.

---

## Which paper idea is where

| Idea | Borrowed from | In plain words | File |
|---|---|---|---|
| Rules find, model judges | Meta-labeling (López de Prado; Joubert, 2022) | Your SMC logic proposes trades. A second model only says "take it" or "skip it". | `setups.py`, `model.py` |
| Rules as votes | Snorkel (Ratner et al., 2017) | Overlapping, contradicting rules are fine if each is a *vote* and data decides their weight. | `rule_votes.py` |
| Look at the chart | Jiang, Kelly & Xiu (2023) | Draw each setup as a small picture and let a CNN look at it, like your eye does. | `chart_images.py`, `model.py` (cnn) |
| Rare winners | Focal loss (Lin et al., 2017) | Stop the 95% obvious losers from drowning out the 5% winners during training. | `model.py` |
| Copy your eye | NVIDIA self-driving (Bojarski et al., 2016) | Mark setups TAKE/SKIP; a model learns to copy your judgement. | `my_calls.py` |
| Don't fool yourself | "Spurious predictability" (2026); López de Prado | Placebo model, walk-forward, lockbox, luck checks. | `walkforward.py`, `run_all.py` |
| Grade by outcome, not direction | Triple-barrier labeling | A setup is good if the target comes before the stop, costs included. | `labels.py` |

---

## Step by step: using it for your bot

### Step 1: Put your own SMC logic in
- **How swings, OBs and FVGs are defined:** `detectors.py`
- **The minimal setup sequence:** `setups.py` (the `SetupMachine` class)
- **Your judgement rules:** add one line per rule to `RULES` in `rule_votes.py`.
  Rules may overlap and contradict each other; that's allowed now.
- **New measurements of a setup:** add them in `SetupMachine._candidate()` and to the
  `FEATURES` list. Everything must be known at the placement bar.
- **After every change:** `python -m pytest tests`. The no-lookahead test cuts the
  data off right after a sample of setups and checks nothing about them changes. I checked that it
  catches the classic SMC bug (using a swing point before it's confirmed).

If you use the `smartmoneyconcepts` library instead of `detectors.py`: its swing points
are marked on the swing bar but need `swing_length` future bars to be known. Shift
them forward by `swing_length` before using them, then run the tests.

### Step 2: Run it on your V75 history
First check that the code marks setups the way you would:
```
python run_all.py --data "data/V75_M15.csv" --plot-trades 12
```
This saves 12 pictures (winners, stop-outs, unfilled orders) in
`outputs/<run>/trade_charts/`: the swept swing, the sweep, the CHoCH/BOS, the order
block, the FVG, entry/stop/target, and the whole trade. If a picture shows something
you wouldn't call a setup, change the definition in `detectors.py` / `setups.py`.

Then read **section 3** (raw results at 3R/5R/10R/20R next to the random-walk rate) and
**section 7A** (do the setups themselves beat a random walk, beyond luck?). This is the
direct test of "4 in 21". A real 4 in 21 over a few hundred trades would show
`p` far below 0.01 there.

### Step 3: Clean up your rules
Section 4, the rule report: for each rule, the win rate when it says YES vs NO.
If they're about equal, the rule does nothing; delete it. It also shows how often your
rules contradict each other (on the demo data: about 94% of setups have at least one
YES and one NO, which is why hand-coding the priorities kept breaking).

### Step 4: Test your own eye, blind
```
python run_all.py --data "data/V75_M15.csv" --export-review 150
```
This saves 150 chart images in `review/`. Each one **stops at the moment the order
would be placed**, so hindsight can't help. Open each image, write `TAKE` or `SKIP` in
`review/my_calls.csv`, then:
```
python run_all.py --data "data/V75_M15.csv" --my-calls review/my_calls.csv
```
Section 5a shows your TAKE hit rate vs your SKIP hit rate vs the random walk, with a
95% range and a p-value. Decide from the picture only; don't look the date up.
- TAKE clearly above 1 in 21 with p < 0.01: your eye has real skill. The
  `my_style` judge learns to copy it (it needs about 150+ calls to train).
- TAKE about the same as SKIP and about 1 in 21: your eye isn't adding anything on
  this market, and it's better to know that before risking money.

To see what this looks like first: `python run_all.py --sim planted --pretend-calls`.

### Step 5: Read the walk-forward exam (section 5)
Every number there is out-of-sample: each setup was judged by a model trained only on
earlier data. The judges:
- **all**: take every setup (no filter)
- **rules**: count of your rule votes
- **model**: the trained model
- **placebo**: the same model trained on *shuffled* results, so it learned nothing.
  If the model isn't clearly better than this, the model found nothing.
- **my_style**: copies your calls (only if you gave some)

Then **"Does the model RANK setups well?"**: the model's top third of setups vs
its bottom third. A useful model shows a clear positive gap with p < 0.05.

### Step 6: Improve, using only 80% of your history
The last 20% of your data is locked away (the **lockbox**). Tinker with features,
rules and settings as much as you like; normal runs never see the lockbox.
Use the loss-review loop for this (`iterate.py`, see "The loss-review loop" below):
it charts the losses, scores every change on data the change wasn't designed on, and
keeps the log for you.

### Step 7: Final exam, once
```
python run_all.py --data "data/V75_M15.csv" --final
```
This trains on everything before the lockbox and tests on it. **Run it once.** If you
change things after seeing the result and run it again, the lockbox is spent and
its result means nothing.

### Step 8: Demo account
Only if the final exam passes: run `live_mt5.py` on a demo account until you have 100+
trades, and check the live win rate and average R match the report (see "Running it
on MT5" below).

### Step 9: Live, with the risk guard
`RISK_PER_TRADE_PCT`, `MAX_DAILY_LOSS_PCT` and `MAX_OPEN_TRADES` in `config.py`.
In the live bot these must be enforced in code with no override button.

---

## Do's and don'ts

**Do**
- Run the random and planted demos after any big change. They're your smoke test.
- Include real costs. At 1:20 with tight stops, spread can be 10–30% of 1R.
- Judge a 1:20 system on **hundreds** of trades. With 50 trades, luck decides everything.
- Know your losing streaks before you trade. At a 5% win rate, the report shows
  runs of **80+ losses in a row** are normal over 1,000 trades. Size your risk so you
  survive that without touching anything.

**Don't**
- Don't shuffle data or use random train/test splits. Time only moves forward.
- Don't judge by accuracy. "Always predict loss" is 95% accurate and worthless.
- Don't rerun `--final` after tweaking.
- Don't believe a p-value after 50 reruns. Every rerun with a small change is another
  lottery ticket, which is why the lockbox exists.
- Don't override the risk guard. A limit you can switch off isn't a limit.

---

## About Volatility 75 / 100 specifically

Deriv describes these indices as prices "generated using a cryptographically secure
random process", "not linked to real markets", with "fixed and known volatility levels"
([Deriv Academy](https://traders-academy.deriv.com/trading-guides/guide-to-understanding-volatility-indices)).
That is a random walk, which is exactly what `--sim random` builds.

On a random walk, a trade with a 1R stop and a 20R target reaches the target first
**about 1 time in 21**, whatever the entry rule. I checked the grader on 16 simulated
runs (about 11,000 trades): 20R was hit 4.3% of the time, and the average result was
slightly negative, as it should be once costs are included.

So on V75 the question is whether your setups beat 1 in 21 by more than luck, over
hundreds of trades, with costs. Sections 3 and 7A answer exactly that. If the answer
on your real history looks like the random demo, the setups aren't adding anything on
that market, however convincing individual 20R winners look.

---

## The loss-review loop (`iterate.py`)

This is the "run it, look at the losses, change the code" loop, with the guardrails
built in. Removing the losses you're looking at is always possible: add a rule that
happens to exclude them. What matters is whether the rule also works on trades you
haven't looked at, so every change is scored on those.

```
python iterate.py start --data "data/V75_M15.csv"        once
python iterate.py check "skip setups that sweep into an H4 order block"
python iterate.py log
python iterate.py restore 3
```

Your history is split three ways:

| Part | Share | What it's for |
|---|---|---|
| Development | first 60% | the losses you chart, study and design changes from |
| Validation | next 20% | never charted; every change is scored here |
| Lockbox | last 20% | not touched by the loop at all; `run_all.py --final`, once, at the very end |

**`start`** scores the current code and saves charts of 30 losing trades and 6 winners
from the development period in `experiments/000_baseline/losses/`, with
`loss_notes.csv` to write why each one lost. Look for a pattern ("most losses swept
into a D1 order block", "stops inside the Asian range"), then change the code
(`setups.py`, `detectors.py`, `rule_votes.py` or `config.py`) as a **general rule**,
never a patch for one specific trade.

**`check "what you changed"`** runs all the tests, re-scores everything and prints the
kept version next to your change. For each period it shows how many losses the change
removed, and **how many winners it removed**. A filter that removes 10 losses and one
20R winner has made things worse. The change is **KEPT** only if:
1. all tests pass (nothing uses future prices),
2. the development period doesn't get worse,
3. the validation period makes more R, and
4. random prices don't start to look profitable. Nothing can beat chance on a random
   walk, so a change that seems to is using the future or fitting noise.

Otherwise it says **REVERT** and why, and `python iterate.py restore <n>` puts back the
code of any attempt. Every attempt is snapshotted in `experiments/` and listed by
`python iterate.py log`.

Two limits you can't code around:
- **Each check is another roll of the dice** against the same validation period. After
  about 10 attempts, some KEEPs are luck. Then get fresh data (a newer export, or
  months of demo trades from `live_mt5.py`) and `start --fresh` on it.
- **With 20R targets, one winner is worth 20 losses.** When a KEEP gains less than one
  winning trade, the loop says so, because that's within luck.

What it looks like on real gold (2012–2022, the checks I ran while building it):
- The starting code made -7.4R on development and +12.6R on validation.
- Changing `MIN_RISK_ATR` from 0.25 to 0.5 removed 3 losses and no winners on
  development. The development period still got worse (-15.2R), because with one trade
  at a time, skipping some trades changes which later trades get taken. Validation gained
  +1.7R, less than one winner. Under the current rules that's a REVERT.
- A planted cheat (skip setups where price hits the stop in the next 60 bars) was
  caught by the tests and reverted.

---

## Running it on MT5 (`live_mt5.py`)

The running version uses **the same code as the backtest**. The same Engine
(`smcml/live.py`) makes the decisions whether it's fed live MT5 bars or history, so what
you test is what trades.

**1. Paper replay first (no MT5 needed):**
```
python live_mt5.py --replay "data/V75_M15.csv"
```
This trades your whole history bar by bar through the Engine and checks the result
against the backtest. On 2012–2022 gold, all 600 trades were identical. If it ever
prints a WARNING that they disagree, don't trade until that's fixed.

**2. Demo account.** On the Windows machine or VPS where MT5 is installed and logged in:
```
pip install MetaTrader5
python live_mt5.py --symbol "Volatility 75 Index" --timeframe M15
```
- **Symbol and timeframe:** use the exact symbol name from Market Watch. The timeframe
  is your entry chart (M5 or M15). D1, H4 and H1 are built from it, the same as in the
  backtest.
- **Each bar:** at every bar close it reads the last 30,000 closed bars, finds new
  setups and re-entries, applies the risk guard, and places limit orders with the stop
  and target attached. It cancels orders not filled in `MAX_BARS_WAIT_FILL` bars and
  closes trades held longer than `MAX_HOLD_BARS`.
- **Lot size** comes from the stop distance, so a stop-out loses `RISK_PER_TRADE_PCT`
  of the balance. If even the broker's smallest lot would lose more, the trade is
  skipped and the reason is logged.
- **Demo only:** it refuses a real account unless you add `--allow-real`. There is no
  switch to change the risk settings from the command line.

Everything goes to `live/<symbol>_<timeframe>/`:
- `trades.csv`: every order, placed, filled, closed, expired or skipped (and why).
  `R` is the live result and `backtest_R` is what the backtest says the same order did.
  If they keep disagreeing, live differs from the backtest (spread, slippage, broker
  rules), and that's the first thing to fix.
- `charts/`: a chart of every losing trade, for the loss review.
- `state.json`: what the bot remembers if it's restarted.

The MT5 connection code is tested against a pretend MT5 terminal (`tests/fake_mt5.py`),
not a real one. Watch the first few demo orders in MT5 by eye to confirm the lots,
stops and targets look right.

The alternative to this (export the model to ONNX and run it inside an MQL5 EA) means
re-writing every detector and feature in MQL5 exactly as here. Any small difference
means live trading doesn't match the backtest.

---

## Where MuZero / reinforcement learning fits

Not for "take or skip": that's a one-off yes/no decision, which this template's
model handles more cheaply. Where RL could help is **managing an open trade**: when
to move the stop to breakeven, take a partial, or hold for 20R. That's a chain of
decisions. Try it only once this pipeline shows a real edge, and judge it with the
same exam: same setups, same walk-forward, same placebo and lockbox.

---

## Settings you'll most likely touch (`config.py`)

| Setting | What it does |
|---|---|
| `DATA_CSV` | path to your MT5 export (or use `--data`) |
| `TARGET_R` | the target in R (default 25, from your journal) |
| `COST_MODE`, `COST_PRICE` | your real spread / commission |
| `BIAS_TIMEFRAMES`, `BIAS_SWING_N`, `BIAS_FILTER` | the D1 → H4 → H1 bias rule |
| `SWING_N`, `HTF_SWING_N` | swing size for structure on the entry chart |
| `MAX_BARS_SWEEP_TO_SHIFT`, `MAX_BARS_WAIT_FILL` | how patient the state machine is |
| `MAX_HOLD_BARS` | time limit per trade (default 2,000 M15 bars, about 3 weeks) |
| `DYNAMIC_TARGET`, `DD_SWITCH_PCT`, `TARGET_R_IN_DRAWDOWN` | 12R targets while more than 20% below the peak (journal) |
| `BREAKEVEN_AT_R` | move the stop to the entry once a trade is this many R in profit (journal: 7; `None` = never) |
| `REENTRY`, `REENTRY_MAX_SHOTS`, `REENTRY_WINDOW_BARS`, `REENTRY_ENTRY` | more shots after a stop-out, if structure breaks your way within 96 bars (1 day on M15); up to 3 trades per setup (journal); entry at the order block or at the break |
| `COOLDOWN`, `COOLDOWN_TRIGGER`, `COOLDOWN_LOSING_DAYS`, `COOLDOWN_WINDOW_DAYS`, `COOLDOWN_AFTER_STOPS`, `COOLDOWN_DAYS`, `COOLDOWN_ENDS_ON_BREAKOUT` | the range lock (journal: 3 losing days in 10 = 15-day lock, or until the range breaks) |
| `LOSS_BLOCK_COUNT`, `LOSS_BLOCK_WINDOW_HOURS`, `LOSS_BLOCK_HOURS` | journal: 3 losses within 24h = no orders for 24h (0 = off) |
| `MAX_ORDERS_PER_24H` | journal: at most 5 orders in any 24 hours (0 = off) |
| `MODEL` | `"gbm"` (start here) or `"cnn"` (needs PyTorch, takes a few minutes) |
| `RISK_PER_TRADE_PCT`, `MAX_DAILY_LOSS_PCT`, `MAX_OPEN_TRADES` | the risk guard |
| `LOCKBOX_FRACTION` | share of history kept for the one-time final exam |
| `DEV_FRACTION` | share of history the loss-review loop lets you study (the next 20% scores your changes) |

---

## Glossary

- **R**: one unit of risk. Hitting the stop = -1R; a 20R winner = +20 times what you risked.
- **ATR**: average bar size. Sizes are measured in ATRs so features mean the same thing in quiet and wild markets.
- **Lookahead**: code that accidentally uses information from after the decision. It makes backtests look great and live trading lose.
- **Walk-forward**: train on the past, test on the next period, move forward, repeat.
- **Out-of-sample**: results on data the model never saw while being built.
- **Placebo**: the same model trained on shuffled results. A yardstick for "learned nothing".
- **Lockbox**: the last 20% of history, kept unseen for one final test.
- **p-value**: the chance that luck alone would produce a result at least this good. Below 0.01 is strong; below 0.05 is moderate.
- **MFE**: how far price went in your favour before the stop (in R).
- **Filled**: the limit order was actually reached. Unfilled orders count as 0R.

## Files

```
config.py              every setting
run_all.py             runs everything, writes the report
iterate.py             the loss-review loop (charts losses, scores changes, keeps a log)
live_mt5.py            runs the bot on MT5 (demo), or paper-replays your history
get_sample_data.py     downloads free real gold / FX M15 history to try it on
example_reports/       my reports on that real data (made with the earlier 20R settings)
smcml/data.py          MT5 CSV loader + V75-style simulator (random / planted)
smcml/detectors.py     swings, structure, order blocks, FVGs (no lookahead)
smcml/bias.py          D1 / H4 / H1 bias from closed candles only
smcml/setups.py        state machine + features
smcml/labels.py        outcome of every setup (fill, stop/target, costs)
smcml/rule_votes.py    your rules as votes + rule report
smcml/chart_images.py  chart pictures for the CNN
smcml/model.py         gradient-boosted trees, CNN with focal loss
smcml/walkforward.py   walk-forward exam, placebo, ranking test
smcml/my_calls.py      blind review of your own calls
smcml/plot_trades.py   pictures of trades, to check the setups look right
smcml/reentry.py       the one re-entry after a stop-out
smcml/risk.py          risk guard, account simulation, losing-streak maths
smcml/live.py          the running version's Engine, paper broker, and live/replay signals
tests/                 no-lookahead tests, must-always-be-true checks, replay = backtest,
                       and the MT5 code against a pretend terminal (fake_mt5.py)
data/                  put your MT5 exports here
```
