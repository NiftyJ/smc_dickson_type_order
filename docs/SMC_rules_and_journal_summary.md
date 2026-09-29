# SMC Rules and Journal Summary

Items marked **(open)** still need one exact answer before the bot can be coded.

## 1. SMC rules (as you trade them)

### Top-down bias
- Longs only when **D1, H4 and H1 are all bullish**; shorts only when all three are bearish.
- Bullish = the last break of structure was a close above a swing high.
- **(open)** How many candles on each side make a swing on each timeframe?

### Where to trade (point of interest)
- Only at the "correct areas": order blocks, ideally a **higher-timeframe (H4/D1) order block**.
- "They come back to the same order blocks."
- A high-timeframe POI is where you go for the **home run (50R+)**.
- **(open)** How exactly you mark an order block (last opposite candle? body or wicks?) and whether a fair value gap is required.

### Setup sequence
1. **Liquidity sweep:** price takes out a swing low (for longs).
2. **Break of structure** in your direction on a lower timeframe (5 min or 1 min): "mark peak high and low and 5/1 min BOS before a shot".
3. **Entry** at the order block created by that move.

### Stop and target
- Stop beyond the sweep. "Your stops by default are okay."
- Target: **25R when the account is in profit, 12R (about half) in drawdown**.
- **Don't move the stop**, except to **break-even once the trade is +7R**.
- Don't move take-profits in drawdown: "stick to 12R".

### Shots
- **Max 3 tries per setup** (sometimes written as 2 per BOS), each after a fresh break of structure in the same direction.
- If it fails 2–3 times, it's a range: leave it. "There's always one trade you can't catch because of range."

## 2. What your journal says from practice

### Ranges are the main killer
- A range can last **12–20 days**. "2 killers: range and 15 day range."
- Lock rule: **3 losing days within 10 days on a symbol = 15-day lock**, switch symbol, or wait for the range top or bottom to break.
- "You need to wait it out. You'll lose money either way if you force it."

### Frequency and account protection
- Trade at **intervals of a day or 2–3 days**, not every 6–12 hours. "Frequent trading is always a bad sign."
- **Block orders for 24 hours after 3+ losses.** No more than **5 trades** or **4 losses** in 24 hours.
- **48-hour lock after 3 consecutive losing days.** **2 days off after a 5R loss.**
- "Please don't tamper with the EA, it's right."

### Markets and news
- You need **fast markets** and want to avoid slow ones. **(open)** How do you tell them apart?
- News can flip direction, but the **5-minute break of structure shows it**. Trust price.

### Why manual trading failed (your own words)
- **Entry quality:** "take orders only directly at correct areas, which is unlikely for me to do manually."
- **Stress:** "5 day losing streak makes me want to quit", headaches after many losses.
- **Impatience:** "you're impatient"; "for patience you need to trade like in 2–3 days intervals."
- **Time and over-confidence:** "When you get excited and think you figured it, you haven't."
- **Conclusion:** "Automation is the way." "Impossible to trade full time; mix it with a job."

## 3. Contradictions in the journal (pick one each)

| Topic | Versions in the journal | Your choice |
|---|---|---|
| Drawdown switch | 15%, 20% or 30% below peak | |
| Target in drawdown | 10R, 12R or half of the home run | |
| Shots | 2 per BOS, or max 3 | |
| Loss block | 24h after 3 losses; 8h and max 4 losses in 24h; 48h after 3 losing days | |
| Risk per trade | "4% max" (advised against: at 1:20, a 20-loss streak at 4% = -56%) | |

## 4. Open questions to answer for the bot

1. Swing size (candles each side) on D1, H4, H1 and the entry timeframe.
2. How an order block is marked, and which one counts as a high-timeframe POI.
3. Which low must be swept, and by how much.
4. BOS timeframe (1m / 5m / 15m), close or wick, and how soon after the sweep.
5. Entry type: limit at the order block, at the fair value gap, or market on the break. How long before an unfilled order is cancelled.
6. Stop placement and buffer.
7. What counts as "the same setup" for a retry.
8. How to spot a slow market or a range before entering.
9. 2–3 example screenshots with entry, stop and target marked.
