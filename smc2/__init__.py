"""The new version of the bot, built one indicator at a time.

Every indicator here follows the same rules (smc2/core.py):
  * row t = what was known at the CLOSE of candle t (no future candles)
  * one row per candle of YOUR chart, even when the indicator runs on a higher timeframe
  * prices are real prices; directions are +1 up / -1 down / 0 none
  * down versions come from flipping the chart, so up and down can never drift apart

Indicators so far:
  range_type.py   pause / Wyckoff / staircase ranges, as one indicator
"""
