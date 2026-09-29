"""
IDEA FROM "SNORKEL" (Ratner et al., 2017): turn rules into VOTERS instead of if-branches.

The problem you described: your SMC rules overlap and contradict each other, so
hand-coding "which rule wins" creates endless bugs.

Snorkel's answer: let every rule cast a vote on every setup
    +1 = "looks good", -1 = "looks bad", 0 = "no opinion"
and let the DATA decide how much each rule is worth. Snorkel had to guess the
accuracy of each rule from how the rules agreed with each other. You are luckier:
the market grades every setup (labels.py), so we can measure each rule directly.

How to use this file:
  * ADD YOUR OWN RULES below. One small function per rule. They may overlap and
    contradict each other as much as they like; that is now allowed.
  * rule_report() tells you, for each rule, how often it fires and whether setups
    it likes really do better. Delete rules that do nothing.
  * The votes are also handed to the model as extra features, so the model can
    learn which combinations matter.
"""
import numpy as np
import pandas as pd


def _v(good, bad=None):
    """Helper: +1 where good, -1 where bad, 0 elsewhere."""
    out = np.where(good, 1, 0)
    if bad is not None:
        out = np.where(bad, -1, out)
    return out


# Each rule receives the candidates table and returns one vote per row.
RULES = {
    "htf_agrees":          lambda c, cfg: _v(c.htf_trend > 0, c.htf_trend < 0),
    "entry_in_discount":   lambda c, cfg: _v(c.pd_pos < 0.5, c.pd_pos >= 0.5),
    "is_choch":            lambda c, cfg: _v(c.is_choch == 1),
    "left_an_fvg":         lambda c, cfg: _v(c.n_fvg > 0, c.n_fvg == 0),
    "ob_inside_fvg":       lambda c, cfg: _v(c.ob_fvg_overlap == 1),
    "deep_sweep":          lambda c, cfg: _v(c.sweep_depth_atr >= 0.5),
    "strong_displacement": lambda c, cfg: _v(c.displacement_atr >= 1.5, c.displacement_atr < 0.8),
    "killzone":            lambda c, cfg: _v(((c.hour >= 7) & (c.hour < 11)) | ((c.hour >= 12) & (c.hour < 16))),
    "room_to_run":         lambda c, cfg: _v(c.room_R >= cfg.TARGET_R, c.room_R < cfg.TARGET_R / 2),
    "cheap_to_trade":      lambda c, cfg: _v(c.cost_R <= 0.1, c.cost_R > 0.25),
}


def votes(cands, cfg):
    """Table of votes: one column per rule, named vote_<rule>."""
    return pd.DataFrame({f"vote_{name}": fn(cands, cfg).astype(int) for name, fn in RULES.items()},
                        index=cands.index)


def rule_report(labeled, cfg):
    """How each rule performs on the (early part of the) history.

    coverage        = share of setups where the rule has an opinion
    win_if_yes      = TARGET_R hit rate among filled setups the rule votes +1 on
    win_if_no       = same, among setups it votes -1 on
    avgR_if_yes/no  = average result in R, same split (unfilled orders count as 0R)
    """
    v = votes(labeled, cfg)
    done = labeled["complete"].to_numpy()
    filled = (labeled["filled"] == 1).to_numpy()
    y = labeled["y"].to_numpy()
    r = labeled["outcome_R"].to_numpy()
    base_win = np.nanmean(y[done & filled]) if (done & filled).any() else np.nan
    rows = []
    for col in v.columns:
        x = v[col].to_numpy()
        yes, no = done & (x == 1), done & (x == -1)
        rows.append({
            "rule": col.replace("vote_", ""),
            "coverage": np.mean(x[done] != 0),
            "n_yes": int(yes.sum()), "n_no": int(no.sum()),
            "win_if_yes": np.nanmean(y[yes & filled]) if (yes & filled).any() else np.nan,
            "win_if_no": np.nanmean(y[no & filled]) if (no & filled).any() else np.nan,
            "avgR_if_yes": np.nanmean(r[yes]) if yes.any() else np.nan,
            "avgR_if_no": np.nanmean(r[no]) if no.any() else np.nan,
        })
    rep = pd.DataFrame(rows)
    rep.attrs["base_win_rate"] = base_win

    # How often do rules contradict each other? (this is what made hand-coding so buggy)
    vv = v.to_numpy()[done]
    rep.attrs["share_with_conflict"] = float(np.mean((vv == 1).any(1) & (vv == -1).any(1))) if len(vv) else np.nan
    return rep
