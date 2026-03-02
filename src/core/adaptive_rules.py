from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class BullWeightPlan:
    trend: float
    meanrev: float
    defensive: float


def choose_bull_weights(vol_regime: str, cfg: dict) -> BullWeightPlan:
    """
    Decide bull allocator weights based on vol regime.

    Config:
      cfg["adaptive"]["bull_trend_weights"] = {
        "high_vol": 0.60,
        "normal_vol": 0.70,
        "low_vol": 0.75
      }
      meanrev_weight comes from cfg["allocator"]["bull"]["meanrev"] (default 0.0)
      defensive is computed as 1 - trend - meanrev (clipped to >= 0)
    """
    a = (cfg.get("adaptive", {}) or {})
    tw = (a.get("bull_trend_weights", {}) or {})
    w_high = float(tw.get("high_vol", 0.60))
    w_norm = float(tw.get("normal_vol", 0.70))
    w_low = float(tw.get("low_vol", 0.75))

    alloc = (cfg.get("allocator", {}) or {})
    bull_alloc = (alloc.get("bull", {}) or {})
    w_mr = float(bull_alloc.get("meanrev", 0.0))

    if vol_regime == "HIGH":
        w_tr = w_high
    elif vol_regime == "LOW":
        w_tr = w_low
    else:
        w_tr = w_norm

    w_df = 1.0 - w_tr - w_mr
    if w_df < 0:
        # if user set meanrev too large, clamp defensive to 0
        w_df = 0.0

    return BullWeightPlan(trend=w_tr, meanrev=w_mr, defensive=w_df)


def soxx_allowed(
    mom_row: pd.Series,
    cfg: dict,
    vol_regime: str,
) -> bool:
    """
    Simple rule:
      allow SOXX only if mom(SOXX) >= mom(compare_to) + require_outperformance

    Optional additional block:
      cfg["adaptive"]["soxx"]["block_if_high_vol"] (default True)

    If required data missing, returns True (do not block by accident).
    """
    a = (cfg.get("adaptive", {}) or {})
    scfg = (a.get("soxx", {}) or {})
    if not bool(scfg.get("enabled", True)):
        return True

    if bool(scfg.get("block_if_high_vol", True)) and vol_regime == "HIGH":
        return False

    cmp_ticker = (scfg.get("allow_if", {}) or {}).get("compare_to", "QQQ")
    req = float((scfg.get("allow_if", {}) or {}).get("require_outperformance", 0.0))

    if "SOXX" not in mom_row.index:
        return True
    if cmp_ticker not in mom_row.index:
        return True

    soxx_m = mom_row.get("SOXX")
    cmp_m = mom_row.get(cmp_ticker)

    if pd.isna(soxx_m) or pd.isna(cmp_m):
        return True

    return (soxx_m - cmp_m) >= req


def filter_trend_picks(
    picks: List[str],
    mom_row: pd.Series,
    cfg: dict,
    vol_regime: str,
) -> List[str]:
    """
    Apply SOXX conditional allowance:
      - If SOXX is picked but not allowed, drop it and refill with next best.

    Keeps list length as much as possible.
    """
    if not picks:
        return picks

    if "SOXX" not in picks:
        return picks

    if soxx_allowed(mom_row, cfg, vol_regime):
        return picks

    banned = {"SOXX"}
    remaining = [p for p in picks if p not in banned]

    candidates = (cfg.get("trend_engine", {}) or {}).get("candidates", ["QQQ", "SPY", "SOXX"])
    ranked = (
        mom_row.reindex(candidates)
        .dropna()
        .sort_values(ascending=False)
        .index.tolist()
    )
    for t in ranked:
        if t in remaining or t in banned:
            continue
        remaining.append(t)
        if len(remaining) >= len(picks):
            break

    return remaining