from __future__ import annotations
import pandas as pd
import numpy as np


def compute_state_flags(prices, cfg):
    """
    Returns DataFrame with columns:
      bull_flag (bool), bear_flag (bool), crash_flag (bool), state (str)
    state priority: CRASH > BEAR > BULL
    All signals are shifted(1) to avoid look-ahead.
    """
    base = cfg["state"]["base_ticker"]
    if base not in prices.columns:
        raise ValueError(f"state.base_ticker {base} not in prices")

    p = prices[base]
    ma_days = int(cfg["state"]["ma_days"])
    min_hold = int(cfg["state"].get("min_hold_days", 0))

    ma = p.rolling(ma_days).mean()
    bull = (p > ma).astype(bool).shift(1).fillna(False)
    bear = (~bull).astype(bool)

    crash_cfg = cfg.get("crash", {})
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash = pd.Series(False, index=prices.index)

    if crash_enabled:
        lb = int(crash_cfg["lookback_days"])
        thr = float(crash_cfg["threshold"])
        r = p.pct_change(lb).shift(1)
        crash = (r <= thr).fillna(False)

    # hysteresis on bull/bear: require min_hold days before switching
    if min_hold > 0:
        bull = _min_hold_filter(bull, min_hold)

    state = pd.Series("BULL", index=prices.index)
    state[bear] = "BEAR"
    state[crash] = "CRASH"

    out = pd.DataFrame({
        "bull_flag": bull.astype(bool),
        "bear_flag": bear.astype(bool),
        "crash_flag": crash.astype(bool),
        "state": state
    }, index=prices.index)
    return out


def _min_hold_filter(bull_flag: pd.Series, min_hold_days: int) -> pd.Series:
    """
    Prevent rapid flipping: once state changes, keep it for min_hold_days.
    Simple run-length enforcement.
    """
    bull_flag = bull_flag.astype(bool).copy()
    vals = bull_flag.values
    out = vals.copy()

    last = out[0]
    hold = 0
    for i in range(1, len(out)):
        if out[i] == last:
            hold = 0
        else:
            hold += 1
            if hold <= min_hold_days:
                out[i] = last
            else:
                last = out[i]
                hold = 0

    return pd.Series(out, index=bull_flag.index)