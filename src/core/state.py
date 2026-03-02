from __future__ import annotations
import pandas as pd
import numpy as np


def compute_state_flags(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      bull_flag (bool), bear_flag (bool), crash_flag (bool), state (str)

    state priority: CRASH > BEAR > BULL
    Look-ahead prevention: signals use shift(1)
    """
    debug = bool(cfg.get("debug", {}).get("state", False))

    base = cfg["state"]["base_ticker"]
    if base not in prices.columns:
        raise ValueError(f"state.base_ticker '{base}' not in prices columns: {list(prices.columns)}")

    p = prices[base].astype(float)

    ma_days = int(cfg["state"]["ma_days"])
    min_hold = int(cfg["state"].get("min_hold_days", 0))

    ma = p.rolling(ma_days).mean()

    bull_raw = (p > ma)                       # raw signal on same day (NOT used directly)
    bull = bull_raw.shift(1).fillna(False)    # look-ahead safe
    bear = (~bull).astype(bool)

    crash_cfg = cfg.get("crash", {})
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash = pd.Series(False, index=prices.index)

    if crash_enabled:
        lb = int(crash_cfg["lookback_days"])
        thr = float(crash_cfg["threshold"])
        r = p.pct_change(lb).shift(1)         # look-ahead safe
        crash = (r <= thr).fillna(False)

    # ---- DEBUG BEFORE HYSTERESIS ----
    if debug:
        idx0 = prices.index[0]
        idx1 = prices.index[-1]
        p_min, p_max = float(np.nanmin(p.values)), float(np.nanmax(p.values))
        ma_nan = int(ma.isna().sum())
        p_nan = int(p.isna().sum())
        bull_true = int(bull.sum())
        bull_ratio = float(bull_true) / float(len(bull)) if len(bull) else 0.0
        bullraw_true = int(bull_raw.fillna(False).sum())
        bullraw_ratio = float(bullraw_true) / float(len(bull_raw)) if len(bull_raw) else 0.0

        first_bullraw = bull_raw[bull_raw.fillna(False)].index.min()
        first_bull = bull[bull].index.min()

        print("[STATE-DEBUG] ===============================")
        print(f"[STATE-DEBUG] date_range: {idx0} -> {idx1}  (n={len(prices)})")
        print(f"[STATE-DEBUG] base_ticker: {base}")
        print(f"[STATE-DEBUG] p_nan={p_nan}, ma_nan={ma_nan}, ma_days={ma_days}")
        print(f"[STATE-DEBUG] p_min={p_min:.6f}, p_max={p_max:.6f}")
        print(f"[STATE-DEBUG] bull_raw_true={bullraw_true} ({bullraw_ratio*100:.2f}%) first={first_bullraw}")
        print(f"[STATE-DEBUG] bull_shift_true={bull_true} ({bull_ratio*100:.2f}%) first={first_bull}")
        if p_max < 10.0:
            print("[STATE-DEBUG] WARNING: price scale looks like a normalized index (close-to-1 style).")
            print("[STATE-DEBUG]          If this is unintended, you're NOT using real SPY prices.")
        print("[STATE-DEBUG] ===============================")

    # hysteresis (min hold)
    if min_hold > 0:
        bull_before = bull.copy()
        bull = _min_hold_filter(bull, min_hold)

        if debug:
            b1 = int(bull_before.sum())
            b2 = int(bull.sum())
            r1 = float(b1) / float(len(bull_before)) if len(bull_before) else 0.0
            r2 = float(b2) / float(len(bull)) if len(bull) else 0.0
            first_after = bull[bull].index.min()
            print(f"[STATE-DEBUG] hysteresis min_hold_days={min_hold}")
            print(f"[STATE-DEBUG] bull_before_true={b1} ({r1*100:.2f}%) -> bull_after_true={b2} ({r2*100:.2f}%) first_after={first_after}")

    bear = (~bull).astype(bool)

    state = pd.Series("BULL", index=prices.index)
    state[bear] = "BEAR"
    state[crash] = "CRASH"

    out = pd.DataFrame(
        {
            "bull_flag": bull.astype(bool),
            "bear_flag": bear.astype(bool),
            "crash_flag": crash.astype(bool),
            "state": state,
        },
        index=prices.index,
    )

    if debug:
        vc = out["state"].value_counts(dropna=False).to_dict()
        print(f"[STATE-DEBUG] state_counts: {vc}")

    return out


def _min_hold_filter(bull_flag: pd.Series, min_hold_days: int) -> pd.Series:
    """
    Run-length enforcement:
      once state changes, keep previous state for min_hold_days.
    """
    bull_flag = bull_flag.astype(bool).copy()

    vals = bull_flag.values
    if len(vals) == 0:
        return bull_flag

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