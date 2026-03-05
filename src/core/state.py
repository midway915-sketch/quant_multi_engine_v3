from __future__ import annotations
import pandas as pd
import numpy as np


def compute_state_flags(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Priority: CRASH > BULL > BEAR
    Look-ahead prevention: signals use shift(1)

    Additions:
      - crash.fast: optional short-lookback crash trigger
      - state.hold_mode: "both" or "reentry"
      - state.reentry_hold_days: delay only BULL re-entry
      - bear_fast: optional short-lookback BEAR trigger (suppresses BULL without declaring CRASH)
    """
    debug = bool(cfg.get("debug", {}).get("state", False))

    base = cfg["state"]["base_ticker"]
    if base not in prices.columns:
        raise ValueError(f"state.base_ticker '{base}' not in prices columns: {list(prices.columns)}")

    p = prices[base].astype(float)

    ma_days = int(cfg["state"]["ma_days"])
    min_hold = int(cfg["state"].get("min_hold_days", 0))

    hold_mode = str(cfg["state"].get("hold_mode", "both")).lower().strip()  # both|reentry
    reentry_hold = int(cfg["state"].get("reentry_hold_days", min_hold if hold_mode == "reentry" else 0))

    ma = p.rolling(ma_days).mean()

    bull_raw = (p > ma)
    bull = bull_raw.shift(1).fillna(False)

    # ---- CRASH ----
    crash_cfg = cfg.get("crash", {}) or {}
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash = pd.Series(False, index=prices.index)

    if crash_enabled:
        lb = int(crash_cfg["lookback_days"])
        thr = float(crash_cfg["threshold"])
        r = p.pct_change(lb).shift(1)
        crash = (r <= thr).fillna(False)

    fast = (crash_cfg.get("fast", {}) or {})
    fast_enabled = bool(fast.get("enabled", False))
    if crash_enabled and fast_enabled:
        flb = int(fast.get("lookback_days", 5))
        fthr = float(fast.get("threshold", -0.08))
        fr = p.pct_change(flb).shift(1)
        crash = (crash | (fr <= fthr).fillna(False))

    # ---- BEAR FAST (new) ----
    # If triggered, it suppresses BULL (i.e., forces BEAR unless CRASH overrides)
    bear_fast_cfg = (cfg.get("bear_fast", {}) or {})
    bear_fast_enabled = bool(bear_fast_cfg.get("enabled", False))
    bear_fast = pd.Series(False, index=prices.index)
    if bear_fast_enabled:
        blb = int(bear_fast_cfg.get("lookback_days", 10))
        bthr = float(bear_fast_cfg.get("threshold", -0.06))
        br = p.pct_change(blb).shift(1)
        bear_fast = (br <= bthr).fillna(False)

        # 핵심: bear_fast가 True면 BULL을 억제
        bull = bull & (~bear_fast)

    # ---- HOLD FILTER ----
    if hold_mode == "both":
        if min_hold > 0:
            bull = _min_hold_filter(bull, min_hold)
    elif hold_mode == "reentry":
        if reentry_hold > 0:
            bull = _reentry_hold_filter(bull, reentry_hold)

    # ---- STATE ASSIGNMENT ----
    state = pd.Series("BEAR", index=prices.index)
    state.loc[bull] = "BULL"
    state.loc[crash] = "CRASH"

    out = pd.DataFrame(
        {
            "bull_flag": bull.astype(bool),
            "bear_flag": (state == "BEAR"),
            "crash_flag": crash.astype(bool),
            "state": state,
        },
        index=prices.index,
    )

    if debug:
        vc = out["state"].value_counts(dropna=False).to_dict()
        print(f"[STATE-DEBUG] state_counts: {vc}")
        if bear_fast_enabled:
            print(f"[STATE-DEBUG] bear_fast_true={int(bear_fast.sum())}")

    return out


def _min_hold_filter(bull_flag: pd.Series, min_hold_days: int) -> pd.Series:
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


def _reentry_hold_filter(bull_flag: pd.Series, reentry_hold_days: int) -> pd.Series:
    bull_flag = bull_flag.astype(bool).copy()
    vals = bull_flag.values
    if len(vals) == 0:
        return bull_flag

    out = vals.copy()
    last = out[0]
    consec_true = 0

    for i in range(1, len(out)):
        cur = out[i]

        if last is True:
            # exit immediate
            if cur is False:
                last = False
                consec_true = 0
            out[i] = last
            continue

        # last False -> reentry requires consecutive True
        if cur is True:
            consec_true += 1
            if consec_true <= reentry_hold_days:
                out[i] = False
                last = False
            else:
                out[i] = True
                last = True
                consec_true = 0
        else:
            consec_true = 0
            out[i] = False
            last = False

    return pd.Series(out, index=bull_flag.index)