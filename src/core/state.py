from __future__ import annotations
import pandas as pd
import numpy as np


def compute_state_flags(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      bull_flag (bool), bear_flag (bool), crash_flag (bool), state (str)

    Priority: CRASH > BULL > BEAR
    Look-ahead prevention: signals use shift(1)

    Additions:
      - FAST_CRASH: optional short lookback crash trigger (e.g. 5d <= -8%)
      - hold_mode:
          * "both": (legacy) min_hold_days applies to BOTH exit/reentry (run-length enforcement)
          * "reentry": delay ONLY BULL re-entry (exit remains fast)
    """
    debug = bool(cfg.get("debug", {}).get("state", False))

    base = cfg["state"]["base_ticker"]
    if base not in prices.columns:
        raise ValueError(f"state.base_ticker '{base}' not in prices columns: {list(prices.columns)}")

    p = prices[base].astype(float)

    ma_days = int(cfg["state"]["ma_days"])
    min_hold = int(cfg["state"].get("min_hold_days", 0))

    # new: hold_mode + reentry_hold_days
    hold_mode = str(cfg["state"].get("hold_mode", "both")).lower().strip()  # "both" | "reentry"
    reentry_hold = int(cfg["state"].get("reentry_hold_days", min_hold if hold_mode == "reentry" else 0))

    ma = p.rolling(ma_days).mean()

    bull_raw = (p > ma)                        # raw (same-day)
    bull = bull_raw.shift(1).fillna(False)     # look-ahead safe

    crash_cfg = cfg.get("crash", {}) or {}
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash = pd.Series(False, index=prices.index)

    # main crash (legacy)
    if crash_enabled:
        lb = int(crash_cfg["lookback_days"])
        thr = float(crash_cfg["threshold"])
        r = p.pct_change(lb).shift(1)          # look-ahead safe
        crash = (r <= thr).fillna(False)

    # fast crash (new, optional)
    fast = (crash_cfg.get("fast", {}) or {})
    fast_enabled = bool(fast.get("enabled", False))
    fast_crash = pd.Series(False, index=prices.index)
    if crash_enabled and fast_enabled:
        flb = int(fast.get("lookback_days", 5))
        fthr = float(fast.get("threshold", -0.08))
        fr = p.pct_change(flb).shift(1)
        fast_crash = (fr <= fthr).fillna(False)
        crash = (crash | fast_crash)

    # ---- DEBUG BEFORE HOLD FILTER ----
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
        print(f"[STATE-DEBUG] hold_mode={hold_mode} min_hold_days={min_hold} reentry_hold_days={reentry_hold}")
        if crash_enabled:
            print(f"[STATE-DEBUG] crash_main: lb={crash_cfg.get('lookback_days')} thr={crash_cfg.get('threshold')}")
            if fast_enabled:
                print(f"[STATE-DEBUG] crash_fast: lb={fast.get('lookback_days')} thr={fast.get('threshold')}")
        if p_max < 10.0:
            print("[STATE-DEBUG] WARNING: price scale looks like a normalized index (close-to-1 style).")
            print("[STATE-DEBUG]          If this is unintended, you're NOT using real SPY prices.")
        print("[STATE-DEBUG] ===============================")

    # ---- HOLD FILTER ----
    if hold_mode == "both":
        # legacy: run-length enforcement on bull flag (slows BOTH exit and reentry)
        if min_hold > 0:
            bull_before = bull.copy()
            bull = _min_hold_filter(bull, min_hold)

            if debug:
                b1 = int(bull_before.sum())
                b2 = int(bull.sum())
                r1 = float(b1) / float(len(bull_before)) if len(bull_before) else 0.0
                r2 = float(b2) / float(len(bull)) if len(bull) else 0.0
                first_after = bull[bull].index.min()
                print(f"[STATE-DEBUG] hysteresis(min_hold BOTH) min_hold_days={min_hold}")
                print(f"[STATE-DEBUG] bull_before_true={b1} ({r1*100:.2f}%) -> bull_after_true={b2} ({r2*100:.2f}%) first_after={first_after}")

    elif hold_mode == "reentry":
        # new: delay ONLY BULL re-entry (exit is fast)
        if reentry_hold > 0:
            bull_before = bull.copy()
            bull = _reentry_hold_filter(bull, reentry_hold)

            if debug:
                b1 = int(bull_before.sum())
                b2 = int(bull.sum())
                r1 = float(b1) / float(len(bull_before)) if len(bull_before) else 0.0
                r2 = float(b2) / float(len(bull)) if len(bull) else 0.0
                first_after = bull[bull].index.min()
                print(f"[STATE-DEBUG] hysteresis(reentry ONLY) reentry_hold_days={reentry_hold}")
                print(f"[STATE-DEBUG] bull_before_true={b1} ({r1*100:.2f}%) -> bull_after_true={b2} ({r2*100:.2f}%) first_after={first_after}")

    else:
        # unknown -> no hold filter
        pass

    # ---- STATE ASSIGNMENT ----
    state = pd.Series("BEAR", index=prices.index)
    state.loc[bull] = "BULL"
    state.loc[crash] = "CRASH"  # override

    bear = (state == "BEAR")
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
        mismatch = out.index[(out["bull_flag"] == True) & (out["state"] != "BULL")]
        if len(mismatch) > 0:
            print(f"[STATE-DEBUG] WARNING: bull_flag True but state != BULL (n={len(mismatch)}) first={mismatch.min()}")
        else:
            print("[STATE-DEBUG] OK: bull_flag aligns with state=BULL")

    return out


def _min_hold_filter(bull_flag: pd.Series, min_hold_days: int) -> pd.Series:
    """
    Legacy run-length enforcement:
      once flag changes, keep previous flag for min_hold_days.
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


def _reentry_hold_filter(bull_flag: pd.Series, reentry_hold_days: int) -> pd.Series:
    """
    Delay ONLY BULL re-entry:
      - If bull_flag tries to flip False->True, require it to stay True for (reentry_hold_days+1) days.
      - True->False happens immediately (no delay).
    """
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
            # exit is immediate
            if cur is False:
                last = False
                consec_true = 0
            else:
                last = True
                consec_true = 0
            out[i] = last
            continue

        # last is False: candidate for re-entry
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