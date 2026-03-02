from __future__ import annotations

import numpy as np
import pandas as pd


def _annualized_vol(daily_rets: pd.Series, lookback_days: int) -> pd.Series:
    """
    Annualized realized volatility from daily returns.
    Uses rolling std with ddof=0, then * sqrt(252).
    """
    vol = daily_rets.rolling(lookback_days, min_periods=lookback_days).std(ddof=0)
    return vol * np.sqrt(252.0)


def _compute_vol_regime(
    prices: pd.DataFrame,
    cfg: dict,
) -> pd.Series:
    """
    Returns a Series of {"LOW","NORMAL","HIGH"} indexed by date.
    Look-ahead safe: uses shift(1) so today's regime uses data up to yesterday.

    Config (cfg["adaptive"]["vol"]):
      lookback_days: int (default 20)
      regime_window_days: int rolling window for quantiles (default 756 ~ 3y)
      high_q: float (default 0.80)
      low_q: float (default 0.20)
      base_ticker: optional override (default cfg["state"]["base_ticker"])
    """
    a = (cfg.get("adaptive", {}) or {})
    vcfg = (a.get("vol", {}) or {})

    base_ticker = vcfg.get("base_ticker") or (cfg.get("state", {}) or {}).get("base_ticker", "SPY")
    lookback = int(vcfg.get("lookback_days", 20))
    qwin = int(vcfg.get("regime_window_days", 756))
    high_q = float(vcfg.get("high_q", 0.80))
    low_q = float(vcfg.get("low_q", 0.20))

    if base_ticker not in prices.columns:
        # Fallback: if missing, treat all as NORMAL
        return pd.Series(["NORMAL"] * len(prices.index), index=prices.index)

    rets = prices[base_ticker].pct_change().fillna(0.0)
    vol = _annualized_vol(rets, lookback)

    # rolling quantile thresholds
    hi = vol.rolling(qwin, min_periods=qwin).quantile(high_q)
    lo = vol.rolling(qwin, min_periods=qwin).quantile(low_q)

    # look-ahead prevention: decide using yesterday's available values
    vol_s = vol.shift(1)
    hi_s = hi.shift(1)
    lo_s = lo.shift(1)

    regime = pd.Series(["NORMAL"] * len(prices.index), index=prices.index, dtype="object")
    regime[(vol_s.notna()) & (hi_s.notna()) & (vol_s >= hi_s)] = "HIGH"
    regime[(vol_s.notna()) & (lo_s.notna()) & (vol_s <= lo_s)] = "LOW"

    return regime


def compute_state_flags_adaptive(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Extended version of src.core.state.compute_state_flags.

    Returns DataFrame with columns:
      bull_flag (bool), bear_flag (bool), crash_flag (bool), state (str), vol_regime (str)

    Priority: CRASH > BULL > BEAR
    Look-ahead prevention: signals use shift(1)
    """
    debug = bool(cfg.get("debug", {}).get("state", False))

    base_ticker = (cfg.get("state", {}) or {}).get("base_ticker", "SPY")
    ma_days = int((cfg.get("state", {}) or {}).get("ma_days", 200))
    min_hold = int((cfg.get("state", {}) or {}).get("min_hold_days", 0))

    crash_cfg = cfg.get("crash", {}) or {}
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash_lb = int(crash_cfg.get("lookback_days", 20))
    crash_th = float(crash_cfg.get("threshold", -0.15))

    if base_ticker not in prices.columns:
        raise KeyError(f"base_ticker '{base_ticker}' not found in prices columns.")

    p = prices[base_ticker].astype(float)

    # --- bull flag: price > MA ---
    ma = p.rolling(ma_days, min_periods=ma_days).mean()
    bull_raw = p > ma

    # look-ahead prevention
    bull_flag = bull_raw.shift(1).fillna(False)

    # --- crash flag: max drawdown over lookback <= threshold ---
    if crash_enabled:
        roll_max = p.rolling(crash_lb, min_periods=crash_lb).max()
        dd = p / roll_max - 1.0
        crash_raw = dd <= crash_th
        crash_flag = crash_raw.shift(1).fillna(False)
    else:
        crash_flag = pd.Series([False] * len(p), index=p.index)

    bear_flag = (~bull_flag) & (~crash_flag)

    # --- min_hold_days: smooth state changes ---
    # We enforce that once state changes, it must persist for min_hold days.
    # This is applied on the final state series (post lookahead shift).
    state = pd.Series(["BEAR"] * len(p), index=p.index, dtype="object")
    state[bull_flag] = "BULL"
    state[crash_flag] = "CRASH"  # priority

    if min_hold > 0:
        # convert to runs and enforce holding
        st = state.values.tolist()
        last = st[0]
        hold = 0
        for i in range(1, len(st)):
            if st[i] == last:
                hold += 1
                continue
            # state changed
            if hold < min_hold:
                st[i] = last  # revert change
                hold += 1
            else:
                last = st[i]
                hold = 0
        state = pd.Series(st, index=state.index, dtype="object")

        bull_flag = (state == "BULL")
        crash_flag = (state == "CRASH")
        bear_flag = (state == "BEAR")

    vol_regime = _compute_vol_regime(prices, cfg)

    out = pd.DataFrame(
        {
            "bull_flag": bull_flag.astype(bool),
            "bear_flag": bear_flag.astype(bool),
            "crash_flag": crash_flag.astype(bool),
            "state": state.astype(str),
            "vol_regime": vol_regime.astype(str),
        },
        index=prices.index,
    )

    if debug:
        # keep debug minimal; user can toggle in config
        vc = out["vol_regime"].value_counts(dropna=False).to_dict()
        sc = out["state"].value_counts(dropna=False).to_dict()
        print("[STATE-DEBUG] state_counts:", sc)
        print("[STATE-DEBUG] vol_regime_counts:", vc)

    return out