from __future__ import annotations
import pandas as pd
import numpy as np


def compute_state_flags(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    debug = bool(cfg.get("debug", {}).get("state", False))

    base = cfg["state"]["base_ticker"]
    if base not in prices.columns:
        raise ValueError(f"state.base_ticker '{base}' not in prices columns: {list(prices.columns)}")
    p = prices[base].astype(float)

    ma_days = int(cfg["state"]["ma_days"])
    ma = p.rolling(ma_days).mean()

    bull_raw = (p > ma)
    bull = bull_raw.shift(1).fillna(False)

    # ---- CRASH (main + fast) ----
    crash_cfg = cfg.get("crash", {}) or {}
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash = pd.Series(False, index=prices.index)

    if crash_enabled:
        lb = int(crash_cfg["lookback_days"])
        thr = float(crash_cfg["threshold"])
        r = p.pct_change(lb).shift(1)
        crash = (r <= thr).fillna(False)

    fast = (crash_cfg.get("fast", {}) or {})
    if crash_enabled and bool(fast.get("enabled", False)):
        flb = int(fast.get("lookback_days", 5))
        fthr = float(fast.get("threshold", -0.08))
        fr = p.pct_change(flb).shift(1)
        crash = (crash | (fr <= fthr).fillna(False))

    # ---- Switches (1~3) ----
    sw = (cfg.get("switches", {}) or {})
    # (1) Momentum gate on SPY
    if bool(sw.get("mom_gate", False)):
        mom_lb = int(sw.get("mom_lookback_days", 63))
        mom_thr = float(sw.get("mom_threshold", 0.0))
        mom = p.pct_change(mom_lb).shift(1)
        bull = bull & (mom > mom_thr).fillna(False)

    # (3) Breadth gate (RSP/SPY)
    if bool(sw.get("breadth_gate", False)):
        rsp = str(sw.get("breadth_ticker", "RSP"))
        if rsp not in prices.columns:
            raise ValueError(f"breadth_gate enabled but '{rsp}' not in prices columns. Add to data.tickers.")
        lb = int(sw.get("breadth_lookback_days", 63))
        rel = (prices[rsp].astype(float) / p.astype(float)).pct_change(lb).shift(1)
        # if relative momentum negative -> suppress BULL
        bull = bull & (rel > 0.0).fillna(False)

    # base state assignment (no hold/cooldown yet)
    state = pd.Series("BEAR", index=prices.index)
    state.loc[bull] = "BULL"
    state.loc[crash] = "CRASH"

    # (2) Cooldown after exiting BEAR/CRASH: delay BULL re-entry N days
    if bool(sw.get("cooldown", False)):
        cd = int(sw.get("cooldown_days", 20))
        if cd > 0:
            # whenever state is not BULL, reset counter; when trying to go BULL, require counter>cd
            cooldown_ok = np.ones(len(state), dtype=bool)
            cool = 10**9  # start large so first bull allowed
            for i in range(len(state)):
                if state.iat[i] != "BULL":
                    cool = 0
                    cooldown_ok[i] = True
                else:
                    cool += 1
                    cooldown_ok[i] = (cool > cd)
            # suppress BULL where cooldown not satisfied (turn into BEAR, CRASH stays CRASH)
            suppress = (~pd.Series(cooldown_ok, index=state.index)) & (state == "BULL")
            state.loc[suppress] = "BEAR"

    out = pd.DataFrame(
        {
            "bull_flag": (state == "BULL"),
            "bear_flag": (state == "BEAR"),
            "crash_flag": (state == "CRASH"),
            "state": state,
        },
        index=prices.index,
    )

    if debug:
        print("[STATE-DEBUG] switches:", sw)
        print("[STATE-DEBUG] state_counts:", out["state"].value_counts(dropna=False).to_dict())

    return out