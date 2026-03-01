
import pandas as pd

def compute_regime(mkt_price: pd.Series, cfg) -> pd.Series:
    if not cfg.get("regime", {}).get("enabled", True):
        return pd.Series(["STRONG"] * len(mkt_price), index=mkt_price.index)

    ma_fast = mkt_price.rolling(int(cfg["regime"]["ma_fast"])).mean()
    ma_slow = mkt_price.rolling(int(cfg["regime"]["ma_slow"])).mean()

    # STRONG when above slow MA, else WEAK
    state = pd.Series("WEAK", index=mkt_price.index)
    state[mkt_price > ma_slow] = "STRONG"
    return state
