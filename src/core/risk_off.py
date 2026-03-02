def risk_off_weights(mode: str):
    # Cash-like / T-Bills
    if mode == "SHY_100":
        return {"SHY": 1.0}
    if mode == "BIL_100":
        return {"BIL_MIX": 1.0}
    if mode == "SGOV_100":
        return {"SGOV_MIX": 1.0}

    # Gold mixes
    if mode == "SHY_GLD_50_50":
        return {"SHY": 0.5, "GLD": 0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY": 0.7, "GLD": 0.3}
    if mode == "GLD_100":
        return {"GLD": 1.0}

    # Inverse equity (uses MIX = proxy pre-listing, real after listing)
    if mode == "SH_100":
        return {"SH_MIX": 1.0}   # -1x SPY
    if mode == "PSQ_100":
        return {"PSQ_MIX": 1.0}  # -1x QQQ

    return {}