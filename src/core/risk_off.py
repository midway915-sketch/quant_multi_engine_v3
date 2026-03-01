
def risk_off_weights(mode: str):
    if mode == "SHY_100":
        return {"SHY": 1.0}
    if mode == "SHY_GLD_50_50":
        return {"SHY": 0.5, "GLD": 0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY": 0.7, "GLD": 0.3}
    if mode == "GLD_100":
        return {"GLD": 1.0}
    return {}
