def map_to_leveraged(ticker: str, cfg, state: str) -> str:
    lev = cfg.get("leverage_etf", {})
    if not isinstance(lev, dict):
        return ticker

    if not bool(lev.get("enabled", False)):
        return ticker

    m = lev.get("map", {})
    if not isinstance(m, dict):
        return ticker

    if state == "STRONG" and bool(lev.get("use_3x_in_strong", True)):
        return m.get(ticker, ticker)

    if state == "WEAK" and bool(lev.get("use_3x_in_weak", False)):
        return m.get(ticker, ticker)

    return ticker