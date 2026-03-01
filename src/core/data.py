from __future__ import annotations
import yfinance as yf
import pandas as pd


def download_prices_and_build_proxies(cfg) -> pd.DataFrame:
    tickers = list(cfg["data"]["tickers"])

    df = yf.download(
        tickers,
        start=cfg["data"]["start"],
        auto_adjust=False,
        progress=False,
        group_by="column"
    )
    if df.empty:
        raise ValueError("Downloaded price data is empty.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.levels[0]:
            px = df["Adj Close"]
        elif "Close" in df.columns.levels[0]:
            px = df["Close"]
        else:
            px = df.xs(df.columns.levels[0][0], level=0, axis=1)
    else:
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]]
        elif "Close" in df.columns:
            px = df[["Close"]]
        else:
            raise ValueError("Neither Adj Close nor Close found.")

    px = px.dropna(how="all").sort_index().ffill()

    # Build synthetic leveraged proxies for execution
    # (research mode) daily returns * k -> cumprod
    rets = px.pct_change().fillna(0.0)

    def make_proxy(base: str, name: str, k: float):
        if base not in px.columns:
            return
        proxy = (1.0 + (rets[base] * k)).cumprod()
        px[name] = proxy

    make_proxy("QQQ", "TQQQ_PROXY", 3.0)
    make_proxy("SPY", "UPRO_PROXY", 3.0)
    make_proxy("SOXX", "SOXL_PROXY", 3.0)

    # 1.5x proxy for mean reversion (SPY 기반)
    make_proxy("SPY", "SPY_1P5_PROXY", 1.5)

    # Align: require base tickers exist
    base_cols = [c for c in cfg["data"]["tickers"] if c in px.columns]
    px = px.dropna(subset=base_cols, how="any")
    if px.empty:
        raise ValueError("Price data empty after alignment.")

    return px