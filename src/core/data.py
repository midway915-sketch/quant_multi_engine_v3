import yfinance as yf
import pandas as pd


def _collect_tickers(cfg):
    tickers = list(cfg["data"]["tickers"])

    lev = cfg.get("leverage_etf", {})
    if isinstance(lev, dict):
        m = lev.get("map", {})
        if isinstance(m, dict):
            for _, v in m.items():
                if v and v not in tickers:
                    tickers.append(v)

    return tickers


def download_prices(cfg) -> pd.DataFrame:
    tickers = _collect_tickers(cfg)

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
            prices = df["Adj Close"]
        elif "Close" in df.columns.levels[0]:
            prices = df["Close"]
        else:
            prices = df.xs(df.columns.levels[0][0], level=0, axis=1)
    else:
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]]
        elif "Close" in df.columns:
            prices = df[["Close"]]
        else:
            raise ValueError("Neither Adj Close nor Close found in downloaded data.")

    prices = prices.dropna(how="all").dropna()
    if prices.empty:
        raise ValueError("Price data empty after cleaning.")

    return prices