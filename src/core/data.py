from __future__ import annotations
import yfinance as yf
import pandas as pd


def download_prices_and_build_proxies(cfg) -> pd.DataFrame:
    base_tickers = list(cfg["data"]["tickers"])

    # 실제 3배 ETF도 같이 받음
    real_leveraged = ["TQQQ", "UPRO", "SOXL"]

    tickers = list(set(base_tickers + real_leveraged))

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

    rets = px.pct_change().fillna(0.0)

    # --- Proxy 생성 ---
    def make_proxy(base: str, name: str, k: float):
        if base not in px.columns:
            return
        proxy = (1.0 + (rets[base] * k)).cumprod()
        px[name] = proxy

    make_proxy("QQQ", "TQQQ_PROXY", 3.0)
    make_proxy("SPY", "UPRO_PROXY", 3.0)
    make_proxy("SOXX", "SOXL_PROXY", 3.0)

    # --- Hybrid MIX 생성 ---
    def make_mix(real: str, proxy: str, mix_name: str):
        if proxy not in px.columns:
            return
        mix = px[proxy].copy()
        if real in px.columns:
            real_start = px[real].first_valid_index()
            if real_start is not None:
                mix.loc[real_start:] = px.loc[real_start:, real]
        px[mix_name] = mix

    make_mix("TQQQ", "TQQQ_PROXY", "TQQQ_MIX")
    make_mix("UPRO", "UPRO_PROXY", "UPRO_MIX")
    make_mix("SOXL", "SOXL_PROXY", "SOXL_MIX")

    # 정렬
    base_cols = [c for c in base_tickers if c in px.columns]
    px = px.dropna(subset=base_cols, how="any")

    if px.empty:
        raise ValueError("Price data empty after alignment.")

    return px