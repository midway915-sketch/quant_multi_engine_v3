from __future__ import annotations
import yfinance as yf
import pandas as pd


def download_prices_and_build_proxies(cfg) -> pd.DataFrame:
    base_tickers = list(cfg["data"]["tickers"])

    real_3x = ["TQQQ", "UPRO", "SOXL"]
    real_2x = ["QLD", "SSO", "USD"]  # QQQ/SPY/SOXX의 2x 대응

    tickers = sorted(set(base_tickers + real_3x + real_2x))

    df = yf.download(
        tickers,
        start=cfg["data"]["start"],
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if df.empty:
        raise ValueError("Downloaded price data is empty.")

    # pick Adj Close if exists, else Close
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.levels[0]
        if "Adj Close" in lvl0:
            px = df["Adj Close"]
        elif "Close" in lvl0:
            px = df["Close"]
        else:
            px = df.xs(lvl0[0], level=0, axis=1)
    else:
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]]
        elif "Close" in df.columns:
            px = df[["Close"]]
        else:
            raise ValueError("Neither Adj Close nor Close found.")

    px = px.dropna(how="all").sort_index().ffill()

    # daily returns for proxy building
    rets = px.pct_change().fillna(0.0)

    def make_proxy(base: str, name: str, k: float):
        if base not in px.columns:
            return
        proxy = (1.0 + (rets[base] * k)).cumprod()
        px[name] = proxy

    # 3x proxies (pre-listing)
    make_proxy("QQQ", "TQQQ_PROXY", 3.0)
    make_proxy("SPY", "UPRO_PROXY", 3.0)
    make_proxy("SOXX", "SOXL_PROXY", 3.0)

    # 2x proxies (pre-listing)
    make_proxy("QQQ", "QLD_PROXY", 2.0)
    make_proxy("SPY", "SSO_PROXY", 2.0)
    make_proxy("SOXX", "USD_PROXY", 2.0)

    def make_mix(real: str, proxy: str, mix_name: str):
        if proxy not in px.columns:
            return
        mix = px[proxy].copy()
        if real in px.columns:
            real_start = px[real].first_valid_index()
            if real_start is not None:
                mix.loc[real_start:] = px.loc[real_start:, real]
        px[mix_name] = mix

    # 3x mixes
    make_mix("TQQQ", "TQQQ_PROXY", "TQQQ_MIX")
    make_mix("UPRO", "UPRO_PROXY", "UPRO_MIX")
    make_mix("SOXL", "SOXL_PROXY", "SOXL_MIX")

    # 2x mixes
    make_mix("QLD", "QLD_PROXY", "QLD_MIX")
    make_mix("SSO", "SSO_PROXY", "SSO_MIX")
    make_mix("USD", "USD_PROXY", "USD_MIX")

    # Align by base tickers existence (keeps your config clean)
    base_cols = [c for c in base_tickers if c in px.columns]
    px = px.dropna(subset=base_cols, how="any")

    if px.empty:
        raise ValueError("Price data empty after alignment.")

    return px