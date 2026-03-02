from __future__ import annotations

import yfinance as yf
import pandas as pd


def download_prices_and_build_proxies(cfg) -> pd.DataFrame:
    base_tickers = list(cfg["data"]["tickers"])

    # Real leveraged ETFs you already use
    real_3x = ["TQQQ", "UPRO", "SOXL"]
    real_2x = ["QLD", "SSO", "USD"]  # QQQ/SPY/SOXX의 2x 대응

    # ✅ Extra defensives / inverse (for risk_off.mode experiments)
    extra_real = ["BIL", "SGOV", "SH", "PSQ"]

    tickers = sorted(set(base_tickers + real_3x + real_2x + extra_real))

    df = yf.download(
        tickers,
        start=cfg["data"]["start"],
        auto_adjust=False,
        progress=False,
        group_by="column",
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
    rets = px.pct_change()

    def make_proxy(base: str, name: str, k: float) -> None:
        """
        Build kx daily-return proxy starting at 1.0, then cumprod.
        NOTE: This is a synthetic proxy (no fees/decay/borrow), used only pre-listing.
        """
        if base not in px.columns:
            return

        r = rets[base].fillna(0.0)
        step = (1.0 + k * r).clip(lower=1e-6)
        px[name] = step.cumprod()

    def make_mix(real: str, proxy: str, mix_name: str) -> None:
        """
        Create MIX series:
        - use PROXY before real ETF listing
        - use REAL from listing onward
        - rebase PROXY level to match REAL just before the switch to avoid discontinuity
        """
        if proxy not in px.columns:
            return

        mix = px[proxy].copy()

        if real in px.columns:
            real_start = px[real].first_valid_index()
            if real_start is not None:
                idx = px.index
                loc = idx.get_loc(real_start)
                prev_date = idx[loc - 1] if isinstance(loc, int) and loc > 0 else None

                if prev_date is not None and pd.notna(px.at[prev_date, real]) and pd.notna(mix.at[prev_date]):
                    denom = float(mix.at[prev_date])
                    if denom != 0.0:
                        mix = mix * (float(px.at[prev_date, real]) / denom)
                else:
                    if pd.notna(px.at[real_start, real]) and pd.notna(mix.at[real_start]):
                        denom = float(mix.at[real_start])
                        if denom != 0.0:
                            mix = mix * (float(px.at[real_start, real]) / denom)

                mix.loc[real_start:] = px.loc[real_start:, real]

        px[mix_name] = mix

    # 3x proxies/mixes
    make_proxy("QQQ", "TQQQ_PROXY", 3.0)
    make_proxy("SPY", "UPRO_PROXY", 3.0)
    make_proxy("SOXX", "SOXL_PROXY", 3.0)
    make_mix("TQQQ", "TQQQ_PROXY", "TQQQ_MIX")
    make_mix("UPRO", "UPRO_PROXY", "UPRO_MIX")
    make_mix("SOXL", "SOXL_PROXY", "SOXL_MIX")

    # 2x proxies/mixes
    make_proxy("QQQ", "QLD_PROXY", 2.0)
    make_proxy("SPY", "SSO_PROXY", 2.0)
    make_proxy("SOXX", "USD_PROXY", 2.0)
    make_mix("QLD", "QLD_PROXY", "QLD_MIX")
    make_mix("SSO", "SSO_PROXY", "SSO_MIX")
    make_mix("USD", "USD_PROXY", "USD_MIX")

    # ✅ Inverse equity proxies/mixes (-1x)
    make_proxy("SPY", "SH_PROXY", -1.0)
    make_proxy("QQQ", "PSQ_PROXY", -1.0)
    make_mix("SH", "SH_PROXY", "SH_MIX")
    make_mix("PSQ", "PSQ_PROXY", "PSQ_MIX")

    # ✅ Cash-like T-bills mixes (pre-listing proxy: SHY)
    make_proxy("SHY", "BIL_PROXY", 1.0)
    make_proxy("SHY", "SGOV_PROXY", 1.0)
    make_mix("BIL", "BIL_PROXY", "BIL_MIX")
    make_mix("SGOV", "SGOV_PROXY", "SGOV_MIX")

    # Align by base tickers existence
    base_cols = [c for c in base_tickers if c in px.columns]
    px = px.dropna(subset=base_cols, how="any")

    if px.empty:
        raise ValueError("Price data empty after alignment.")

    return px