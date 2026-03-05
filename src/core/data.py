from __future__ import annotations

import time
import pandas as pd
import yfinance as yf


def download_prices_and_build_proxies(cfg) -> pd.DataFrame:
    base_tickers = list(cfg["data"]["tickers"])

    # Real leveraged ETFs you already use
    real_3x = ["TQQQ", "UPRO", "SOXL"]
    real_2x = ["QLD", "SSO", "USD"]  # QQQ/SPY/SOXX의 2x 대응

    # Extra defensives / inverse (for risk_off.mode experiments)
    extra_real = ["BIL", "SGOV", "SH", "PSQ", "SHY"]

    tickers = sorted(set(base_tickers + real_3x + real_2x + extra_real))

    # ✅ price field 선택 (adj_close | close)
    data_cfg = (cfg.get("data", {}) or {})
    price_field = str(data_cfg.get("price_field", "adj_close")).lower().strip()
    use_adj = price_field in ("adj", "adjclose", "adj_close", "adj close")

    # ---- robust download (retry) ----
    last_err = None
    df = None
    for attempt in range(1, 4):
        try:
            df = yf.download(
                tickers,
                start=cfg["data"]["start"],
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
        time.sleep(2 * attempt)

    if df is None or df.empty:
        raise ValueError(f"Downloaded price data is empty. last_err={last_err!r}")

    # pick Adj Close (default) or Close
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.levels[0]
        if use_adj and ("Adj Close" in lvl0):
            px = df["Adj Close"].copy()
        elif "Close" in lvl0:
            px = df["Close"].copy()
        elif "Adj Close" in lvl0:
            px = df["Adj Close"].copy()
        else:
            px = df.xs(lvl0[0], level=0, axis=1).copy()
    else:
        # single ticker fallback
        if use_adj and ("Adj Close" in df.columns):
            px = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            px = df[["Close"]].copy()
        elif "Adj Close" in df.columns:
            px = df[["Adj Close"]].copy()
        else:
            raise ValueError("Neither Adj Close nor Close found.")

    px = px.dropna(how="all").sort_index().ffill()

    # daily returns for proxy building
    rets = px.pct_change()

    def make_proxy(base: str, name: str, k: float) -> None:
        if base not in px.columns:
            return
        r = rets[base].fillna(0.0)
        step = (1.0 + k * r).clip(lower=1e-6)
        px[name] = step.cumprod()

    def make_mix(real: str, proxy: str, mix_name: str) -> None:
        if proxy not in px.columns:
            return

        mix = px[proxy].copy()

        if real in px.columns:
            real_start = px[real].first_valid_index()
            if real_start is not None:
                idx = px.index
                loc = idx.get_loc(real_start)
                prev_date = idx[loc - 1] if isinstance(loc, int) and loc > 0 else None

                # rebase proxy to match real just before switching
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

    # inverse -1x
    make_proxy("SPY", "SH_PROXY", -1.0)
    make_proxy("QQQ", "PSQ_PROXY", -1.0)
    make_mix("SH", "SH_PROXY", "SH_MIX")
    make_mix("PSQ", "PSQ_PROXY", "PSQ_MIX")

    # cash-like proxy from SHY
    make_proxy("SHY", "BIL_PROXY", 1.0)
    make_proxy("SHY", "SGOV_PROXY", 1.0)
    make_mix("BIL", "BIL_PROXY", "BIL_MIX")
    make_mix("SGOV", "SGOV_PROXY", "SGOV_MIX")

    # ---- alignment (robust common-start) ----
    # drop all-NaN columns
    all_nan_cols = [c for c in px.columns if px[c].isna().all()]
    if all_nan_cols:
        px = px.drop(columns=all_nan_cols)

    required = [t for t in base_tickers if t in px.columns]
    if not required:
        raise ValueError(
            f"None of base tickers exist in downloaded price columns. "
            f"base_tickers={base_tickers} available_cols(sample)={list(px.columns)[:20]}"
        )

    first_valid = {t: px[t].first_valid_index() for t in required}
    missing = [t for t, d in first_valid.items() if d is None]
    if missing:
        raise ValueError(f"Some required tickers have no valid data (all NaN): {missing}")

    common_start = max(first_valid.values())
    px = px.loc[common_start:].copy()
    px = px.dropna(subset=required, how="any")

    if px.empty:
        raise ValueError(f"Price data empty after alignment. common_start={common_start} required={required}")

    return px


# backward-compatible alias
def download_prices(cfg) -> pd.DataFrame:
    return download_prices_and_build_proxies(cfg)