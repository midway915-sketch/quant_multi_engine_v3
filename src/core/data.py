from __future__ import annotations

import yfinance as yf
import pandas as pd


def download_prices_and_build_proxies(cfg) -> pd.DataFrame:
    """
    Download prices via yfinance and build synthetic proxy/mix series.

    Fixes:
    - GitHub Actions에서 yfinance가 간헐적으로 일부 티커를 비워주는 경우가 있어,
      단순 dropna(how="any")로 전체가 날아가던 문제를 방지.
    - required tickers의 공통 구간(start = max(first_valid_index))으로 자르고 그 뒤 dropna 수행.
    - 전체 NaN 컬럼은 required에서 제외하고, 진단 메시지 강화.
    - yfinance download 재시도(최대 3회) 추가.
    """
    import time

    base_tickers = list(cfg["data"]["tickers"])

    # Real leveraged ETFs you already use
    real_3x = ["TQQQ", "UPRO", "SOXL"]
    real_2x = ["QLD", "SSO", "USD"]  # QQQ/SPY/SOXX의 2x 대응

    # Extra defensives / inverse (for risk_off.mode experiments)
    extra_real = ["BIL", "SGOV", "SH", "PSQ", "SHY"]

    tickers = sorted(set(base_tickers + real_3x + real_2x + extra_real))

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
                threads=False,  # CI에서 가끔 안정성↑
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
        time.sleep(2 * attempt)

    if df is None or df.empty:
        raise ValueError(f"Downloaded price data is empty. last_err={last_err!r}")

    # pick Adj Close if exists, else Close
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.levels[0]
        if "Adj Close" in lvl0:
            px = df["Adj Close"].copy()
        elif "Close" in lvl0:
            px = df["Close"].copy()
        else:
            # fallback: take first field
            px = df.xs(lvl0[0], level=0, axis=1).copy()
    else:
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            px = df[["Close"]].copy()
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

                # rebase proxy level to match real just before switch
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

    # Inverse equity proxies/mixes (-1x)
    make_proxy("SPY", "SH_PROXY", -1.0)
    make_proxy("QQQ", "PSQ_PROXY", -1.0)
    make_mix("SH", "SH_PROXY", "SH_MIX")
    make_mix("PSQ", "PSQ_PROXY", "PSQ_MIX")

    # Cash-like T-bills mixes (pre-listing proxy: SHY)
    make_proxy("SHY", "BIL_PROXY", 1.0)
    make_proxy("SHY", "SGOV_PROXY", 1.0)
    make_mix("BIL", "BIL_PROXY", "BIL_MIX")
    make_mix("SGOV", "SGOV_PROXY", "SGOV_MIX")

    # ---- alignment (robust) ----
    # 1) drop all-NaN columns to avoid "any" drop wiping everything
    all_nan_cols = [c for c in px.columns if px[c].isna().all()]
    if all_nan_cols:
        px = px.drop(columns=all_nan_cols)

    # 2) required tickers = base_tickers ∩ px.columns
    required = [t for t in base_tickers if t in px.columns]
    if not required:
        raise ValueError(
            "None of base tickers exist in downloaded price columns. "
            f"base_tickers={base_tickers} available_cols(sample)={list(px.columns)[:20]}"
        )

    # 3) find common start date (max of first valid among required)
    first_valid = {t: px[t].first_valid_index() for t in required}
    missing = [t for t, d in first_valid.items() if d is None]
    if missing:
        raise ValueError(
            "Some required tickers have no valid data (all NaN). "
            f"missing={missing} first_valid={first_valid}"
        )

    common_start = max(first_valid.values())
    px = px.loc[common_start:].copy()

    # 4) now enforce all required present each day
    px = px.dropna(subset=required, how="any")

    if px.empty:
        diag = {
            "required": required,
            "common_start": str(common_start),
            "first_valid": {k: str(v) for k, v in first_valid.items()},
            "last_valid": {t: str(px_full_last) for t, px_full_last in {t: df["Adj Close"][t].last_valid_index() if (isinstance(df.columns, pd.MultiIndex) and "Adj Close" in df.columns.levels[0] and t in df["Adj Close"].columns) else None for t in required}.items()},
            "note": "After trimming to common_start, dropna(any) still produced empty. Likely data gaps or download failure on CI.",
        }
        raise ValueError(f"Price data empty after alignment. diag={diag}")

    return px


def download_prices(cfg) -> pd.DataFrame:
    """
    ✅ Compatibility wrapper.
    Existing scripts expect:
        from src.core.data import download_prices
    so we provide it and reuse your original implementation.
    """
    return download_prices_and_build_proxies(cfg)