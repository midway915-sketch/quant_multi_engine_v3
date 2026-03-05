# src/core/data.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import yfinance as yf


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]


def _extract_px(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance download 결과에서 가격 매트릭스(px)를 뽑는다.
    기본: Adj Close 우선, 없으면 Close.
    반환: index=DatetimeIndex, columns=tickers
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.levels[0]
        if "Adj Close" in lvl0:
            px = df["Adj Close"].copy()
        elif "Close" in lvl0:
            px = df["Close"].copy()
        else:
            # fallback: 첫 레벨 선택
            px = df.xs(lvl0[0], level=0, axis=1).copy()
    else:
        # 단일 티커일 때 종종 단일 컬럼 구조가 나올 수 있음
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].copy()
            # 컬럼명을 티커로 바꾸는 건 호출부에서 처리
        elif "Close" in df.columns:
            px = df[["Close"]].copy()
        else:
            return pd.DataFrame()

    px = px.dropna(how="all").sort_index()
    return px


def _yf_download(
    tickers: List[str],
    start: str,
    tries: int = 3,
    sleep_sec: float = 1.5,
) -> pd.DataFrame:
    """
    yfinance 다운로드 재시도 래퍼
    """
    last_exc = None
    for i in range(tries):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,   # Actions에서 종종 더 안정적
            )
            return df
        except Exception as e:
            last_exc = e
            time.sleep(sleep_sec * (i + 1))
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def download_prices_and_build_proxies(cfg: Dict) -> pd.DataFrame:
    """
    - cfg["data"]["tickers"] 기반으로 가격 다운로드
    - (필요 시) 프록시/믹스 시계열 생성
    - 공통 시작일로 정렬하여 빈 데이터 방지
    """
    base_tickers = _as_list(cfg.get("data", {}).get("tickers"))
    if not base_tickers:
        raise ValueError("cfg.data.tickers is empty")

    start = str(cfg.get("data", {}).get("start", "2010-01-01"))

    # 네가 쓰는 실티커/실험티커 포함 (원하면 여기 줄여도 됨)
    real_3x = ["TQQQ", "UPRO", "SOXL"]
    real_2x = ["QLD", "SSO", "USD"]
    extra_real = ["BIL", "SGOV", "SH", "PSQ", "SHY"]

    tickers = sorted(set(base_tickers + real_3x + real_2x + extra_real))

    df = _yf_download(tickers, start=start, tries=3)
    if df is None or df.empty:
        raise ValueError("Downloaded price data is empty (yf returned empty).")

    px = _extract_px(df)
    if px.empty:
        raise ValueError("Failed to extract price matrix from yfinance result.")

    # 컬럼이 제대로 ticker명인지 보정 (단일 티커 케이스 등)
    # MultiIndex에서 뽑은 경우는 이미 tickers 컬럼이 맞는 편이고,
    # 단일 컬럼(Adj Close/Close)만 남는 경우가 있어 base_tickers로 매핑 시도.
    if len(px.columns) == 1 and px.columns[0] in ["Adj Close", "Close"]:
        # 단일 티커 다운로드였다고 가정하고 첫 base_ticker로 이름 부여
        px.columns = [base_tickers[0]]

    px = px.ffill()

    # ---- (A) 프록시/믹스 생성 ----
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

    # 3x
    make_proxy("QQQ", "TQQQ_PROXY", 3.0)
    make_proxy("SPY", "UPRO_PROXY", 3.0)
    make_proxy("SOXX", "SOXL_PROXY", 3.0)
    make_mix("TQQQ", "TQQQ_PROXY", "TQQQ_MIX")
    make_mix("UPRO", "UPRO_PROXY", "UPRO_MIX")
    make_mix("SOXL", "SOXL_PROXY", "SOXL_MIX")

    # 2x
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

    # cash-like (임시)
    make_proxy("SHY", "BIL_PROXY", 1.0)
    make_proxy("SHY", "SGOV_PROXY", 1.0)
    make_mix("BIL", "BIL_PROXY", "BIL_MIX")
    make_mix("SGOV", "SGOV_PROXY", "SGOV_MIX")

    # ---- (B) base ticker 유효성 체크 + 개별 재다운로드 ----
    bad = []
    for t in base_tickers:
        if t not in px.columns:
            bad.append((t, "missing_column"))
            continue
        if px[t].dropna().empty:
            bad.append((t, "all_nan"))

    # 개별로 다시 받아서 복구 시도 (GitHub Actions에서 가장 효과적)
    if bad:
        for (t, reason) in bad:
            # t만 단독으로 재시도 다운로드
            df1 = _yf_download([t], start=start, tries=3)
            px1 = _extract_px(df1)
            if px1.empty:
                continue

            # 단일 티커면 보통 컬럼이 1개라서 이름 맞춰줌
            if len(px1.columns) == 1:
                px1.columns = [t]
            # 병합
            px = px.join(px1[[t]].ffill(), how="outer")

        px = px.sort_index().ffill()

    # ---- (C) 공통 시작일로 정렬(핵심) ----
    base_cols = [t for t in base_tickers if t in px.columns]
    if not base_cols:
        raise ValueError(f"No base tickers found in price columns. base_tickers={base_tickers}")

    firsts = []
    for t in base_cols:
        fi = px[t].first_valid_index()
        if fi is None:
            firsts.append(None)
        else:
            firsts.append(fi)

    if any(x is None for x in firsts):
        missing = [base_cols[i] for i, x in enumerate(firsts) if x is None]
        raise ValueError(f"Some base tickers have no valid data after retries: {missing}")

    common_start = max(firsts)
    px = px.loc[common_start:].copy()

    # 마지막 안전망: 공통 시작 이후에도 NA 남아있으면 그 구간만 제거
    px = px.dropna(subset=base_cols, how="any")
    if px.empty:
        # 디버그 정보 포함해서 실패 원인 바로 보이게
        info = {}
        for t in base_cols:
            s = px[t] if t in px.columns else pd.Series(dtype=float)
            info[t] = {
                "has_col": t in px.columns,
                "non_na": int(px[t].notna().sum()) if t in px.columns else 0,
            }
        raise ValueError(f"Price data empty after alignment. common_start={common_start} info={info}")

    return px


# 기존 코드 호환용 alias
def download_prices(cfg: Dict) -> pd.DataFrame:
    return download_prices_and_build_proxies(cfg)