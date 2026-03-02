from __future__ import annotations
import numpy as np
import pandas as pd


def _slice_last_n_years(equity: pd.Series, years: int = 10) -> pd.Series:
    equity = equity.dropna()
    if equity.empty:
        return equity

    end = equity.index[-1]
    start = end - pd.DateOffset(years=years)

    eq = equity[equity.index >= start]
    # 만약 데이터가 너무 짧으면(예: 시작이 10년보다 더 최근) 그냥 전체를 사용
    if eq.empty:
        return equity
    return eq


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if equity.empty:
        raise ValueError("equity curve empty")

    start = equity.index[0]
    end = equity.index[-1]
    days = (end - start).days
    years = days / 365.25 if days > 0 else np.nan

    seed_multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(seed_multiple ** (1.0 / years) - 1.0) if years and years > 0 else np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    # ---- last 10y metrics ----
    eq10 = _slice_last_n_years(equity, years=10)
    start10 = eq10.index[0]
    end10 = eq10.index[-1]
    days10 = (end10 - start10).days
    years10 = days10 / 365.25 if days10 > 0 else np.nan

    seed_multiple_10y = float(eq10.iloc[-1] / eq10.iloc[0]) if len(eq10) > 1 else np.nan
    cagr_10y = float(seed_multiple_10y ** (1.0 / years10) - 1.0) if years10 and years10 > 0 else np.nan

    peak10 = eq10.cummax()
    dd10 = eq10 / peak10 - 1.0
    mdd_10y = float(dd10.min()) if len(eq10) > 0 else np.nan

    return {
        # full period
        "start": str(start.date()),
        "end": str(end.date()),
        "years": float(years),
        "seed_multiple": float(seed_multiple),
        "cagr": float(cagr),
        "mdd": float(mdd),

        # last 10y
        "start_10y": str(start10.date()),
        "end_10y": str(end10.date()),
        "years_10y": float(years10),
        "seed_multiple_10y": float(seed_multiple_10y),
        "cagr_10y": float(cagr_10y),
        "mdd_10y": float(mdd_10y),
    }