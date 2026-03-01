from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if equity.empty:
        raise ValueError("equity curve empty")

    start = equity.index[0]
    end = equity.index[-1]
    days = (end - start).days
    years = days / 365.25 if days > 0 else np.nan

    seed_multiple = float(equity.iloc[-1] / equity.iloc[0])

    if years and years > 0:
        cagr = seed_multiple ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "years": float(years),
        "seed_multiple": float(seed_multiple),
        "cagr": float(cagr),
        "mdd": float(mdd),
    }