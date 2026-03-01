
import pandas as pd

def cagr(curve: pd.Series) -> float:
    if curve is None or len(curve) < 2:
        return float("nan")
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (curve.iloc[-1] / curve.iloc[0]) ** (1.0 / years) - 1.0

def mdd(curve: pd.Series) -> float:
    if curve is None or len(curve) < 2:
        return float("nan")
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())
