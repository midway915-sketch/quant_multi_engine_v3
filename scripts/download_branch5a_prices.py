from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def _extract_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].copy()
        elif ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)].copy()
        else:
            raise KeyError(f"Close/Adj Close not found for {ticker}")
    else:
        if "Close" in df.columns:
            s = df["Close"].copy()
        elif "Adj Close" in df.columns:
            s = df["Adj Close"].copy()
        else:
            raise KeyError(f"Close/Adj Close not found for {ticker}")
    s.name = ticker
    return s.astype(float)


def _download_series(
    tickers: list[str],
    start: str,
    end: str | None,
) -> dict[str, pd.Series]:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            s = _extract_close(raw, t)
            s.index = pd.to_datetime(s.index)
            out[t] = s.sort_index()
        except Exception:
            out[t] = pd.Series(dtype=float, name=t)
    return out


def _synthetic_leveraged_from_underlying(
    underlying: pd.Series,
    leverage: float,
    actual: pd.Series | None,
    out_name: str,
    start_value: float = 100.0,
) -> pd.Series:
    px = underlying.astype(float).dropna().copy()
    if px.empty:
        return pd.Series(dtype=float, name=out_name)

    rets = px.pct_change().fillna(0.0)
    lev_rets = (leverage * rets).clip(lower=-0.999)
    synth = (1.0 + lev_rets).cumprod() * start_value
    synth.name = out_name

    if actual is None or actual.dropna().empty:
        return synth

    actual = actual.astype(float).dropna().copy()
    actual = actual[actual.index.isin(synth.index)]
    if actual.empty:
        return synth

    first_actual_dt = actual.index[0]
    if first_actual_dt not in synth.index:
        return synth

    scale = float(actual.loc[first_actual_dt]) / float(synth.loc[first_actual_dt])
    synth = synth * scale

    out = synth.copy()
    out.loc[out.index >= first_actual_dt] = actual.loc[actual.index >= first_actual_dt]
    out.name = out_name
    return out


def _splice_defensive(
    sgov: pd.Series | None,
    bil: pd.Series | None,
    out_name: str = "SGOV_MIX",
) -> pd.Series:
    if sgov is not None:
        sgov = sgov.astype(float).dropna().copy()
    else:
        sgov = pd.Series(dtype=float)

    if bil is not None:
        bil = bil.astype(float).dropna().copy()
    else:
        bil = pd.Series(dtype=float)

    if sgov.empty and bil.empty:
        return pd.Series(dtype=float, name=out_name)
    if sgov.empty:
        out = bil.copy()
        out.name = out_name
        return out
    if bil.empty:
        out = sgov.copy()
        out.name = out_name
        return out

    first_sgov_dt = sgov.index.min()
    out = bil.copy()
    out = out[out.index < first_sgov_dt]
    out = pd.concat([out, sgov]).sort_index()
    out.name = out_name
    return out


def build_prices(start: str, end: str | None) -> pd.DataFrame:
    tickers = ["QQQ", "SPY", "SOXX", "TQQQ", "UPRO", "SOXL", "SGOV", "BIL"]
    data = _download_series(tickers=tickers, start=start, end=end)

    qqq = data["QQQ"]
    spy = data["SPY"]
    soxx = data["SOXX"]

    tqqq_mix = _synthetic_leveraged_from_underlying(
        underlying=qqq,
        leverage=3.0,
        actual=data["TQQQ"],
        out_name="TQQQ_MIX",
    )
    upro_mix = _synthetic_leveraged_from_underlying(
        underlying=spy,
        leverage=3.0,
        actual=data["UPRO"],
        out_name="UPRO_MIX",
    )
    soxl_mix = _synthetic_leveraged_from_underlying(
        underlying=soxx,
        leverage=3.0,
        actual=data["SOXL"],
        out_name="SOXL_MIX",
    )
    sgov_mix = _splice_defensive(
        sgov=data["SGOV"],
        bil=data["BIL"],
        out_name="SGOV_MIX",
    )

    all_idx = qqq.index.union(spy.index).union(soxx.index)
    all_idx = all_idx.union(tqqq_mix.index).union(upro_mix.index).union(soxl_mix.index).union(sgov_mix.index)
    all_idx = pd.DatetimeIndex(sorted(all_idx))

    df = pd.DataFrame(index=all_idx)
    df["QQQ"] = qqq.reindex(all_idx)
    df["SPY"] = spy.reindex(all_idx)
    df["SOXX"] = soxx.reindex(all_idx)
    df["TQQQ_MIX"] = tqqq_mix.reindex(all_idx)
    df["UPRO_MIX"] = upro_mix.reindex(all_idx)
    df["SOXL_MIX"] = soxl_mix.reindex(all_idx)
    df["SGOV_MIX"] = sgov_mix.reindex(all_idx)

    df = df.sort_index()
    df = df[df["QQQ"].notna() & df["SPY"].notna() & df["SOXX"].notna()]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_prices(start=args.start, end=args.end)
    df.to_csv(out_csv, index=True)
    print(f"saved: {out_csv}")
    print(df.tail())


if __name__ == "__main__":
    main()