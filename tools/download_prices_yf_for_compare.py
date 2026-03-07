from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_close(symbol: str, start_date: str, end_date: str) -> pd.Series:
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_plus,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"yfinance returned empty dataframe for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            s = df["Close"].iloc[:, 0].copy()
        else:
            raise ValueError(f"{symbol} result does not contain Close")
    else:
        if "Close" not in df.columns:
            raise ValueError(f"{symbol} result does not contain Close")
        s = df["Close"].copy()

    s.index = pd.to_datetime(s.index)
    s = s.sort_index().astype(float)
    s.name = symbol
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--out-csv", default="data/prices.csv")
    args = parser.parse_args()

    raw_symbols = ["QQQ", "SPY", "SOXX", "TQQQ", "UPRO", "SOXL", "SGOV"]

    series_list: list[pd.Series] = []
    for sym in raw_symbols:
        print(f"downloading {sym} ...", flush=True)
        s = download_close(sym, args.start_date, args.end_date)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1).sort_index()

    rename_map = {
        "TQQQ": "TQQQ_MIX",
        "UPRO": "UPRO_MIX",
        "SOXL": "SOXL_MIX",
        "SGOV": "SGOV_MIX",
    }
    prices = prices.rename(columns=rename_map)

    prices.index.name = "date"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_csv)

    print(f"saved: {out_csv}")
    print(f"rows={len(prices)} cols={list(prices.columns)}")


if __name__ == "__main__":
    main()