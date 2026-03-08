from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["TQQQ", "UPRO", "SOXL", "SGOV"]


def download_one(symbol: str, start_date: str, end_date: str) -> pd.Series:
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

    if df is None or df.empty:
        raise RuntimeError(f"empty price data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise RuntimeError(f"missing Close for {symbol}")
        s = df["Close"].iloc[:, 0].copy()
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"missing Close for {symbol}")
        s = df["Close"].copy()

    s.index = pd.to_datetime(s.index)
    s = s.sort_index().astype(float)
    s.name = f"{symbol}_MIX" if symbol != "SGOV" else "SGOV_MIX"
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-csv", default="data/prices.csv")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    series_list: list[pd.Series] = []
    for symbol in TICKERS:
        print(f"downloading {symbol} ...", flush=True)
        s = download_one(symbol, args.start_date, args.end_date)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1).sort_index()
    prices.index.name = "date"

    prices.to_csv(out_csv)
    print(f"saved: {out_csv}")
    print(prices.tail(10).to_string())


if __name__ == "__main__":
    main()