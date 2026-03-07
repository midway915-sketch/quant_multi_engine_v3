from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def load_proxy_from_snapshot_root(snapshot_root: Path, symbols: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for symbol in symbols:
        p = snapshot_root / symbol / "snapshot_closes_wide.csv"
        if not p.exists():
            raise FileNotFoundError(f"snapshot wide file not found: {p}")

        df = pd.read_csv(p)
        if "date" not in df.columns or "15:50:00" not in df.columns:
            raise ValueError(f"{p} must contain columns: date, 15:50:00")

        df = df[["date", "15:50:00"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"15:50:00": symbol})
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="outer")

    assert merged is not None
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def download_yfinance_close(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    # yfinance end는 보통 exclusive 느낌이라 하루 더해서 받는 게 안전
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_plus,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise ValueError("yfinance returned empty dataframe")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("yfinance result does not contain Close level")
        close = data["Close"].copy()
    else:
        # 단일 심볼 방어
        close = data[["Close"]].copy()
        close.columns = symbols[:1]

    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    close = close.reset_index().sort_values("date").reset_index(drop=True)
    return close


def compute_summary(
    yf_close: pd.DataFrame,
    proxy_close: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    merged = yf_close.merge(proxy_close, on="date", how="inner", suffixes=("_yf", "_proxy"))
    rows: list[dict] = []

    for symbol in symbols:
        c_yf = f"{symbol}_yf"
        c_px = f"{symbol}_proxy"

        sub = merged[["date", c_yf, c_px]].dropna().copy()
        if sub.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "rows_overlap": 0,
                    "start_date": "",
                    "end_date": "",
                    "corr_daily_ret": float("nan"),
                    "mean_abs_ret_diff": float("nan"),
                    "std_ret_diff": float("nan"),
                    "mean_abs_price_diff": float("nan"),
                }
            )
            continue

        sub["ret_yf"] = sub[c_yf].pct_change()
        sub["ret_proxy"] = sub[c_px].pct_change()
        rets = sub[["ret_yf", "ret_proxy"]].dropna().copy()

        corr = float(rets["ret_yf"].corr(rets["ret_proxy"])) if not rets.empty else float("nan")
        mean_abs_ret_diff = float((rets["ret_yf"] - rets["ret_proxy"]).abs().mean()) if not rets.empty else float("nan")
        std_ret_diff = float((rets["ret_yf"] - rets["ret_proxy"]).std()) if not rets.empty else float("nan")
        mean_abs_price_diff = float((sub[c_yf] - sub[c_px]).abs().mean())

        rows.append(
            {
                "symbol": symbol,
                "rows_overlap": int(len(sub)),
                "start_date": sub["date"].min().strftime("%Y-%m-%d"),
                "end_date": sub["date"].max().strftime("%Y-%m-%d"),
                "corr_daily_ret": corr,
                "mean_abs_ret_diff": mean_abs_ret_diff,
                "std_ret_diff": std_ret_diff,
                "mean_abs_price_diff": mean_abs_price_diff,
            }
        )

    return pd.DataFrame(rows)


def filter_period(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", required=True, help="e.g. data/twelvedata_snapshots")
    parser.add_argument("--symbols", default="QQQ,SPY,SOXX")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--out-dir", default="data/compare_1v2")
    args = parser.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_root = Path(args.snapshot_root)

    proxy_close = load_proxy_from_snapshot_root(snapshot_root, symbols)
    proxy_close = filter_period(proxy_close, args.start_date, args.end_date)

    yf_close = download_yfinance_close(symbols, args.start_date, args.end_date)
    yf_close = filter_period(yf_close, args.start_date, args.end_date)

    summary = compute_summary(yf_close, proxy_close, symbols)

    proxy_path = out_dir / "proxy_close_1550_wide.csv"
    yf_path = out_dir / "yf_close_wide.csv"
    summary_path = out_dir / "compare_summary.csv"

    proxy_close.to_csv(proxy_path, index=False)
    yf_close.to_csv(yf_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"saved: {proxy_path}")
    print(f"saved: {yf_path}")
    print(f"saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()