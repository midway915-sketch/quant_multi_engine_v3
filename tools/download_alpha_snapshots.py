from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


API_URL = "https://www.alphavantage.co/query"
DEFAULT_SNAPSHOT_TIMES = ["15:45:00", "15:50:00", "15:54:00"]


@dataclass
class Config:
    api_key: str
    symbols: list[str]
    start_month: str
    end_month: str
    interval: str
    adjusted: bool
    extended_hours: bool
    datatype: str
    sleep_seconds: float
    out_dir: Path
    snapshot_times: list[str]


def month_range(start_month: str, end_month: str) -> list[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must be >= start_month")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def fetch_month_csv(
    api_key: str,
    symbol: str,
    month: str,
    interval: str,
    adjusted: bool,
    extended_hours: bool,
    datatype: str = "csv",
    timeout: int = 60,
) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": month,
        "outputsize": "full",
        "datatype": datatype,
        "adjusted": "true" if adjusted else "false",
        "extended_hours": "true" if extended_hours else "false",
        "apikey": api_key,
    }

    resp = requests.get(API_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()

    if not text:
        raise RuntimeError(f"Empty response for {symbol} {month}")

    lowered = text.lower()
    if lowered.startswith("{"):
        # Alpha Vantage often returns json error/info even if datatype=csv
        raise RuntimeError(f"Non-CSV response for {symbol} {month}: {text[:500]}")

    df = pd.read_csv(io.StringIO(text))
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    if not expected.issubset(set(df.columns)):
        raise RuntimeError(
            f"Unexpected CSV columns for {symbol} {month}: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_snapshots(
    month_df: pd.DataFrame,
    symbol: str,
    snapshot_times: Iterable[str],
) -> pd.DataFrame:
    df = month_df.copy()
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")

    keep = set(snapshot_times)
    snap = df[df["time"].isin(keep)].copy()

    if snap.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "date",
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    snap.insert(0, "symbol", symbol)
    snap = snap[
        ["symbol", "timestamp", "date", "time", "open", "high", "low", "close", "volume"]
    ].reset_index(drop=True)
    return snap


def save_symbol_outputs(
    out_dir: Path,
    symbol: str,
    all_monthly_snaps: list[pd.DataFrame],
) -> tuple[Path, Path]:
    symbol_dir = out_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    if all_monthly_snaps:
        snaps = pd.concat(all_monthly_snaps, ignore_index=True).sort_values("timestamp")
    else:
        snaps = pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "date",
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    out_csv = symbol_dir / "snapshots.csv"
    snaps.to_csv(out_csv, index=False)

    pivot = snaps.pivot_table(
        index="date",
        columns="time",
        values="close",
        aggfunc="last",
    ).reset_index()

    pivot.columns.name = None
    wanted_cols = ["date"] + [t for t in DEFAULT_SNAPSHOT_TIMES if t in pivot.columns]
    pivot = pivot[wanted_cols] if not pivot.empty else pd.DataFrame(columns=["date"] + DEFAULT_SNAPSHOT_TIMES)

    out_daily = symbol_dir / "snapshot_closes_wide.csv"
    pivot.to_csv(out_daily, index=False)

    return out_csv, out_daily


def run(cfg: Config) -> None:
    months = month_range(cfg.start_month, cfg.end_month)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for symbol in cfg.symbols:
        print(f"\n=== {symbol} ===")
        monthly_snaps: list[pd.DataFrame] = []

        for i, month in enumerate(months, start=1):
            print(f"[{symbol}] {month} ({i}/{len(months)})", flush=True)
            try:
                month_df = fetch_month_csv(
                    api_key=cfg.api_key,
                    symbol=symbol,
                    month=month,
                    interval=cfg.interval,
                    adjusted=cfg.adjusted,
                    extended_hours=cfg.extended_hours,
                    datatype=cfg.datatype,
                )
                snaps = extract_snapshots(
                    month_df=month_df,
                    symbol=symbol,
                    snapshot_times=cfg.snapshot_times,
                )
                monthly_snaps.append(snaps)

                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "month": month,
                        "rows_month": len(month_df),
                        "rows_snapshots": len(snaps),
                        "status": "ok",
                    }
                )
            except Exception as e:
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "month": month,
                        "rows_month": 0,
                        "rows_snapshots": 0,
                        "status": f"error: {e}",
                    }
                )
                print(f"ERROR [{symbol} {month}] {e}", file=sys.stderr, flush=True)

            if cfg.sleep_seconds > 0:
                time.sleep(cfg.sleep_seconds)

        out_csv, out_daily = save_symbol_outputs(
            out_dir=cfg.out_dir,
            symbol=symbol,
            all_monthly_snaps=monthly_snaps,
        )
        print(f"[{symbol}] saved: {out_csv}")
        print(f"[{symbol}] saved: {out_daily}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = cfg.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nmanifest saved: {manifest_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        required=True,
        help='comma-separated, e.g. "QQQ,SPY,SOXX,SGOV"',
    )
    parser.add_argument("--start-month", required=True, help="YYYY-MM")
    parser.add_argument("--end-month", required=True, help="YYYY-MM")
    parser.add_argument("--interval", default="1min", choices=["1min", "5min", "15min", "30min", "60min"])
    parser.add_argument("--adjusted", action="store_true", help="Use adjusted intraday values")
    parser.add_argument(
        "--extended-hours",
        action="store_true",
        help="Include extended hours. Default is regular hours only.",
    )
    parser.add_argument("--datatype", default="csv", choices=["csv"])
    parser.add_argument("--sleep-seconds", type=float, default=12.5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--snapshot-times",
        default="15:45:00,15:50:00,15:54:00",
        help='comma-separated ET times, e.g. "15:45:00,15:50:00,15:54:00"',
    )

    args = parser.parse_args()
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set ALPHAVANTAGE_API_KEY in environment first.")

    snapshot_times = [x.strip() for x in args.snapshot_times.split(",") if x.strip()]
    if not snapshot_times:
        raise SystemExit("snapshot_times cannot be empty")

    return Config(
        api_key=api_key,
        symbols=[x.strip().upper() for x in args.symbols.split(",") if x.strip()],
        start_month=args.start_month,
        end_month=args.end_month,
        interval=args.interval,
        adjusted=bool(args.adjusted),
        extended_hours=bool(args.extended_hours),
        datatype=args.datatype,
        sleep_seconds=float(args.sleep_seconds),
        out_dir=Path(args.out_dir),
        snapshot_times=snapshot_times,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)