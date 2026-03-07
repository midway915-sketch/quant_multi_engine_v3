from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


API_URL = "https://api.twelvedata.com/time_series"


@dataclass
class Config:
    api_key: str
    symbols: list[str]
    start_date: str
    end_date: str
    interval: str
    timezone: str
    out_dir: Path
    snapshot_times: list[str]
    pause_seconds: float
    type_: str


def chunk_ranges(start_date: str, end_date: str, chunk_days: int = 30) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        ranges.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return ranges


def fetch_chunk(
    api_key: str,
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    interval: str,
    timezone: str,
    type_: str,
    timeout: int = 60,
) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "timezone": timezone,
        "order": "ASC",
        "format": "JSON",
        "apikey": api_key,
        "type": type_,
    }

    resp = requests.get(API_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if "status" in payload and str(payload["status"]).lower() == "error":
        msg = payload.get("message", "unknown error")
        raise RuntimeError(f"{symbol} {start_dt.date()}~{end_dt.date()} error: {msg}")

    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(values)
    expected = {"datetime", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"{symbol} missing columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return df


def extract_snapshots(df: pd.DataFrame, symbol: str, snapshot_times: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["symbol", "datetime", "date", "time", "open", "high", "low", "close", "volume"]
        )

    out = df.copy()
    out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["time"] = out["datetime"].dt.strftime("%H:%M:%S")
    out = out[out["time"].isin(set(snapshot_times))].copy()
    if out.empty:
        return pd.DataFrame(
            columns=["symbol", "datetime", "date", "time", "open", "high", "low", "close", "volume"]
        )

    out.insert(0, "symbol", symbol)
    return out[["symbol", "datetime", "date", "time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def save_outputs(out_dir: Path, symbol: str, snapshots: pd.DataFrame) -> tuple[Path, Path]:
    symbol_dir = out_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    raw_path = symbol_dir / "snapshots.csv"
    snapshots.to_csv(raw_path, index=False)

    if snapshots.empty:
        wide = pd.DataFrame(columns=["date", "15:45:00", "15:50:00", "15:54:00"])
    else:
        wide = snapshots.pivot_table(
            index="date",
            columns="time",
            values="close",
            aggfunc="last",
        ).reset_index()
        wide.columns.name = None
        ordered = ["date", "15:45:00", "15:50:00", "15:54:00"]
        for c in ordered:
            if c not in wide.columns:
                wide[c] = pd.NA
        wide = wide[ordered]

    wide_path = symbol_dir / "snapshot_closes_wide.csv"
    wide.to_csv(wide_path, index=False)

    return raw_path, wide_path


def run(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ranges = chunk_ranges(cfg.start_date, cfg.end_date, chunk_days=30)
    manifest_rows: list[dict] = []

    for symbol in cfg.symbols:
        print(f"\n=== {symbol} ===", flush=True)
        all_parts: list[pd.DataFrame] = []

        for idx, (start_dt, end_dt) in enumerate(ranges, start=1):
            print(
                f"[{symbol}] chunk {idx}/{len(ranges)} {start_dt.date()} ~ {end_dt.date()}",
                flush=True,
            )
            try:
                chunk = fetch_chunk(
                    api_key=cfg.api_key,
                    symbol=symbol,
                    start_dt=start_dt,
                    end_dt=end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59),
                    interval=cfg.interval,
                    timezone=cfg.timezone,
                    type_=cfg.type_,
                )
                all_parts.append(chunk)
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "chunk_start": str(start_dt.date()),
                        "chunk_end": str(end_dt.date()),
                        "rows_raw": int(len(chunk)),
                        "status": "ok",
                    }
                )
            except Exception as e:
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "chunk_start": str(start_dt.date()),
                        "chunk_end": str(end_dt.date()),
                        "rows_raw": 0,
                        "status": f"error: {e}",
                    }
                )
                print(f"ERROR [{symbol} {start_dt.date()}~{end_dt.date()}] {e}", file=sys.stderr, flush=True)

            if cfg.pause_seconds > 0:
                time.sleep(cfg.pause_seconds)

        if all_parts:
            raw = pd.concat(all_parts, ignore_index=True).sort_values("datetime")
            raw = raw.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        else:
            raw = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        snaps = extract_snapshots(raw, symbol, cfg.snapshot_times)
        raw_path, wide_path = save_outputs(cfg.out_dir, symbol, snaps)

        print(f"[{symbol}] snapshots saved: {raw_path}", flush=True)
        print(f"[{symbol}] wide saved: {wide_path}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = cfg.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nmanifest saved: {manifest_path}", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--symbols", default="QQQ,SPY,SOXX")
    parser.add_argument("--start-date", default="2022-04-01")
    parser.add_argument("--end-date", default="2022-09-30")
    parser.add_argument("--interval", default="1min")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--out-dir", default="data/twelvedata_snapshots")
    parser.add_argument("--snapshot-times", default="15:45:00,15:50:00,15:54:00")
    parser.add_argument("--pause-seconds", type=float, default=8.0)
    parser.add_argument("--type", dest="type_", default="ETF")
    args = parser.parse_args()

    return Config(
        api_key=args.api_key.strip(),
        symbols=[x.strip().upper() for x in args.symbols.split(",") if x.strip()],
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        timezone=args.timezone,
        out_dir=Path(args.out_dir),
        snapshot_times=[x.strip() for x in args.snapshot_times.split(",") if x.strip()],
        pause_seconds=float(args.pause_seconds),
        type_=args.type_,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)