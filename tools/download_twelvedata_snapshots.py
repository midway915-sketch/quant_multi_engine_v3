from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import requests


API_URL = "https://api.twelvedata.com/time_series"


@dataclass
class Config:
    api_key: str
    symbols: list[str]
    test_dates: list[str]
    interval: str
    timezone: str
    security_type: str
    pause_seconds: float
    out_csv: Path


def fetch_one_day(
    api_key: str,
    symbol: str,
    date_str: str,
    interval: str,
    timezone: str,
    security_type: str,
    timeout: int = 60,
) -> dict:
    start_dt = f"{date_str}T15:40:00"
    end_dt = f"{date_str}T16:00:00"

    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_dt,
        "end_date": end_dt,
        "timezone": timezone,
        "order": "ASC",
        "format": "JSON",
        "apikey": api_key,
    }
    if security_type:
        params["type"] = security_type

    resp = requests.get(API_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    status = str(payload.get("status", "")).lower()
    message = str(payload.get("message", "")).strip()

    if status == "error":
        return {
            "status": "error",
            "message": message or "unknown error",
            "rows": 0,
            "first_datetime": "",
            "last_datetime": "",
        }

    values = payload.get("values", [])
    if not values:
        return {
            "status": "no_data",
            "message": message or "no values",
            "rows": 0,
            "first_datetime": "",
            "last_datetime": "",
        }

    first_dt = str(values[0].get("datetime", ""))
    last_dt = str(values[-1].get("datetime", ""))

    return {
        "status": "ok",
        "message": "",
        "rows": len(values),
        "first_datetime": first_dt,
        "last_datetime": last_dt,
    }


def run(cfg: Config) -> None:
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "symbol",
        "test_date",
        "status",
        "rows",
        "first_datetime",
        "last_datetime",
        "message",
    ]

    with cfg.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for symbol in cfg.symbols:
            print(f"\n=== {symbol} ===", flush=True)
            for idx, test_date in enumerate(cfg.test_dates, start=1):
                print(f"[{symbol}] {idx}/{len(cfg.test_dates)} {test_date}", flush=True)
                try:
                    result = fetch_one_day(
                        api_key=cfg.api_key,
                        symbol=symbol,
                        date_str=test_date,
                        interval=cfg.interval,
                        timezone=cfg.timezone,
                        security_type=cfg.security_type,
                    )
                except Exception as e:
                    result = {
                        "status": "exception",
                        "message": str(e),
                        "rows": 0,
                        "first_datetime": "",
                        "last_datetime": "",
                    }

                row = {
                    "symbol": symbol,
                    "test_date": test_date,
                    **result,
                }
                writer.writerow(row)
                f.flush()

                print(
                    f"  -> status={row['status']} rows={row['rows']} "
                    f"first={row['first_datetime']} last={row['last_datetime']} "
                    f"msg={row['message']}",
                    flush=True,
                )

                if cfg.pause_seconds > 0:
                    time.sleep(cfg.pause_seconds)

    print(f"\nsaved: {cfg.out_csv}", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--symbols", default="QQQ,SPY,SOXX")
    parser.add_argument(
        "--test-dates",
        default="2025-02-28,2024-02-28,2023-02-28,2022-02-28,2021-02-26,2020-02-28",
        help="comma-separated YYYY-MM-DD dates",
    )
    parser.add_argument("--interval", default="1min")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--security-type", default="ETF")
    parser.add_argument("--pause-seconds", type=float, default=10.0)
    parser.add_argument("--out-csv", default="data/twelvedata_availability_check/results.csv")
    args = parser.parse_args()

    return Config(
        api_key=args.api_key.strip(),
        symbols=[x.strip().upper() for x in args.symbols.split(",") if x.strip()],
        test_dates=[x.strip() for x in args.test_dates.split(",") if x.strip()],
        interval=args.interval,
        timezone=args.timezone,
        security_type=args.security_type.strip(),
        pause_seconds=float(args.pause_seconds),
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)