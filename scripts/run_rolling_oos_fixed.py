#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import yaml

from src.core.data import download_prices_and_build_proxies
from src.core.meta import run_meta_portfolio
from src.core.metrics import compute_metrics


@dataclass
class Window:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DEFAULT_WINDOWS: List[Window] = [
    Window("2000-01-01", "2009-12-31", "2010-01-01", "2013-12-31"),
    Window("2000-01-01", "2013-12-31", "2014-01-01", "2017-12-31"),
    Window("2000-01-01", "2017-12-31", "2018-01-01", "2021-12-31"),
    Window("2000-01-01", "2021-12-31", "2022-01-01", "2026-12-31"),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config/default.yml (fixed params live here)")
    ap.add_argument("--out", required=True, help="out/rolling_oos_fixed")
    ap.add_argument(
        "--windows-yml",
        default="",
        help="Optional YAML file that defines windows; if empty, uses built-in defaults",
    )
    ap.add_argument(
        "--prices-csv",
        default="",
        help="Optional path to prebuilt prices.csv to skip download (still enforces start/end)",
    )
    return ap.parse_args()


def load_windows(path: str) -> List[Window]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, dict) or "windows" not in obj:
        raise ValueError("windows-yml must be a dict with key: windows")

    windows = []
    for w in obj["windows"]:
        windows.append(
            Window(
                train_start=str(w["train_start"]),
                train_end=str(w["train_end"]),
                test_start=str(w["test_start"]),
                test_end=str(w["test_end"]),
            )
        )
    if not windows:
        raise ValueError("windows-yml has 0 windows")
    return windows


def enforce_date_range(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data_cfg = (cfg.get("data", {}) or {})
    start = data_cfg.get("start", "")
    end = data_cfg.get("end", "")
    px = prices
    if start:
        px = px.loc[px.index >= pd.Timestamp(start)]
    if end:
        px = px.loc[px.index <= pd.Timestamp(end)]
    return px


def rebase_equity(eq: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    """
    Slice equity [start,end], then rebase to 1.0 at start (using start day's equity as base).
    Returns None if window cannot be formed.
    """
    if eq.empty:
        return None
    seg = eq.loc[(eq.index >= start) & (eq.index <= end)]
    if seg.empty:
        return None
    base = float(seg.iloc[0])
    if base <= 0 or pd.isna(base):
        return None
    return seg / base


def main():
    args = parse_args()
    ensure_dir(args.out)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    windows = load_windows(args.windows_yml) if args.windows_yml else DEFAULT_WINDOWS

    # prices
    if args.prices_csv:
        if not os.path.exists(args.prices_csv):
            raise FileNotFoundError(f"--prices-csv not found: {args.prices_csv}")
        prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    else:
        prices = download_prices_and_build_proxies(cfg)

    prices = enforce_date_range(prices, cfg)
    if prices.empty:
        raise ValueError("prices empty after enforcing config date range")

    prices.to_csv(os.path.join(args.out, "prices.csv"), index=True)

    # run full backtest once (lookahead-free + costs handled inside src.core.meta)
    equity, engine_choice_log, picks_df, holdings_daily, holdings_weekly = run_meta_portfolio(prices, cfg)

    # persist main artifacts (so you can inspect)
    equity.to_csv(os.path.join(args.out, "equity_curve_full.csv"), index=True)
    pd.DataFrame(engine_choice_log).to_csv(os.path.join(args.out, "engine_choice_log_full.csv"), index=False)
    picks_df.to_csv(os.path.join(args.out, "picks_top2_weekly_full.csv"), index=False)
    holdings_daily.to_csv(os.path.join(args.out, "holdings_daily_full.csv"), index=False)
    holdings_weekly.to_csv(os.path.join(args.out, "holdings_weekly_full.csv"), index=False)

    # full metrics
    full_met = compute_metrics(equity)
    with open(os.path.join(args.out, "metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump(full_met, f, indent=2, ensure_ascii=False)

    # rolling OOS windows
    rows = []
    for w in windows:
        ts = pd.Timestamp(w.test_start)
        te = pd.Timestamp(w.test_end)

        seg_eq = rebase_equity(equity, ts, te)
        if seg_eq is None or len(seg_eq) < 20:
            rows.append(
                {
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "n_days": 0,
                    "seed_multiple": float("nan"),
                    "cagr": float("nan"),
                    "mdd": float("nan"),
                    "cagr_pct": float("nan"),
                    "mdd_pct": float("nan"),
                    "note": "insufficient data in window",
                }
            )
            continue

        met = compute_metrics(seg_eq)

        out_curve_path = os.path.join(
            args.out,
            f"equity_oos_{w.test_start.replace('-','')}_{w.test_end.replace('-','')}.csv",
        )
        seg_eq.to_csv(out_curve_path, index=True)

        row = {
            "train_start": w.train_start,
            "train_end": w.train_end,
            "test_start": w.test_start,
            "test_end": w.test_end,
            "n_days": int(len(seg_eq)),
            "seed_multiple": float(met["seed_multiple"]),
            "cagr": float(met["cagr"]),
            "mdd": float(met["mdd"]),
            "cagr_pct": float(met["cagr"]) * 100.0,
            "mdd_pct": float(met["mdd"]) * 100.0,
            "note": "",
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # add a simple robustness score (optional): prefer higher CAGR, less negative MDD
    # score = cagr - 0.5*abs(mdd)
    if not summary.empty:
        summary["robust_score"] = summary["cagr"] - 0.5 * summary["mdd"].abs()

    summary_path = os.path.join(args.out, "rolling_oos_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[DONE] full_metrics: seed={full_met['seed_multiple']:.3f} cagr={full_met['cagr']*100:.2f}% mdd={full_met['mdd']*100:.2f}%")
    print(f"[DONE] rolling_oos_summary -> {summary_path}")


if __name__ == "__main__":
    main()