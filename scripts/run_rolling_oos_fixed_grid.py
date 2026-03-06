#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def deep_set(d: dict, key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_param_sets(grid: dict) -> List[dict]:
    """
    Supports two formats:
      1) combos: [ {overlay1}, {overlay2}, ... ]   # exact combos
      2) nested dict with list leaves -> cartesian product (grid scan)
    """
    if isinstance(grid, dict) and "combos" in grid:
        combos = grid.get("combos", [])
        if not isinstance(combos, list) or not combos:
            raise ValueError("grid.yml has 'combos' but it's empty or not a list")
        if not all(isinstance(x, dict) for x in combos):
            raise ValueError("grid.yml combos must be a list of dict overlays")
        return combos

    flat: List[Tuple[str, List[Any]]] = []

    def walk(node: Any, prefix: str = ""):
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{prefix}.{k}" if prefix else k
                walk(v, p)
        else:
            if isinstance(node, list):
                flat.append((prefix, node))
            else:
                flat.append((prefix, [node]))

    walk(grid)

    keys = [k for k, _ in flat]
    values_lists = [vals for _, vals in flat]

    from itertools import product

    param_sets: List[dict] = []
    for combo in product(*values_lists):
        p: dict = {}
        for k, v in zip(keys, combo):
            deep_set(p, k, v)
        param_sets.append(p)
    return param_sets


def short_param_id(overlay: dict) -> str:
    s = json.dumps(overlay, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def max_recovery_days(eq: pd.Series) -> Tuple[float, float]:
    """
    Max time-to-recover: peak -> first time equity >= that peak again.
    If never recovers, uses last date.
    Returns (days, years).
    """
    if eq is None or eq.empty:
        return (math.nan, math.nan)

    s = eq.dropna()
    if s.empty:
        return (math.nan, math.nan)

    idx = s.index
    vals = s.values

    peak_val = float(vals[0])
    peak_date = idx[0]
    in_dd = False
    max_days = 0.0

    for i in range(1, len(vals)):
        v = float(vals[i])
        d = idx[i]

        if v >= peak_val:
            if in_dd:
                days = float((d - peak_date).days)
                if days > max_days:
                    max_days = days
                in_dd = False
            peak_val = v
            peak_date = d
        else:
            in_dd = True

    if in_dd:
        days = float((idx[-1] - peak_date).days)
        if days > max_days:
            max_days = days

    return (max_days, max_days / 365.25)


def rebase_equity(eq: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    seg = eq.loc[(eq.index >= start) & (eq.index <= end)]
    if seg.empty:
        return None
    base = float(seg.iloc[0])
    if base <= 0 or pd.isna(base):
        return None
    return seg / base


def load_windows(path: str) -> List[Window]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or "windows" not in obj:
        raise ValueError("windows-yml must be a dict with key: windows")
    windows: List[Window] = []
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
    start = str(data_cfg.get("start", "") or "")
    end = str(data_cfg.get("end", "") or "")
    px = prices
    if start:
        px = px.loc[px.index >= pd.Timestamp(start)]
    if end:
        px = px.loc[px.index <= pd.Timestamp(end)]
    return px


def stitch_oos_curves(segments: List[pd.Series]) -> pd.Series:
    """
    Stitch rebased segments sequentially: first starts at 1,
    next segment is multiplied by last equity of previous, etc.
    """
    out_parts = []
    level = 1.0
    for seg in segments:
        if seg is None or seg.empty:
            continue
        stitched = seg * level
        level = float(stitched.iloc[-1])
        out_parts.append(stitched)
    if not out_parts:
        return pd.Series(dtype=float)
    oos = pd.concat(out_parts).sort_index()
    oos = oos[~oos.index.duplicated(keep="last")]
    return oos


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config yml (fixed settings)")
    ap.add_argument("--grid", required=True, help="Grid yml (scan space OR combos list)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--windows-yml", default="", help="Optional windows.yml; if empty uses defaults")
    ap.add_argument("--prices-csv", default="", help="Optional path to prebuilt prices.csv to skip download")
    ap.add_argument("--rank-by", default="oos_cagr", help="oos_cagr|oos_seed|oos_score (default oos_cagr)")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out)

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.grid, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)

    windows = load_windows(args.windows_yml) if args.windows_yml else DEFAULT_WINDOWS
    param_sets = make_param_sets(grid_cfg)
    n = len(param_sets)
    if n <= 0:
        raise ValueError("Grid produced 0 param sets.")

    # prices (once)
    if args.prices_csv:
        if not os.path.exists(args.prices_csv):
            raise FileNotFoundError(f"--prices-csv not found: {args.prices_csv}")
        prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    else:
        prices = download_prices_and_build_proxies(base_cfg)

    prices = enforce_date_range(prices, base_cfg)
    if prices.empty:
        raise ValueError("prices empty after enforcing config date range")

    prices.to_csv(os.path.join(args.out, "prices.csv"), index=True)

    rows = []
    best_row = None

    t0 = time.time()
    progress_every = max(1, n // 25)

    for i, overlay in enumerate(param_sets, 1):
        cfg = deep_merge(base_cfg, overlay)
        pid = short_param_id(overlay)
        run_dir = os.path.join(args.out, f"param_{pid}")
        ensure_dir(run_dir)

        # run full backtest once
        equity, engine_choice_log, picks_df, holdings_daily, holdings_weekly = run_meta_portfolio(prices, cfg)

        equity.to_csv(os.path.join(run_dir, "equity_curve_full.csv"), index=True)
        pd.DataFrame(engine_choice_log).to_csv(os.path.join(run_dir, "engine_choice_log_full.csv"), index=False)
        picks_df.to_csv(os.path.join(run_dir, "picks_top2_weekly_full.csv"), index=False)
        holdings_daily.to_csv(os.path.join(run_dir, "holdings_daily_full.csv"), index=False)
        holdings_weekly.to_csv(os.path.join(run_dir, "holdings_weekly_full.csv"), index=False)

        full_met = compute_metrics(equity)
        full_rec_days, full_rec_years = max_recovery_days(equity)
        full_met["max_recovery_days"] = float(full_rec_days)
        full_met["max_recovery_years"] = float(full_rec_years)

        with open(os.path.join(run_dir, "metrics_full.json"), "w", encoding="utf-8") as f:
            json.dump(full_met, f, indent=2, ensure_ascii=False)

        # window OOS metrics
        win_rows = []
        oos_segments = []
        for w in windows:
            ts = pd.Timestamp(w.test_start)
            te = pd.Timestamp(w.test_end)
            seg = rebase_equity(equity, ts, te)

            if seg is None or len(seg) < 20:
                win_rows.append(
                    {
                        "param_id": pid,
                        "train_start": w.train_start,
                        "train_end": w.train_end,
                        "test_start": w.test_start,
                        "test_end": w.test_end,
                        "n_days": 0,
                        "seed_multiple": math.nan,
                        "cagr": math.nan,
                        "mdd": math.nan,
                        "max_recovery_years": math.nan,
                        "note": "insufficient data",
                    }
                )
                continue

            met = compute_metrics(seg)
            rec_days, rec_years = max_recovery_days(seg)
            met["max_recovery_years"] = float(rec_years)

            seg.to_csv(os.path.join(run_dir, f"equity_oos_{w.test_start.replace('-','')}_{w.test_end.replace('-','')}.csv"), index=True)

            win_rows.append(
                {
                    "param_id": pid,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "n_days": int(len(seg)),
                    "seed_multiple": float(met["seed_multiple"]),
                    "cagr": float(met["cagr"]),
                    "mdd": float(met["mdd"]),
                    "max_recovery_years": float(rec_years),
                    "note": "",
                }
            )
            oos_segments.append(seg)

        win_df = pd.DataFrame(win_rows)
        win_df.to_csv(os.path.join(run_dir, "rolling_oos_summary.csv"), index=False)

        # stitched OOS (single curve across windows)
        oos = stitch_oos_curves(oos_segments)
        if not oos.empty:
            oos.to_csv(os.path.join(run_dir, "equity_curve_oos_stitched.csv"), index=True)
            oos_met = compute_metrics(oos / float(oos.iloc[0]))
            oos_rec_days, oos_rec_years = max_recovery_days(oos / float(oos.iloc[0]))
        else:
            oos_met = {"seed_multiple": math.nan, "cagr": math.nan, "mdd": math.nan}
            oos_rec_years = math.nan

        # simple robustness score
        # score = oos_cagr - 0.5*abs(oos_mdd)
        if pd.notna(oos_met.get("cagr")) and pd.notna(oos_met.get("mdd")):
            oos_score = float(oos_met["cagr"]) - 0.5 * abs(float(oos_met["mdd"]))
        else:
            oos_score = math.nan

        row = {
            "param_id": pid,
            "full_seed_multiple": float(full_met["seed_multiple"]),
            "full_cagr": float(full_met["cagr"]),
            "full_mdd": float(full_met["mdd"]),
            "full_max_recovery_years": float(full_met["max_recovery_years"]),
            "oos_seed_multiple": float(oos_met["seed_multiple"]),
            "oos_cagr": float(oos_met["cagr"]),
            "oos_mdd": float(oos_met["mdd"]),
            "oos_max_recovery_years": float(oos_rec_years),
            "oos_score": float(oos_score),
            "params_json": json.dumps(overlay, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        }

        # add quick percent columns
        row["full_cagr_pct"] = row["full_cagr"] * 100.0
        row["full_mdd_pct"] = row["full_mdd"] * 100.0
        row["oos_cagr_pct"] = row["oos_cagr"] * 100.0
        row["oos_mdd_pct"] = row["oos_mdd"] * 100.0

        rows.append(row)

        # ranking
        def better(a: dict, b: dict) -> bool:
            if b is None:
                return True
            key = args.rank_by.lower().strip()
            if key == "oos_seed":
                return float(a.get("oos_seed_multiple", math.nan)) > float(b.get("oos_seed_multiple", math.nan))
            if key == "oos_score":
                return float(a.get("oos_score", math.nan)) > float(b.get("oos_score", math.nan))
            # default: oos_cagr
            return float(a.get("oos_cagr", math.nan)) > float(b.get("oos_cagr", math.nan))

        if better(row, best_row):
            best_row = row

        # progress
        elapsed = time.time() - t0
        per = elapsed / i
        eta = per * (n - i)
        if i == 1 or i % progress_every == 0 or i == n:
            print(f"[PROGRESS] {i}/{n} per={per:.2f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m best={best_row['param_id'] if best_row else ''}")

    summary = pd.DataFrame(rows).sort_values("oos_cagr", ascending=False)
    summary_path = os.path.join(args.out, "wf_grid_summary.csv")
    summary.to_csv(summary_path, index=False)

    best_path = os.path.join(args.out, "best_params_wf_grid.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"rank_by": args.rank_by, "best": best_row}, f, indent=2, ensure_ascii=False)

    print(f"[DONE] wf_grid_summary -> {summary_path}")
    print(f"[DONE] best_params_wf_grid -> {best_path}")


if __name__ == "__main__":
    main()