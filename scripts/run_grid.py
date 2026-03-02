#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import product
from typing import Any, List, Tuple

import pandas as pd
import yaml

from src.core.data import download_prices_and_build_proxies
from src.core.meta import run_meta_portfolio
from src.core.metrics import compute_metrics


def deep_set(d: dict, key_path: str, value: Any) -> None:
    """
    Set nested dict key via dot-path, creating dicts if needed.
    Example: deep_set(cfg, "allocator.bull.trend", 0.8)
    """
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay onto base (non-mutating)."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_param_sets(grid: dict) -> List[dict]:
    """
    grid yaml is a nested dict where leaves are lists (scan values).
    Create list of param dicts (nested) corresponding to cartesian product.
    """
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

    param_sets: List[dict] = []
    for combo in product(*values_lists):
        p: dict = {}
        for k, v in zip(keys, combo):
            deep_set(p, k, v)
        param_sets.append(p)
    return param_sets


def short_param_id(param_dict: dict) -> str:
    s = json.dumps(param_dict, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config/default.yml")
    ap.add_argument("--grid", required=True, help="config/grid_fast.yml")
    ap.add_argument("--out", required=True, help="out/grid_fast")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out
    ensure_dir(out_dir)

    # --- load yaml ---
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.grid, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)

    param_sets = make_param_sets(grid_cfg)
    n = len(param_sets)
    if n <= 0:
        raise ValueError("Grid produced 0 param sets. Check grid.yml format (leaves should be lists/scalars).")

    # --- download prices + build proxies (once) ---
    prices = download_prices_and_build_proxies(base_cfg)
    prices_path = os.path.join(out_dir, "prices.csv")
    prices.to_csv(prices_path, index=True)

    # --- iterate grid ---
    best_cagr = None
    best_row = None
    best_params = None
    rows = []

    t0 = time.time()
    progress_every = max(1, n // 20)  # ~5% 단위
    for i, overlay in enumerate(param_sets, 1):
        cfg = deep_merge(base_cfg, overlay)

        # safety guard: allocator 값이 list로 남아있으면 grid 확장 실패
        if "allocator" in cfg:
            for bucket in ("bull", "bear", "crash"):
                if bucket in cfg["allocator"]:
                    for k in ("trend", "meanrev", "defensive"):
                        v = cfg["allocator"][bucket].get(k)
                        if isinstance(v, list):
                            raise ValueError(
                                f"Allocator value is list after merge: allocator.{bucket}.{k}={v}. "
                                f"Grid expansion failed; ensure scan values are only in grid.yml and expanded by run_grid."
                            )

        eq, choice_log, picks, holdings_daily, holdings_weekly = run_meta_portfolio(prices, cfg)
        met = compute_metrics(eq)

        # recent 10y metrics
        end = eq.index.max()
        start_10y = end - pd.DateOffset(years=10)
        eq_10y = eq.loc[eq.index >= start_10y]
        if len(eq_10y) > 10:
            met_10y = compute_metrics(eq_10y)
        else:
            met_10y = {"seed_multiple": math.nan, "cagr": math.nan, "mdd": math.nan}

        pid = short_param_id(overlay)

        row = {
            "param_id": pid,
            "seed_multiple": float(met["seed_multiple"]),
            "cagr": float(met["cagr"]),
            "mdd": float(met["mdd"]),
            "seed_multiple_10y": float(met_10y["seed_multiple"]),
            "cagr_10y": float(met_10y["cagr"]),
            "mdd_10y": float(met_10y["mdd"]),
        }
        # friendly percent columns
        row["cagr_pct"] = row["cagr"] * 100.0
        row["mdd_pct"] = row["mdd"] * 100.0
        row["cagr_10y_pct"] = row["cagr_10y"] * 100.0
        row["mdd_10y_pct"] = row["mdd_10y"] * 100.0

        rows.append(row)

        # best by CAGR
        if (best_cagr is None) or (row["cagr"] > best_cagr):
            best_cagr = row["cagr"]
            best_row = row
            best_params = overlay

            # write best-run artifacts continuously
            eq.to_csv(os.path.join(out_dir, "equity_curve.csv"), index=True)
            pd.DataFrame(choice_log).to_csv(os.path.join(out_dir, "engine_choice_log.csv"), index=False)
            picks.to_csv(os.path.join(out_dir, "picks_top2_weekly.csv"), index=False)
            holdings_daily.to_csv(os.path.join(out_dir, "holdings_daily.csv"), index=False)
            holdings_weekly.to_csv(os.path.join(out_dir, "holdings_weekly.csv"), index=False)
            with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"all": met, "recent_10y": met_10y}, f, indent=2)

        # progress + ETA (keep)
        elapsed = time.time() - t0
        per_iter = elapsed / i
        eta = per_iter * (n - i)
        if i == 1 or i % progress_every == 0 or i == n:
            print(
                f"[PROGRESS] {i}/{n} iter={per_iter:.2f}s elapsed={elapsed/60:.1f}m "
                f"eta={eta/60:.1f}m best_CAGR={(best_cagr*100):.2f}%"
            )

    # --- write summary + best_params ---
    summary = pd.DataFrame(rows).sort_values("cagr", ascending=False)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[DONE] summary -> {summary_path}")

    if best_row is None or best_params is None:
        raise RuntimeError("No best result selected; unexpected state. Check run_meta_portfolio/compute_metrics outputs.")

    best_params_path = os.path.join(out_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({"param_id": best_row["param_id"], "overlay": best_params}, f, indent=2)
    print(f"[DONE] best_params -> {best_params_path}")
    print(f"[DONE] best_CAGR={(best_cagr*100):.2f}% out={out_dir}")


if __name__ == "__main__":
    main()