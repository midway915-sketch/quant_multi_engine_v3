from __future__ import annotations
import argparse
import json
import os
import time
import itertools
import yaml
import pandas as pd

from src.core.data import download_prices_and_build_proxies
from src.core.utils import deep_set, flatten_grid, deep_copy
from src.core.meta import run_meta_portfolio
from src.core.metrics import compute_metrics


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _validate_weights(cfg) -> bool:
    for k in ["bull", "bear", "crash"]:
        w = cfg["allocator"][k]
        s = float(w["trend"]) + float(w["meanrev"]) + float(w["defensive"])
        if abs(s - 1.0) > 1e-9:
            return False
        if min(w["trend"], w["meanrev"], w["defensive"]) < 0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    base_cfg = load_yaml(args.config)
    grid = load_yaml(args.grid)

    t0 = time.time()
    prices = download_prices_and_build_proxies(base_cfg)
    prices.to_csv(os.path.join(args.out, "prices.csv"))
    print(f"[INFO] prices shape={prices.shape} ({time.time()-t0:.1f}s)")

    grid_items = flatten_grid(grid)
    keys = list(grid_items.keys())
    values = [grid_items[k] for k in keys]
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"[INFO] grid keys={len(keys)} combos={total}")

    rows = []
    best = None
    best_cagr = -1e18

    started = time.time()

    for i, combo in enumerate(combos, start=1):
        cfg = deep_copy(base_cfg)
        for k, v in zip(keys, combo):
            deep_set(cfg, k, v)

        # auto-fill implicit weights (meanrev/trend) from 2 keys if grid only sets 2
        for bucket in ["bull", "bear"]:
            tr = float(cfg["allocator"][bucket]["trend"])
            df = float(cfg["allocator"][bucket]["defensive"])
            cfg["allocator"][bucket]["meanrev"] = 1.0 - (tr + df)

        mr = float(cfg["allocator"]["crash"]["meanrev"])
        df = float(cfg["allocator"]["crash"]["defensive"])
        cfg["allocator"]["crash"]["trend"] = 1.0 - (mr + df)

        if not _validate_weights(cfg):
            continue

        t1 = time.time()
        eq, choice_log, picks = run_meta_portfolio(prices, cfg)
        met = compute_metrics(eq)

        # summary row
        rows.append({
            "idx": i,
            "start": met["start"],
            "end": met["end"],
            "years": met["years"],
            "seed_multiple": met["seed_multiple"],
            "cagr": met["cagr"],
            "mdd": met["mdd"],

            "start_10y": met["start_10y"],
            "end_10y": met["end_10y"],
            "years_10y": met["years_10y"],
            "seed_multiple_10y": met["seed_multiple_10y"],
            "cagr_10y": met["cagr_10y"],
            "mdd_10y": met["mdd_10y"],

            "params": json.dumps({k: v for k, v in zip(keys, combo)}, ensure_ascii=False),
        })

        cagr = float(met["cagr"])
        if cagr > best_cagr:
            best_cagr = cagr
            best = {
                "params": {k: v for k, v in zip(keys, combo)},
                "metrics": met,
                "equity_curve": eq,
                "engine_choice_log": choice_log,
                "picks_top2_weekly": picks,
            }

        elapsed = time.time() - started
        avg = elapsed / i
        eta = avg * (total - i)

        print(
            f"[PROGRESS] {i}/{total} iter={time.time()-t1:.2f}s "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m best_CAGR={best_cagr*100:.2f}%"
        )

    summary = pd.DataFrame(rows).sort_values("cagr", ascending=False)
    summary_path = os.path.join(args.out, "summary.csv")
    summary.to_csv(summary_path, index=False)

    if best is None:
        raise RuntimeError("No valid runs (weights validation failed?)")

    save_json(os.path.join(args.out, "best_params.json"), best["params"])
    save_json(os.path.join(args.out, "metrics.json"), best["metrics"])
    best["equity_curve"].to_csv(os.path.join(args.out, "equity_curve.csv"), header=["equity"])

    pd.DataFrame(best["engine_choice_log"]).to_csv(os.path.join(args.out, "engine_choice_log.csv"), index=False)

    if isinstance(best["picks_top2_weekly"], pd.DataFrame) and not best["picks_top2_weekly"].empty:
        best["picks_top2_weekly"].to_csv(os.path.join(args.out, "picks_top2_weekly.csv"), index=False)

    print(f"[DONE] summary -> {summary_path}")
    print(f"[DONE] best_CAGR={best_cagr*100:.2f}% out={args.out}")


if __name__ == "__main__":
    main()