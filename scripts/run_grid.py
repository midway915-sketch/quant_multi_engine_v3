
import argparse
import itertools
import os
import time
import json
import yaml
import pandas as pd

from src.core.data import download_prices
from src.core.utils import deep_set
from src.core.strategy import run_backtest
from src.core.metrics import cagr, mdd

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config yml")
    ap.add_argument("--grid", required=True, help="Grid config yml (dot-notation keys)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--save-picks", action="store_true", help="Save picks_top2_weekly.csv for best run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    grid = load_yaml(args.grid)

    os.makedirs(args.out, exist_ok=True)

    # download once
    t0 = time.time()
    prices = download_prices(cfg)
    prices.to_csv(os.path.join(args.out, "prices.csv"))
    print(f"[INFO] Downloaded prices: {prices.shape} in {time.time()-t0:.1f}s")

    # build param sets
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = list(itertools.product(*values))
    total = len(combos)

    results = []
    best = None

    start_time = time.time()
    for idx, combo in enumerate(combos, start=1):
        run_cfg = json.loads(json.dumps(cfg))  # deep copy via json
        for k, v in zip(keys, combo):
            deep_set(run_cfg, k, v)

        iter_start = time.time()
        curve = run_backtest(prices, run_cfg, save_picks_path=None)
        iter_cagr = cagr(curve)
        iter_mdd = mdd(curve)

        years = (curve.index[-1] - curve.index[0]).days / 365.25
        final_equity = float(curve.iloc[-1])

        results.append({
            "param_id": f"{idx:05d}",
            "CAGR": float(iter_cagr),
            "MDD": float(iter_mdd),
            "years": float(years),
            "final_equity": final_equity,
            "params": {k: v for k, v in zip(keys, combo)}
        })

        if best is None or iter_cagr > best["CAGR"]:
            best = results[-1]

        # progress + ETA
        elapsed = time.time() - start_time
        avg = elapsed / idx
        remaining = avg * (total - idx)
        iter_time = time.time() - iter_start
        print(f"[PROGRESS] {idx}/{total} | iter {iter_time:.2f}s | elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m | best_CAGR {best['CAGR']:.4f}")

    df = pd.DataFrame(results).sort_values("CAGR", ascending=False)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # save best params
    with open(os.path.join(args.out, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    # save picks for best run if requested
    if args.save_picks and best is not None:
        best_cfg = json.loads(json.dumps(cfg))
        for k, v in best["params"].items():
            deep_set(best_cfg, k, v)
        picks_path = os.path.join(args.out, "picks_top2_weekly.csv")
        curve = run_backtest(prices, best_cfg, save_picks_path=picks_path)
        curve.to_csv(os.path.join(args.out, "equity_curve.csv"))
        metrics = {
            "CAGR": float(cagr(curve)),
            "MDD": float(mdd(curve)),
            "start": str(curve.index[0].date()),
            "end": str(curve.index[-1].date()),
            "years": float((curve.index[-1] - curve.index[0]).days / 365.25),
            "final_equity": float(curve.iloc[-1]),
            "best_params": best["params"]
        }
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("[DONE] Grid complete. Outputs written to:", args.out)

if __name__ == "__main__":
    main()
