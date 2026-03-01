
import argparse
import os
import json
import yaml
from src.core.data import download_prices
from src.core.strategy import run_backtest
from src.core.metrics import cagr, mdd

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    os.makedirs(args.out, exist_ok=True)

    prices = download_prices(cfg)
    prices.to_csv(os.path.join(args.out, "prices.csv"))

    picks_path = os.path.join(args.out, "picks_top2_weekly.csv")
    curve = run_backtest(prices, cfg, save_picks_path=picks_path)
    curve.to_csv(os.path.join(args.out, "equity_curve.csv"))

    metrics = {
        "CAGR": float(cagr(curve)),
        "MDD": float(mdd(curve)),
        "start": str(curve.index[0].date()),
        "end": str(curve.index[-1].date()),
        "years": float((curve.index[-1] - curve.index[0]).days / 365.25),
        "final_equity": float(curve.iloc[-1])
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)

if __name__ == "__main__":
    main()
