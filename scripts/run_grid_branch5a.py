from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_branch5a import run_one


def ensure_list(v):
    if isinstance(v, list):
        return v
    return [v]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-csv", required=True, help="prices.csv 경로")
    parser.add_argument("--grid-yaml", required=True, help="grid 설정 yml")
    parser.add_argument("--out-dir", required=True, help="출력 폴더")
    parser.add_argument("--buy-cost", type=float, default=0.0005)
    parser.add_argument("--sell-cost", type=float, default=0.0005)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    cfg = yaml.safe_load(Path(args.grid_yaml).read_text(encoding="utf-8"))

    lookbacks = ensure_list(cfg.get("lookback", [133]))
    rebalances = ensure_list(cfg.get("rebalance", ["weekly"]))
    top1_weights = ensure_list(cfg.get("top1_weight", [0.80]))

    all_summaries = []

    run_id = 0
    for lookback, rebalance, top1_weight in itertools.product(
        lookbacks, rebalances, top1_weights
    ):
        run_id += 1
        run_name = f"lb_{lookback}__reb_{rebalance}__w1_{str(top1_weight).replace('.', 'p')}"
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        equity, holdings_df, rebalance_df, summary_row = run_one(
            prices=prices,
            lookback=int(lookback),
            rebalance=str(rebalance),
            top1_weight=float(top1_weight),
            buy_cost=float(args.buy_cost),
            sell_cost=float(args.sell_cost),
        )

        equity.to_csv(run_dir / "equity_curve.csv", header=True)
        holdings_df.to_csv(run_dir / "holdings_daily.csv", index=False)
        rebalance_df.to_csv(run_dir / "rebalance_log.csv", index=False)
        pd.DataFrame([summary_row]).to_csv(run_dir / "summary.csv", index=False)

        metrics_json = {
            "run_name": run_name,
            "params": {
                "lookback": int(lookback),
                "rebalance": str(rebalance),
                "top1_weight": float(top1_weight),
                "top2_weight": float(1.0 - float(top1_weight)),
                "buy_cost": float(args.buy_cost),
                "sell_cost": float(args.sell_cost),
            },
            "metrics": summary_row,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        all_summaries.append(summary_row)

    merged = pd.DataFrame(all_summaries)
    merged = merged.sort_values(
        by=["cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, True],
    )
    merged.to_csv(out_dir / "summary_merged.csv", index=False)

    if not merged.empty:
        best = merged.iloc[0].to_dict()
        (out_dir / "best.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"done: {run_id} runs")
    print(f"merged summary: {out_dir / 'summary_merged.csv'}")


if __name__ == "__main__":
    main()