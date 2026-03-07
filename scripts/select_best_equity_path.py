from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--equity-filename", default="equity_curve.csv")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise ValueError("summary csv is empty")

    best = summary.sort_values(
        by=["cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()

    lookback = int(best["lookback"])
    rebalance = str(best["rebalance"])
    top1_weight = float(best["top1_weight"])

    run_name = f"lb_{lookback}__reb_{rebalance}__w1_{str(top1_weight).replace('.', 'p')}"
    equity_path = Path(args.runs_root) / run_name / args.equity_filename

    if not equity_path.exists():
        raise FileNotFoundError(f"best equity path not found: {equity_path}")

    payload = {
        "best_params": best,
        "run_name": run_name,
        "equity_path": str(equity_path),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()