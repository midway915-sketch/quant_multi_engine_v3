from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def rank_summary(
    summary: pd.DataFrame,
    method: str,
    balanced_mdd_weight: float,
    balanced_recovery_weight: float,
) -> pd.DataFrame:
    method = str(method).lower().strip()
    df = summary.copy()

    required = {"cagr", "mdd", "max_recovery_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary csv missing required columns: {sorted(missing)}")

    if method == "cagr":
        return df.sort_values(
            by=["cagr", "mdd", "max_recovery_days"],
            ascending=[False, False, True],
        )

    if method == "recovery":
        return df.sort_values(
            by=["max_recovery_days", "mdd", "cagr"],
            ascending=[True, False, False],
        )

    if method == "balanced":
        df["score_balanced"] = (
            df["cagr"].astype(float)
            - float(balanced_mdd_weight) * df["mdd"].astype(float).abs()
            - float(balanced_recovery_weight) * df["max_recovery_days"].astype(float)
        )
        return df.sort_values(
            by=["score_balanced", "cagr", "mdd", "max_recovery_days"],
            ascending=[False, False, False, True],
        )

    raise ValueError(f"unsupported method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--equity-filename", default="equity_curve.csv")
    parser.add_argument("--method", default="cagr", choices=["cagr", "recovery", "balanced"])
    parser.add_argument("--balanced-mdd-weight", type=float, default=0.35)
    parser.add_argument("--balanced-recovery-weight", type=float, default=0.00015)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise ValueError("summary csv is empty")

    ranked = rank_summary(
        summary=summary,
        method=args.method,
        balanced_mdd_weight=args.balanced_mdd_weight,
        balanced_recovery_weight=args.balanced_recovery_weight,
    )
    best = ranked.iloc[0].to_dict()

    lookback = int(best["lookback"])
    rebalance = str(best["rebalance"])
    top1_weight = float(best["top1_weight"])

    run_name = f"lb_{lookback}__reb_{rebalance}__w1_{str(top1_weight).replace('.', 'p')}"
    equity_path = Path(args.runs_root) / run_name / args.equity_filename

    if not equity_path.exists():
        raise FileNotFoundError(f"best equity path not found: {equity_path}")

    payload = {
        "selection_method": args.method,
        "balanced_mdd_weight": float(args.balanced_mdd_weight),
        "balanced_recovery_weight": float(args.balanced_recovery_weight),
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