from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_abs(x: pd.Series) -> pd.Series:
    return x.astype(float).abs()


def build_rankings(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    rank_by_cagr = df.sort_values(
        by=["cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out["rank_by_cagr"] = rank_by_cagr

    rank_by_recovery = df.sort_values(
        by=["max_recovery_days", "mdd", "cagr"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    out["rank_by_recovery"] = rank_by_recovery

    tmp = df.copy()
    tmp["score_balanced"] = (
        tmp["cagr"].astype(float)
        - 0.35 * _safe_abs(tmp["mdd"])
        - 0.00015 * tmp["max_recovery_days"].astype(float)
    )
    rank_balanced = tmp.sort_values(
        by=["score_balanced", "cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["rank_balanced"] = rank_balanced

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise ValueError("summary csv is empty")

    rankings = build_rankings(df)

    rankings["rank_by_cagr"].to_csv(out_dir / "rank_by_cagr.csv", index=False)
    rankings["rank_by_recovery"].to_csv(out_dir / "rank_by_recovery.csv", index=False)
    rankings["rank_balanced"].to_csv(out_dir / "rank_balanced.csv", index=False)

    best_payload = {
        "best_by_cagr": rankings["rank_by_cagr"].iloc[0].to_dict(),
        "best_by_recovery": rankings["rank_by_recovery"].iloc[0].to_dict(),
        "best_balanced": rankings["rank_balanced"].iloc[0].to_dict(),
    }
    (out_dir / "aggregate_best.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(best_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()