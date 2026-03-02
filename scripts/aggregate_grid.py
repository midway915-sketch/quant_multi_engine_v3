from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-root", required=True, help="e.g. out/shards")
    ap.add_argument("--out", required=True, help="e.g. out/grid_fast_merged")
    return ap.parse_args()


def find_summary_files(shards_root: str) -> List[str]:
    """
    download-artifact 구조가 어떤 형태든 summary.csv를 전부 긁어모으기.
    보통:
      out/shards/grid-fast-results-shard-0/**/summary.csv
    """
    hits: List[str] = []
    for root, _, files in os.walk(shards_root):
        for f in files:
            if f == "summary.csv":
                hits.append(os.path.join(root, f))
    hits.sort()
    return hits


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary_files = find_summary_files(args.shards_root)
    if not summary_files:
        raise ValueError(f"No summary.csv found under shards-root={args.shards_root}")

    dfs = []
    for path in summary_files:
        df = pd.read_csv(path)
        df["source_summary"] = path
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("cagr", ascending=False)

    out_summary = os.path.join(args.out, "summary.csv")
    merged.to_csv(out_summary, index=False)

    # global best = top row by cagr (same rule as run_grid)
    best = merged.iloc[0].to_dict()

    overlay = None
    if isinstance(best.get("params_json"), str) and best["params_json"]:
        try:
            overlay = json.loads(best["params_json"])
        except Exception:
            overlay = None

    best_out = {
        "param_id": best.get("param_id"),
        "seed_multiple": best.get("seed_multiple"),
        "cagr": best.get("cagr"),
        "mdd": best.get("mdd"),
        "seed_multiple_10y": best.get("seed_multiple_10y"),
        "cagr_10y": best.get("cagr_10y"),
        "mdd_10y": best.get("mdd_10y"),
        "overlay": overlay,
        "source_summary": best.get("source_summary"),
    }

    out_best = os.path.join(args.out, "best_params.json")
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(best_out, f, indent=2, ensure_ascii=False)

    print(f"[DONE] merged summary -> {out_summary}")
    print(f"[DONE] merged best_params -> {out_best}")


if __name__ == "__main__":
    main()