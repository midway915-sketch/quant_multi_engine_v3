#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import glob
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="glob for shard outputs, e.g. out/grid_fast_shard_*")
    ap.add_argument("--out", required=True, help="output dir, e.g. out/grid_fast_merged")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    shard_dirs = sorted(glob.glob(args.inputs))
    if not shard_dirs:
        raise ValueError(f"No shard dirs matched: {args.inputs}")

    summaries = []
    best_candidates = []

    for d in shard_dirs:
        s_path = os.path.join(d, "summary.csv")
        if os.path.exists(s_path):
            df = pd.read_csv(s_path)
            df["shard_dir"] = os.path.basename(d)
            summaries.append(df)

        bp_path = os.path.join(d, "best_params.json")
        if os.path.exists(bp_path):
            with open(bp_path, "r", encoding="utf-8") as f:
                best_candidates.append(json.load(f))

    if not summaries:
        raise ValueError("No summary.csv found in shard dirs.")

    merged = pd.concat(summaries, ignore_index=True)
    merged = merged.sort_values("cagr", ascending=False)

    merged_path = os.path.join(args.out, "summary.csv")
    merged.to_csv(merged_path, index=False)

    # pick global best by max CAGR from merged summary (authoritative)
    best_row = merged.iloc[0].to_dict()
    best_overlay = None

    # try to recover overlay from params_json
    if "params_json" in best_row and isinstance(best_row["params_json"], str) and best_row["params_json"]:
        best_overlay = json.loads(best_row["params_json"])

    best_out = {
        "param_id": best_row.get("param_id"),
        "cagr": best_row.get("cagr"),
        "mdd": best_row.get("mdd"),
        "seed_multiple": best_row.get("seed_multiple"),
        "overlay": best_overlay,
    }

    best_path = os.path.join(args.out, "best_params.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_out, f, indent=2, ensure_ascii=False)

    print(f"[DONE] merged summary -> {merged_path}")
    print(f"[DONE] merged best_params -> {best_path}")


if __name__ == "__main__":
    main()