#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_summary_csvs(shards_root: str) -> List[str]:
    """
    Recursively find all summary.csv files under shards_root.
    Handles download-artifact layouts like:
      out/shards/grid-fast-results-shard-0/summary.csv
      out/shards/grid-fast-results-shard-0/out/grid_fast_shard_0/summary.csv
      etc.
    """
    out: List[str] = []
    for root, _, files in os.walk(shards_root):
        for fn in files:
            if fn == "summary.csv":
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def find_best_params_jsons(shards_root: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(shards_root):
        for fn in files:
            if fn == "best_params.json":
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path} err={e!r}") from e


def try_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pick_best_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Choose a "best" row from merged summary.
    Priority:
      1) cagr_10y (desc) if exists, else cagr (desc)
      2) mdd_10y (desc; less negative is better) if exists, else mdd (desc)
      3) seed_multiple_10y (desc) if exists, else seed_multiple (desc)
    """
    if df is None or df.empty:
        return None

    # prefer 10y
    cagr_col = "cagr_10y" if "cagr_10y" in df.columns else ("cagr" if "cagr" in df.columns else None)
    mdd_col = "mdd_10y" if "mdd_10y" in df.columns else ("mdd" if "mdd" in df.columns else None)
    seed_col = "seed_multiple_10y" if "seed_multiple_10y" in df.columns else ("seed_multiple" if "seed_multiple" in df.columns else None)

    if cagr_col is None:
        # fallback: first row
        return df.iloc[0]

    work = df.copy()

    # numeric coercion
    for col in [cagr_col, mdd_col, seed_col]:
        if col and col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    sort_cols: List[str] = [cagr_col]
    ascending: List[bool] = [False]

    if mdd_col:
        sort_cols.append(mdd_col)
        ascending.append(False)  # -0.5 > -0.8 => "desc" is better

    if seed_col:
        sort_cols.append(seed_col)
        ascending.append(False)

    work = work.sort_values(sort_cols, ascending=ascending, na_position="last")
    return work.iloc[0]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-root", required=True, help="Root dir where shard artifacts were downloaded")
    ap.add_argument("--out", required=True, help="Output directory for merged results")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out)

    summary_paths = find_summary_csvs(args.shards_root)
    if not summary_paths:
        raise FileNotFoundError(f"No summary.csv found under shards_root={args.shards_root}")

    dfs: List[pd.DataFrame] = []
    for p in summary_paths:
        df = safe_read_csv(p)
        df["__source_summary_csv"] = p  # provenance
        dfs.append(df)

    # ✅ 핵심: union of columns 자동 보존 (pandas concat은 기본이 union)
    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # best row (10y CAGR 우선)
    best = pick_best_row(merged)

    merged_path = os.path.join(args.out, "summary_merged.csv")
    merged.to_csv(merged_path, index=False)

    # Also write a "best_params_merged.json"
    best_out = {
        "picked_by": "cagr_10y desc, then mdd_10y desc, then seed_multiple_10y desc (fallback to full columns)",
        "best_param_id": "",
        "best_params_json": "",
        "best_row": {},
    }
    if best is not None:
        best_out["best_param_id"] = str(best.get("param_id", ""))
        if "params_json" in best.index:
            best_out["best_params_json"] = str(best.get("params_json", ""))
        best_out["best_row"] = {k: (None if pd.isna(v) else v) for k, v in best.to_dict().items()}

    with open(os.path.join(args.out, "best_params_merged.json"), "w", encoding="utf-8") as f:
        json.dump(best_out, f, indent=2, ensure_ascii=False)

    # Optional: merge shard_info.json if present
    shard_infos = []
    for root, _, files in os.walk(args.shards_root):
        for fn in files:
            if fn == "shard_info.json":
                obj = try_read_json(os.path.join(root, fn))
                if obj:
                    obj["__source_shard_info"] = os.path.join(root, fn)
                    shard_infos.append(obj)

    if shard_infos:
        with open(os.path.join(args.out, "shards_info_merged.json"), "w", encoding="utf-8") as f:
            json.dump(shard_infos, f, indent=2, ensure_ascii=False)

    print(f"[DONE] merged summaries: {len(summary_paths)} files")
    print(f"[DONE] summary_merged -> {merged_path}")
    if best is not None:
        print(f"[DONE] best_param_id={best_out['best_param_id']}")
    else:
        print("[DONE] best_param_id=<none>")


if __name__ == "__main__":
    main()