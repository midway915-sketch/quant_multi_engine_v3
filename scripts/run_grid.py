#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import product
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from src.core.data import download_prices_and_build_proxies
from src.core.meta import run_meta_portfolio
from src.core.metrics import compute_metrics


def deep_set(d: dict, key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_param_sets(grid: dict) -> List[dict]:
    flat: List[Tuple[str, List[Any]]] = []

    def walk(node: Any, prefix: str = ""):
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{prefix}.{k}" if prefix else k
                walk(v, p)
        else:
            if isinstance(node, list):
                flat.append((prefix, node))
            else:
                flat.append((prefix, [node]))

    walk(grid)

    keys = [k for k, _ in flat]
    values_lists = [vals for _, vals in flat]

    param_sets: List[dict] = []
    for combo in product(*values_lists):
        p: dict = {}
        for k, v in zip(keys, combo):
            deep_set(p, k, v)
        param_sets.append(p)
    return param_sets


def short_param_id(param_dict: dict) -> str:
    s = json.dumps(param_dict, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def flatten_dict(d: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dict into dot paths.
    Example: {"a":{"b":1}} -> {"a.b":1}
    Lists are JSON-stringified to keep summary.csv rectangular.
    """
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, p))
    elif isinstance(d, list):
        out[prefix] = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
    else:
        out[prefix] = d
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config/default.yml")
    ap.add_argument("--grid", required=True, help="config/grid_fast.yml")
    ap.add_argument("--out", required=True, help="out/grid_fast")

    # ✅ shard options (for GitHub Actions matrix parallelism)
    ap.add_argument("--shard-index", type=int, default=0, help="0-based shard index")
    ap.add_argument("--shard-count", type=int, default=1, help="number of shards (>=1)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("--shard-index must be in [0, shard-count)")

    out_dir = args.out
    ensure_dir(out_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.grid, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)

    all_param_sets = make_param_sets(grid_cfg)
    n_all = len(all_param_sets)
    if n_all <= 0:
        raise ValueError("Grid produced 0 param sets. Check grid.yml format.")

    # ✅ stable shard split: index modulo shard_count
    param_sets = [p for j, p in enumerate(all_param_sets) if (j % args.shard_count) == args.shard_index]
    n = len(param_sets)
    if n <= 0:
        raise ValueError(
            f"This shard has 0 param sets (shard_index={args.shard_index}, shard_count={args.shard_count}, total={n_all})."
        )

    # write shard info (helps debugging + aggregation sanity)
    with open(os.path.join(out_dir, "shard_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "total_param_sets