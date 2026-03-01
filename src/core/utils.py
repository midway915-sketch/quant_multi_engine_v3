from __future__ import annotations
import copy


def deep_set(d: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_get(d: dict, dotted_key: str, default=None):
    parts = dotted_key.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def flatten_grid(grid: dict) -> dict:
    # grid yaml may already be flat; normalize to dict[str, list]
    out = {}
    for k, v in grid.items():
        if isinstance(v, list):
            out[k] = v
        else:
            out[k] = [v]
    return out


def deep_copy(obj):
    return copy.deepcopy(obj)