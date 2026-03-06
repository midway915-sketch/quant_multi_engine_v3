#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import yaml

from src.core.data import download_prices_and_build_proxies
from src.core.meta import run_meta_portfolio
from src.core.metrics import compute_metrics


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    """
    grid yaml is nested dict; leaves are lists -> cartesian product.
    """
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


def flatten_dict(d: Any, prefix: str = "") -> Dict[str, Any]:
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


def max_recovery_days(eq: pd.Series) -> Tuple[float, float]:
    """
    Max time-to-recover (peak -> first time equity exceeds that peak again).
    If never recovers, uses last date as end.
    Returns (days, years).
    """
    if eq is None or eq.empty:
        return (math.nan, math.nan)

    s = eq.dropna()
    if s.empty:
        return (math.nan, math.nan)

    idx = s.index
    vals = s.values

    peak_val = float(vals[0])
    peak_date = idx[0]
    in_dd = False
    max_days = 0.0

    for i in range(1, len(vals)):
        v = float(vals[i])
        d = idx[i]

        if v >= peak_val:
            if in_dd:
                days = float((d - peak_date).days)
                if days > max_days:
                    max_days = days
                in_dd = False
            peak_val = v
            peak_date = d
        else:
            in_dd = True

    if in_dd:
        days = float((idx[-1] - peak_date).days)
        if days > max_days:
            max_days = days

    return (max_days, max_days / 365.25)


def rebase(eq: pd.Series) -> pd.Series:
    if eq.empty:
        return eq
    base = float(eq.iloc[0])
    if base <= 0 or pd.isna(base):
        return eq
    return eq / base


def slice_with_warmup(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, warmup_days: int) -> pd.DataFrame:
    """
    Include a warmup window before start for indicators, but keep data up to end.
    Warmup is in calendar days (not trading days) to avoid missing holidays.
    """
    warm_start = start - pd.Timedelta(days=int(warmup_days))
    px = prices.loc[(prices.index >= warm_start) & (prices.index <= end)].copy()
    return px


def run_window(prices: pd.DataFrame, cfg: dict, start: pd.Timestamp, end: pd.Timestamp, warmup_days: int) -> Tuple[pd.Series, dict]:
    """
    Run strategy on [warmup..end], then return equity segment restricted to [start..end], rebased to 1.
    Also returns metrics dict for that segment + max_recovery.
    """
    px = slice_with_warmup(prices, start, end, warmup_days=warmup_days)
    if px.empty:
        return pd.Series(dtype=float), {}

    eq, *_ = run_meta_portfolio(px, cfg)
    seg = eq.loc[(eq.index >= start) & (eq.index <= end)]
    if seg.empty:
        return pd.Series(dtype=float), {}

    seg = rebase(seg)
    met = compute_metrics(seg)
    rec_days, rec_years = max_recovery_days(seg)
    met["max_recovery_days"] = float(rec_days)
    met["max_recovery_years"] = float(rec_years)
    return seg, met


def month_end_trading_days(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Monthly rebalance dates: last trading day of each month (based on available price index).
    """
    me = prices.resample("ME").last().index
    # ensure they exist in the actual index
    return prices.index[prices.index.isin(me)]


@dataclass
class WFConfig:
    train_years: int
    test_months: int
    warmup_days: int

    filter_mdd_min: float
    filter_max_recovery_years_max: float

    score_mode: str  # "cagr_only" | "cagr_minus_dd" | "cagr_then_dd_then_rec"
    score_dd_weight: float  # used if cagr_minus_dd

    top_k: int  # if >1, log top candidates; selection uses rank1 anyway


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config (fixed params)")
    ap.add_argument("--grid", required=True, help="Grid yml (scan params, e.g. soxx_gate)")
    ap.add_argument("--wf", required=True, help="WF config yml")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--prices-csv", default="", help="Optional prebuilt prices.csv (skip download)")
    return ap.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wf_cfg(path: str) -> WFConfig:
    obj = load_yaml(path)
    w = obj.get("wf", {}) or {}
    sel = obj.get("selection", {}) or {}
    filt = obj.get("filters", {}) or {}

    return WFConfig(
        train_years=int(w.get("train_years", 10)),
        test_months=int(w.get("test_months", 1)),
        warmup_days=int(w.get("warmup_days", 900)),
        filter_mdd_min=float(filt.get("mdd_min", -0.65)),
        filter_max_recovery_years_max=float(filt.get("max_recovery_years_max", 3.0)),
        score_mode=str(sel.get("score_mode", "cagr_then_dd_then_rec")).lower().strip(),
        score_dd_weight=float(sel.get("dd_weight", 0.5)),
        top_k=int(sel.get("top_k", 5)),
    )


def rank_candidates(rows: List[dict], wf_cfg: WFConfig) -> List[dict]:
    """
    Rank by configured rule, with optional filtering.
    rows must include keys: cagr, mdd, max_recovery_years
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return []

    # numeric coercion
    for c in ["cagr", "mdd", "max_recovery_years"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # filters
    df_f = df.copy()
    df_f = df_f[df_f["mdd"].notna() & df_f["cagr"].notna()]
    df_f = df_f[df_f["mdd"] >= float(wf_cfg.filter_mdd_min)]
    df_f = df_f[df_f["max_recovery_years"].notna() & (df_f["max_recovery_years"] <= float(wf_cfg.filter_max_recovery_years_max))]

    # if all filtered out, fall back to unfiltered
    use = df_f if not df_f.empty else df

    if wf_cfg.score_mode == "cagr_only":
        use = use.sort_values(["cagr"], ascending=[False], na_position="last")
    elif wf_cfg.score_mode == "cagr_minus_dd":
        # score = cagr - w*abs(mdd)
        w = float(wf_cfg.score_dd_weight)
        use["score"] = use["cagr"] - w * use["mdd"].abs()
        use = use.sort_values(["score", "cagr"], ascending=[False, False], na_position="last")
    else:
        # default: cagr desc, then mdd desc(less negative better), then recovery asc (shorter better)
        use = use.sort_values(["cagr", "mdd", "max_recovery_years"], ascending=[False, False, True], na_position="last")

    return use.to_dict(orient="records")


def main():
    args = parse_args()
    ensure_dir(args.out)

    base_cfg = load_yaml(args.config)
    grid_cfg = load_yaml(args.grid)
    wf_cfg = load_wf_cfg(args.wf)

    param_sets = make_param_sets(grid_cfg)
    if not param_sets:
        raise ValueError("Grid produced 0 param sets. Check grid.yml format.")

    # prices
    if args.prices_csv:
        if not os.path.exists(args.prices_csv):
            raise FileNotFoundError(f"--prices-csv not found: {args.prices_csv}")
        prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    else:
        prices = download_prices_and_build_proxies(base_cfg)

    if prices.empty:
        raise ValueError("prices empty")

    prices.to_csv(os.path.join(args.out, "prices.csv"), index=True)

    # monthly decision dates (last trading day)
    rebal_dates = month_end_trading_days(prices)
    if len(rebal_dates) < 24:
        raise ValueError(f"Not enough month-end trading days found: {len(rebal_dates)}")

    # OOS timeline: for each decision date d, apply selected params to next test_months months
    # Example: d=2020-01-31, test is 2020-02-01..2020-02-28 (end=month-end)
    # We will define test_start as next trading day after d (or next index date),
    # and test_end as the month-end after test_months months.
    idx = prices.index

    def next_trading_day(d: pd.Timestamp) -> Optional[pd.Timestamp]:
        loc = idx.get_indexer([d], method="pad")
        if loc[0] < 0:
            return None
        i = int(loc[0])
        if i >= len(idx) - 1:
            return None
        return idx[i + 1]

    # Build decision points where we can form both train and test
    plan = []
    for d in rebal_dates:
        test_start = next_trading_day(d)
        if test_start is None:
            continue

        train_end = d
        train_start = train_end - pd.DateOffset(years=wf_cfg.train_years) + pd.Timedelta(days=1)
        train_start = pd.Timestamp(train_start.date())

        # test end: month-end after test_months
        test_month_end = (pd.Timestamp(test_start) + pd.DateOffset(months=wf_cfg.test_months)).to_period("M").to_timestamp("M")
        test_end = prices.index[prices.index <= test_month_end]
        if len(test_end) == 0:
            continue
        test_end = test_end.max()

        # ensure windows in range
        if train_start < prices.index.min():
            continue
        if test_end > prices.index.max():
            continue

        plan.append(
            {
                "decision_date": d,
                "train_start": pd.Timestamp(train_start),
                "train_end": pd.Timestamp(train_end),
                "test_start": pd.Timestamp(test_start),
                "test_end": pd.Timestamp(test_end),
            }
        )

    if not plan:
        raise ValueError("No valid WF windows could be formed (check price date range).")

    # OOS equity stitching
    oos_curve_parts = []
    selections_rows = []

    # start equity at 1.0
    equity_mult = 1.0

    for step_i, w in enumerate(plan, 1):
        d = w["decision_date"]
        tr_s, tr_e = w["train_start"], w["train_end"]
        te_s, te_e = w["test_start"], w["test_end"]

        # --- TRAIN: evaluate candidates ---
        cand_rows = []
        for overlay in param_sets:
            cfg = deep_merge(base_cfg, overlay)
            seg_eq, met = run_window(prices, cfg, tr_s, tr_e, warmup_days=wf_cfg.warmup_days)
            if not met:
                continue

            row = {
                "decision_date": str(d.date()),
                "train_start": str(tr_s.date()),
                "train_end": str(tr_e.date()),
                "param_id": "",  # filled below
                "cagr": float(met.get("cagr", math.nan)),
                "mdd": float(met.get("mdd", math.nan)),
                "seed_multiple": float(met.get("seed_multiple", math.nan)),
                "max_recovery_days": float(met.get("max_recovery_days", math.nan)),
                "max_recovery_years": float(met.get("max_recovery_years", math.nan)),
                "params_json": json.dumps(overlay, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
            }
            # flatten params for debugging/analysis
            flat = flatten_dict(overlay)
            for k, v in flat.items():
                row[f"params__{k}"] = v

            # deterministic param_id (hash of params_json)
            import hashlib
            row["param_id"] = hashlib.sha1(row["params_json"].encode("utf-8")).hexdigest()[:10]

            cand_rows.append(row)

        ranked = rank_candidates(cand_rows, wf_cfg)
        if not ranked:
            raise RuntimeError(f"No candidates produced metrics for decision_date={d.date()}")

        chosen = ranked[0]
        chosen_overlay = json.loads(chosen["params_json"])

        # store top-k snapshot
        topk = ranked[: max(1, wf_cfg.top_k)]
        topk_path = os.path.join(args.out, f"train_topk_{step_i:03d}_{d.date()}.csv")
        pd.DataFrame(topk).to_csv(topk_path, index=False)

        # --- TEST: run chosen params on OOS month(s) ---
        chosen_cfg = deep_merge(base_cfg, chosen_overlay)
        test_eq, test_met = run_window(prices, chosen_cfg, te_s, te_e, warmup_days=wf_cfg.warmup_days)
        if test_eq.empty or not test_met:
            # still log selection, but skip stitching
            selections_rows.append(
                {
                    "step": step_i,
                    "decision_date": str(d.date()),
                    "train_start": str(tr_s.date()),
                    "train_end": str(tr_e.date()),
                    "test_start": str(te_s.date()),
                    "test_end": str(te_e.date()),
                    "chosen_param_id": chosen["param_id"],
                    "chosen_params_json": chosen["params_json"],
                    "train_cagr": chosen["cagr"],
                    "train_mdd": chosen["mdd"],
                    "train_max_recovery_years": chosen["max_recovery_years"],
                    "test_note": "insufficient test data",
                }
            )
            continue

        # stitch OOS: multiply by current equity_mult
        stitched = test_eq * equity_mult
        equity_mult = float(stitched.iloc[-1])

        oos_curve_parts.append(stitched)

        selections_rows.append(
            {
                "step": step_i,
                "decision_date": str(d.date()),
                "train_start": str(tr_s.date()),
                "train_end": str(tr_e.date()),
                "test_start": str(te_s.date()),
                "test_end": str(te_e.date()),
                "chosen_param_id": chosen["param_id"],
                "chosen_params_json": chosen["params_json"],
                "train_cagr": chosen["cagr"],
                "train_mdd": chosen["mdd"],
                "train_seed_multiple": chosen["seed_multiple"],
                "train_max_recovery_years": chosen["max_recovery_years"],
                "test_seed_multiple": float(test_met.get("seed_multiple", math.nan)),
                "test_cagr": float(test_met.get("cagr", math.nan)),
                "test_mdd": float(test_met.get("mdd", math.nan)),
                "test_max_recovery_years": float(test_met.get("max_recovery_years", math.nan)),
            }
        )

        # persist test curve for this step
        test_curve_path = os.path.join(args.out, f"equity_oos_{step_i:03d}_{te_s.date()}_{te_e.date()}.csv")
        stitched.to_csv(test_curve_path, index=True)

        print(
            f"[WF] {step_i}/{len(plan)} decision={d.date()} "
            f"train={tr_s.date()}..{tr_e.date()} test={te_s.date()}..{te_e.date()} "
            f"pick={chosen['param_id']} train_cagr={chosen['cagr']*100:.2f}% train_mdd={chosen['mdd']*100:.2f}% "
            f"test_cagr={float(test_met.get('cagr',0))*100:.2f}% test_mdd={float(test_met.get('mdd',0))*100:.2f}%"
        )

    # write selections log
    sel_df = pd.DataFrame(selections_rows)
    sel_df.to_csv(os.path.join(args.out, "wf_selections.csv"), index=False)

    # merge OOS curve
    if oos_curve_parts:
        oos = pd.concat(oos_curve_parts).sort_index()
        # drop duplicate dates if windows overlap (keep last)
        oos = oos[~oos.index.duplicated(keep="last")]
        oos.to_csv(os.path.join(args.out, "equity_curve_oos.csv"), index=True)

        oos_met = compute_metrics(rebase(oos))
        rec_days, rec_years = max_recovery_days(rebase(oos))
        oos_met["max_recovery_days"] = float(rec_days)
        oos_met["max_recovery_years"] = float(rec_years)

        with open(os.path.join(args.out, "metrics_oos.json"), "w", encoding="utf-8") as f:
            json.dump(oos_met, f, indent=2, ensure_ascii=False)

    # save config snapshot
    with open(os.path.join(args.out, "wf_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_config": args.config,
                "grid_config": args.grid,
                "wf_config": args.wf,
                "wf": wf_cfg.__dict__,
                "num_candidates": len(param_sets),
                "num_steps": len(selections_rows),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[DONE] out={args.out}")


if __name__ == "__main__":
    main()