#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import yaml

from src.core.data import download_prices_and_build_proxies
from src.core.meta import run_meta_portfolio
from src.core.metrics import compute_metrics


# -------------------------
# helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def rebase(eq: pd.Series) -> pd.Series:
    if eq is None or eq.empty:
        return eq
    base = float(eq.iloc[0])
    if base <= 0 or pd.isna(base):
        return eq
    return eq / base


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


def slice_with_warmup(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, warmup_days: int) -> pd.DataFrame:
    """
    Include a warmup window before start for indicators, but keep data up to end.
    Warmup is in calendar days (not trading days) to avoid missing holidays.
    """
    warm_start = start - pd.Timedelta(days=int(warmup_days))
    px = prices.loc[(prices.index >= warm_start) & (prices.index <= end)].copy()
    return px


def month_end_trading_days(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Monthly rebalance dates: last trading day of each month (based on available price index).
    """
    me = prices.resample("ME").last().index
    return prices.index[prices.index.isin(me)]


@dataclass
class WFConfig:
    train_years: int
    test_months: int
    warmup_days: int

    filter_mdd_min: float
    filter_max_recovery_years_max: float

    score_mode: str  # "cagr_only" | "cagr_minus_dd" | "cagr_then_dd_then_rec"
    score_dd_weight: float
    top_k: int

    progress_every_steps: int
    progress_every_candidates: int


def load_wf_cfg(path: str) -> WFConfig:
    obj = load_yaml(path)
    w = obj.get("wf", {}) or {}
    sel = obj.get("selection", {}) or {}
    filt = obj.get("filters", {}) or {}
    prog = obj.get("progress", {}) or {}

    return WFConfig(
        train_years=int(w.get("train_years", 10)),
        test_months=int(w.get("test_months", 1)),
        warmup_days=int(w.get("warmup_days", 900)),
        filter_mdd_min=float(filt.get("mdd_min", -0.65)),
        filter_max_recovery_years_max=float(filt.get("max_recovery_years_max", 3.0)),
        score_mode=str(sel.get("score_mode", "cagr_then_dd_then_rec")).lower().strip(),
        score_dd_weight=float(sel.get("dd_weight", 0.5)),
        top_k=int(sel.get("top_k", 5)),
        progress_every_steps=int(prog.get("every_steps", 1)),
        progress_every_candidates=int(prog.get("every_candidates", 25)),
    )


def rank_candidates(rows: List[dict], wf_cfg: WFConfig) -> List[dict]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []

    for c in ["cagr", "mdd", "max_recovery_years"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df_f = df.copy()
    df_f = df_f[df_f["mdd"].notna() & df_f["cagr"].notna()]
    df_f = df_f[df_f["mdd"] >= float(wf_cfg.filter_mdd_min)]
    df_f = df_f[df_f["max_recovery_years"].notna() & (df_f["max_recovery_years"] <= float(wf_cfg.filter_max_recovery_years_max))]

    use = df_f if not df_f.empty else df

    if wf_cfg.score_mode == "cagr_only":
        use = use.sort_values(["cagr"], ascending=[False], na_position="last")
    elif wf_cfg.score_mode == "cagr_minus_dd":
        w = float(wf_cfg.score_dd_weight)
        use["score"] = use["cagr"] - w * use["mdd"].abs()
        use = use.sort_values(["score", "cagr"], ascending=[False, False], na_position="last")
    else:
        use = use.sort_values(["cagr", "mdd", "max_recovery_years"], ascending=[False, False, True], na_position="last")

    return use.to_dict(orient="records")


# -------------------------
# parallel worker globals (fork-friendly)
# -------------------------
_G_BASE_CFG: Optional[dict] = None
_G_PX_TRAIN: Optional[pd.DataFrame] = None
_G_TR_START: Optional[pd.Timestamp] = None
_G_TR_END: Optional[pd.Timestamp] = None


def _init_worker(base_cfg: dict, px_train: pd.DataFrame, tr_s: pd.Timestamp, tr_e: pd.Timestamp) -> None:
    global _G_BASE_CFG, _G_PX_TRAIN, _G_TR_START, _G_TR_END
    _G_BASE_CFG = base_cfg
    _G_PX_TRAIN = px_train
    _G_TR_START = tr_s
    _G_TR_END = tr_e


def _eval_candidate(overlay: dict) -> Optional[dict]:
    """
    Evaluate one candidate on the TRAIN window, using global px_train (warmup included).
    Returns row dict or None.
    """
    global _G_BASE_CFG, _G_PX_TRAIN, _G_TR_START, _G_TR_END
    if _G_BASE_CFG is None or _G_PX_TRAIN is None or _G_TR_START is None or _G_TR_END is None:
        raise RuntimeError("Worker globals not initialized")

    cfg = deep_merge(_G_BASE_CFG, overlay)

    eq, *_ = run_meta_portfolio(_G_PX_TRAIN, cfg)
    seg = eq.loc[(eq.index >= _G_TR_START) & (eq.index <= _G_TR_END)]
    if seg.empty:
        return None
    seg = rebase(seg)

    met = compute_metrics(seg)
    rec_days, rec_years = max_recovery_days(seg)

    import hashlib
    params_json = json.dumps(overlay, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    param_id = hashlib.sha1(params_json.encode("utf-8")).hexdigest()[:10]

    row = {
        "param_id": param_id,
        "cagr": float(met.get("cagr", math.nan)),
        "mdd": float(met.get("mdd", math.nan)),
        "seed_multiple": float(met.get("seed_multiple", math.nan)),
        "max_recovery_days": float(rec_days),
        "max_recovery_years": float(rec_years),
        "params_json": params_json,
    }
    flat = flatten_dict(overlay)
    for k, v in flat.items():
        row[f"params__{k}"] = v
    return row


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config (fixed params)")
    ap.add_argument("--grid", required=True, help="Grid yml (scan params)")
    ap.add_argument("--wf", required=True, help="WF config yml")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--prices-csv", default="", help="Optional prebuilt prices.csv (skip download)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for grid eval per step (>=1)")
    return ap.parse_args()


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

    rebal_dates = month_end_trading_days(prices)
    if len(rebal_dates) < 24:
        raise ValueError(f"Not enough month-end trading days found: {len(rebal_dates)}")

    idx = prices.index

    def next_trading_day(d: pd.Timestamp) -> Optional[pd.Timestamp]:
        loc = idx.get_indexer([d], method="pad")
        if loc[0] < 0:
            return None
        i = int(loc[0])
        if i >= len(idx) - 1:
            return None
        return idx[i + 1]

    # build decision plan
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
        cand_end = prices.index[prices.index <= test_month_end]
        if len(cand_end) == 0:
            continue
        test_end = cand_end.max()

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

    # OOS stitching
    oos_curve_parts = []
    selections_rows = []
    equity_mult = 1.0

    # WF progress timer
    wf_t0 = time.time()
    total_steps = len(plan)

    for step_i, w in enumerate(plan, 1):
        step_t0 = time.time()

        d = w["decision_date"]
        tr_s, tr_e = w["train_start"], w["train_end"]
        te_s, te_e = w["test_start"], w["test_end"]

        # TRAIN warm slice ONCE per step
        px_train = slice_with_warmup(prices, tr_s, tr_e, warmup_days=wf_cfg.warmup_days)
        if px_train.empty:
            continue

        # --- TRAIN: eval all candidates (parallel inside a step) ---
        cand_rows: List[dict] = []

        n_cands = len(param_sets)
        cand_t0 = time.time()

        if args.workers <= 1:
            for j, overlay in enumerate(param_sets, 1):
                _init_worker(base_cfg, px_train, tr_s, tr_e)
                row = _eval_candidate(overlay)
                if row:
                    cand_rows.append(row)

                if j == 1 or j % wf_cfg.progress_every_candidates == 0 or j == n_cands:
                    elapsed = time.time() - cand_t0
                    per = elapsed / j
                    eta = per * (n_cands - j)
                    print(
                        f"[WF-TRAIN] step={step_i}/{total_steps} cand={j}/{n_cands} "
                        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m per={per:.2f}s"
                    )
        else:
            # process pool (fork on linux is fast; on windows it will still work but may be slower)
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            ctx = mp.get_context("fork") if hasattr(mp, "get_context") else None
            executor_args = {}
            if ctx is not None:
                executor_args["mp_context"] = ctx

            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=_init_worker,
                initargs=(base_cfg, px_train, tr_s, tr_e),
                **executor_args,
            ) as ex:
                futures = [ex.submit(_eval_candidate, overlay) for overlay in param_sets]
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    row = fut.result()
                    if row:
                        cand_rows.append(row)

                    if done == 1 or done % wf_cfg.progress_every_candidates == 0 or done == n_cands:
                        elapsed = time.time() - cand_t0
                        per = elapsed / done
                        eta = per * (n_cands - done)
                        print(
                            f"[WF-TRAIN] step={step_i}/{total_steps} cand={done}/{n_cands} "
                            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m per={per:.2f}s workers={args.workers}"
                        )

        ranked = rank_candidates(cand_rows, wf_cfg)
        if not ranked:
            raise RuntimeError(f"No candidates produced metrics for decision_date={d.date()}")

        chosen = ranked[0]
        chosen_overlay = json.loads(chosen["params_json"])

        # save top-k snapshot
        topk = ranked[: max(1, wf_cfg.top_k)]
        topk_path = os.path.join(args.out, f"train_topk_{step_i:03d}_{d.date()}.csv")
        pd.DataFrame(topk).to_csv(topk_path, index=False)

        # --- TEST: run chosen params on OOS month(s) ---
        chosen_cfg = deep_merge(base_cfg, chosen_overlay)
        px_test = slice_with_warmup(prices, te_s, te_e, warmup_days=wf_cfg.warmup_days)
        if px_test.empty:
            continue

        eq_test, *_ = run_meta_portfolio(px_test, chosen_cfg)
        seg = eq_test.loc[(eq_test.index >= te_s) & (eq_test.index <= te_e)]
        if seg.empty:
            continue
        seg = rebase(seg)

        test_met = compute_metrics(seg)
        rec_days, rec_years = max_recovery_days(seg)
        test_met["max_recovery_days"] = float(rec_days)
        test_met["max_recovery_years"] = float(rec_years)

        stitched = seg * equity_mult
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
                "train_seed_multiple": chosen.get("seed_multiple", math.nan),
                "train_max_recovery_years": chosen.get("max_recovery_years", math.nan),
                "test_seed_multiple": float(test_met.get("seed_multiple", math.nan)),
                "test_cagr": float(test_met.get("cagr", math.nan)),
                "test_mdd": float(test_met.get("mdd", math.nan)),
                "test_max_recovery_years": float(test_met.get("max_recovery_years", math.nan)),
            }
        )

        test_curve_path = os.path.join(args.out, f"equity_oos_{step_i:03d}_{te_s.date()}_{te_e.date()}.csv")
        stitched.to_csv(test_curve_path, index=True)

        # step progress + ETA
        step_elapsed = time.time() - step_t0
        wf_elapsed = time.time() - wf_t0
        per_step = wf_elapsed / step_i
        wf_eta = per_step * (total_steps - step_i)

        if step_i == 1 or step_i % max(1, wf_cfg.progress_every_steps) == 0 or step_i == total_steps:
            print(
                f"[WF-STEP] {step_i}/{total_steps} decision={d.date()} "
                f"train={tr_s.date()}..{tr_e.date()} test={te_s.date()}..{te_e.date()} "
                f"pick={chosen['param_id']} "
                f"train_cagr={chosen['cagr']*100:.2f}% train_mdd={chosen['mdd']*100:.2f}% "
                f"test_cagr={float(test_met.get('cagr',0))*100:.2f}% test_mdd={float(test_met.get('mdd',0))*100:.2f}% "
                f"step_elapsed={step_elapsed/60:.1f}m wf_elapsed={wf_elapsed/60:.1f}m wf_eta={wf_eta/60:.1f}m"
            )

    # write selections log
    sel_df = pd.DataFrame(selections_rows)
    sel_df.to_csv(os.path.join(args.out, "wf_selections.csv"), index=False)

    # merge OOS curve
    if oos_curve_parts:
        oos = pd.concat(oos_curve_parts).sort_index()
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
                "workers": int(args.workers),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[DONE] out={args.out}")


if __name__ == "__main__":
    main()