from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

SIGNAL_COLS = ["QQQ", "SPY", "SOXX"]
TRADE_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]
SIGNAL_TO_TRADE = {
    "QQQ": "TQQQ_MIX",
    "SPY": "UPRO_MIX",
    "SOXX": "SOXL_MIX",
    "SGOV": "SGOV_MIX",
    "TQQQ": "TQQQ_MIX",
    "UPRO": "UPRO_MIX",
    "SOXL": "SOXL_MIX",
    "SGOV_MIX": "SGOV_MIX",
    "TQQQ_MIX": "TQQQ_MIX",
    "UPRO_MIX": "UPRO_MIX",
    "SOXL_MIX": "SOXL_MIX",
}
DOWNLOAD_TICKERS = ["QQQ", "SPY", "SOXX", "TQQQ", "UPRO", "SOXL", "SGOV"]


def unwrap_singletons(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: unwrap_singletons(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) == 1:
            return unwrap_singletons(obj[0])
        return [unwrap_singletons(x) for x in obj]
    return obj


def load_run_meta_portfolio():
    candidates = [
        Path("src/engine/meta.py"),
        Path("src/engine/meta_portfolio.py"),
        Path("src/meta.py"),
        Path("src/strategy/meta.py"),
        Path("src/strategies/meta.py"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"_meta_mod_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "run_meta_portfolio", None)
        if fn is not None:
            return fn, str(path)
    raise ModuleNotFoundError("run_meta_portfolio not found in known paths")


RUN_META_PORTFOLIO, RUN_META_PATH = load_run_meta_portfolio()


def ensure_dataframe(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, list):
        return pd.DataFrame(x)
    if isinstance(x, dict):
        return pd.DataFrame([x])
    if x is None:
        return pd.DataFrame()
    return pd.DataFrame(x)


def download_one(symbol: str, start_date: str, end_date: str) -> pd.Series:
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_plus,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"empty price data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise RuntimeError(f"missing Close for {symbol}")
        s = df["Close"].iloc[:, 0].copy()
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"missing Close for {symbol}")
        s = df["Close"].copy()

    s.index = pd.to_datetime(s.index)
    s = s.sort_index().astype(float)
    if symbol in ["TQQQ", "UPRO", "SOXL"]:
        s.name = f"{symbol}_MIX"
    elif symbol == "SGOV":
        s.name = "SGOV_MIX"
    else:
        s.name = symbol
    return s


def download_prices(start_date: str, end_date: str) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for symbol in DOWNLOAD_TICKERS:
        print(f"downloading {symbol} ...", flush=True)
        s = download_one(symbol, start_date, end_date)
        parts.append(s)
    prices = pd.concat(parts, axis=1).sort_index()
    prices.index.name = "date"
    return prices


def compute_rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    rebalance = str(rebalance).lower().strip()
    if rebalance == "monthly":
        vals = pd.Series(index=index, data=index).groupby(index.to_period("M")).max().values
        return pd.DatetimeIndex(vals)
    if rebalance == "weekly":
        vals = pd.Series(index=index, data=index).groupby(index.to_period("W-FRI")).max().values
        return pd.DatetimeIndex(vals)
    if rebalance == "biweekly":
        vals = pd.Series(index=index, data=index).groupby(index.to_period("W-FRI")).max().values
        return pd.DatetimeIndex(vals[::2])
    raise ValueError(f"unsupported rebalance: {rebalance}")


def run_branch5a_targets(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy().sort_index()
    signal_prices = prices[SIGNAL_COLS].copy()

    signal_rets = signal_prices.pct_change()
    score = signal_rets.rolling(126).mean() / signal_rets.rolling(126).std().replace(0.0, np.nan)
    reb_dates = set(compute_rebalance_dates(prices.index, "monthly"))

    cur_target = {"TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 1.0}
    rows: list[dict[str, Any]] = []

    for dt in prices.index:
        if dt in reb_dates:
            s = score.loc[dt, SIGNAL_COLS].dropna().sort_values(ascending=False)
            s = s[s > 0.0]

            if len(s) == 0:
                cur_target = {"TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 1.0}
            elif len(s) == 1:
                winner = SIGNAL_TO_TRADE[str(s.index[0])]
                cur_target = {"TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 0.0}
                cur_target[winner] = 1.0
            else:
                w1 = SIGNAL_TO_TRADE[str(s.index[0])]
                w2 = SIGNAL_TO_TRADE[str(s.index[1])]
                cur_target = {"TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 0.0}
                cur_target[w1] = 0.70
                cur_target[w2] = 0.30

        row = {"date": dt}
        row.update(cur_target)
        rows.append(row)

    return pd.DataFrame(rows)


def align_log_dates(log_df: pd.DataFrame, prices_index: pd.DatetimeIndex) -> pd.DataFrame:
    df = log_df.copy()

    date_candidates = ["date", "dt", "trading_date", "signal_date"]
    for c in date_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.rename(columns={c: "date"})
            return df

    if len(df) == len(prices_index):
        df.insert(0, "date", prices_index)
        return df

    raise RuntimeError(
        f"Could not align meta log with dates. meta_log_rows={len(df)} price_rows={len(prices_index)} columns={list(df.columns)}"
    )


def maybe_parse_dict(x: Any) -> dict[str, Any] | None:
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    txt = x.strip()
    if not txt:
        return None

    # try json
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # try python literal
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def normalize_asset_name(name: str) -> str | None:
    if not name:
        return None
    key = str(name).strip().upper()
    return SIGNAL_TO_TRADE.get(key)


def extract_direct_weight_cols(meta_log_df: pd.DataFrame) -> pd.DataFrame | None:
    colmap = {}
    for asset in TRADE_COLS:
        candidates = [
            asset,
            asset.lower(),
            f"w_{asset}",
            f"weight_{asset}",
            f"target_{asset}",
            f"alloc_{asset}",
            f"w_{asset.lower()}",
            f"weight_{asset.lower()}",
            f"target_{asset.lower()}",
            f"alloc_{asset.lower()}",
        ]
        found = None
        for c in candidates:
            if c in meta_log_df.columns:
                found = c
                break
        if found is None:
            return None
        colmap[asset] = found

    out = meta_log_df[["date"]].copy()
    for asset, src in colmap.items():
        out[asset] = pd.to_numeric(meta_log_df[src], errors="coerce").fillna(0.0)
    return out


def extract_from_dict_cols(meta_log_df: pd.DataFrame) -> pd.DataFrame | None:
    dict_cols = ["weights", "target", "target_weights", "holdings", "allocation", "alloc", "portfolio", "position_weights"]
    for c in dict_cols:
        if c not in meta_log_df.columns:
            continue

        rows: list[dict[str, Any]] = []
        any_parsed = False
        for _, r in meta_log_df.iterrows():
            d = maybe_parse_dict(r[c])
            row = {"date": r["date"], "TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 0.0}
            if d:
                any_parsed = True
                for k, v in d.items():
                    mapped = normalize_asset_name(str(k))
                    if mapped is not None:
                        try:
                            row[mapped] = float(v)
                        except Exception:
                            row[mapped] = 0.0
            rows.append(row)

        if any_parsed:
            return pd.DataFrame(rows)

    return None


def extract_by_inference(meta_log_df: pd.DataFrame) -> pd.DataFrame | None:
    state_cols = [c for c in meta_log_df.columns if c.lower() in ["state", "bucket", "regime"]]
    winner_cols = [c for c in meta_log_df.columns if c.lower() in ["selected", "winner", "trend_asset", "selected_1", "signal_asset", "picked", "selected_asset"]]

    if not state_cols and not winner_cols:
        return None

    state_col = state_cols[0] if state_cols else None
    winner_col = winner_cols[0] if winner_cols else None

    rows: list[dict[str, Any]] = []
    for _, r in meta_log_df.iterrows():
        row = {"date": r["date"], "TQQQ_MIX": 0.0, "UPRO_MIX": 0.0, "SOXL_MIX": 0.0, "SGOV_MIX": 0.0}

        state = str(r[state_col]).strip().lower() if state_col else ""
        winner = normalize_asset_name(str(r[winner_col]).strip()) if winner_col else None

        if state in ["bear", "crash", "risk_off", "defensive"]:
            row["SGOV_MIX"] = 1.0
        elif winner is not None:
            row[winner] = 1.0
        else:
            # fallback: if no clear winner/state, assume defensive
            row["SGOV_MIX"] = 1.0

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in TRADE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    s = out[TRADE_COLS].sum(axis=1)
    s = s.replace(0.0, np.nan)
    out[TRADE_COLS] = out[TRADE_COLS].div(s, axis=0).fillna(0.0)
    return out


def extract_meta_targets(prices: pd.DataFrame, meta_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = RUN_META_PORTFOLIO(prices.copy(), meta_cfg)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("run_meta_portfolio returned unexpected structure")

    meta_log_df = ensure_dataframe(result[1])
    meta_log_df = align_log_dates(meta_log_df, prices.index)
    meta_log_df["date"] = pd.to_datetime(meta_log_df["date"])

    direct = extract_direct_weight_cols(meta_log_df)
    if direct is not None:
        return normalize_weights(direct), meta_log_df

    dict_based = extract_from_dict_cols(meta_log_df)
    if dict_based is not None:
        return normalize_weights(dict_based), meta_log_df

    inferred = extract_by_inference(meta_log_df)
    if inferred is not None:
        return normalize_weights(inferred), meta_log_df

    meta_log_df.to_csv("out/meta_log_debug_unparsed.csv", index=False)
    raise RuntimeError(
        "Could not extract meta target weights automatically. "
        "See out/meta_log_debug_unparsed.csv and inspect columns: "
        f"{list(meta_log_df.columns)}"
    )


def combine_final(meta_df: pd.DataFrame, branch_df: pd.DataFrame) -> pd.DataFrame:
    df = meta_df.merge(branch_df, on="date", suffixes=("_meta", "_branch"), how="inner")
    out = pd.DataFrame({"date": df["date"]})
    for c in TRADE_COLS:
        out[c] = 0.70 * df[f"{c}_meta"] + 0.30 * df[f"{c}_branch"]

    s = out[TRADE_COLS].sum(axis=1)
    s = s.replace(0.0, np.nan)
    out[TRADE_COLS] = out[TRADE_COLS].div(s, axis=0).fillna(0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--meta-config-yaml", default="config/final_meta_fixed.yml")
    parser.add_argument("--prices-out-csv", default="data/prices.csv")
    parser.add_argument("--meta-out-csv", default="out/meta_target_weights.csv")
    parser.add_argument("--branch-out-csv", default="out/branch5a_target_weights.csv")
    parser.add_argument("--final-out-csv", default="out/final_target_weights.csv")
    args = parser.parse_args()

    with open(args.meta_config_yaml, "r", encoding="utf-8") as f:
        meta_cfg = unwrap_singletons(yaml.safe_load(f))

    prices = download_prices(args.start_date, args.end_date)

    Path(args.prices_out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.branch_out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.final_out_csv).parent.mkdir(parents=True, exist_ok=True)

    prices.to_csv(args.prices_out_csv)

    meta_targets, meta_log_df = extract_meta_targets(prices, meta_cfg)
    branch_targets = run_branch5a_targets(prices)
    final_targets = combine_final(meta_targets, branch_targets)

    meta_targets.to_csv(args.meta_out_csv, index=False)
    branch_targets.to_csv(args.branch_out_csv, index=False)
    final_targets.to_csv(args.final_out_csv, index=False)

    # debug outputs
    meta_log_df.to_csv("out/meta_log_debug.csv", index=False)

    run_info = {
        "run_meta_portfolio_loaded_from": RUN_META_PATH,
        "prices_rows": int(len(prices)),
        "meta_rows": int(len(meta_targets)),
        "branch_rows": int(len(branch_targets)),
        "final_rows": int(len(final_targets)),
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
    }
    Path("out").mkdir(parents=True, exist_ok=True)
    with open("out/build_final_target_weights_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_info, ensure_ascii=False))
    print(f"saved: {args.prices_out_csv}")
    print(f"saved: {args.meta_out_csv}")
    print(f"saved: {args.branch_out_csv}")
    print(f"saved: {args.final_out_csv}")


if __name__ == "__main__":
    main()