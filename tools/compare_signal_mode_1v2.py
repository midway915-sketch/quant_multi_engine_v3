from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.core.meta import run_meta_portfolio


def unwrap_singletons(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: unwrap_singletons(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) == 1:
            return unwrap_singletons(obj[0])
        return [unwrap_singletons(x) for x in obj]
    return obj


def compute_recovery_stats(equity: pd.Series) -> dict[str, float]:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return {
            "recovery_count": 0,
            "avg_recovery_days": float("nan"),
            "median_recovery_days": float("nan"),
            "p90_recovery_days": float("nan"),
            "max_recovery_days": 0,
        }

    running_peak = equity.cummax()
    underwater = equity < running_peak

    durations: list[int] = []
    cur = 0
    for flag in underwater:
        if bool(flag):
            cur += 1
        else:
            if cur > 0:
                durations.append(cur)
                cur = 0

    if not durations:
        return {
            "recovery_count": 0,
            "avg_recovery_days": 0.0,
            "median_recovery_days": 0.0,
            "p90_recovery_days": 0.0,
            "max_recovery_days": 0,
        }

    arr = np.array(durations, dtype=float)
    return {
        "recovery_count": int(len(arr)),
        "avg_recovery_days": float(arr.mean()),
        "median_recovery_days": float(np.median(arr)),
        "p90_recovery_days": float(np.quantile(arr, 0.9)),
        "max_recovery_days": int(arr.max()),
    }


def compute_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return {
            "cagr": float("nan"),
            "mdd": float("nan"),
            "seed_multiple": float("nan"),
            "recovery_count": 0,
            "avg_recovery_days": float("nan"),
            "median_recovery_days": float("nan"),
            "p90_recovery_days": float("nan"),
            "max_recovery_days": 0,
        }

    seed_multiple = float(equity.iloc[-1] / equity.iloc[0])
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    years = (equity.index[-1] - equity.index[0]).days / 365.2425
    cagr = float(seed_multiple ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    return {
        "cagr": cagr,
        "mdd": mdd,
        "seed_multiple": seed_multiple,
        **compute_recovery_stats(equity),
    }


def build_proxy_wide_from_snapshot_root(
    snapshot_root: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for symbol in symbols:
        p = snapshot_root / symbol / "snapshot_closes_wide.csv"
        if not p.exists():
            raise FileNotFoundError(f"snapshot file not found: {p}")

        df = pd.read_csv(p)
        if "date" not in df.columns or "15:50:00" not in df.columns:
            raise ValueError(f"{p} must contain date and 15:50:00")

        df = df[["date", "15:50:00"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"15:50:00": symbol})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="outer")

    assert merged is not None
    merged = merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    merged = merged[
        (merged["date"] >= pd.Timestamp(start_date)) & (merged["date"] <= pd.Timestamp(end_date))
    ].copy()
    return merged


def turnover_cost_frac(w_prev: dict[str, float], w_new: dict[str, float], buy_cost: float, sell_cost: float) -> float:
    keys = set(w_prev.keys()) | set(w_new.keys())
    buy_turn = 0.0
    sell_turn = 0.0

    for k in keys:
        prev = float(w_prev.get(k, 0.0))
        new = float(w_new.get(k, 0.0))
        d = new - prev
        if d > 0:
            buy_turn += d
        elif d < 0:
            sell_turn += (-d)

    return float(buy_turn * buy_cost + sell_turn * sell_cost)


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


def run_branch5a(
    prices: pd.DataFrame,
    lookback: int = 126,
    rebalance: str = "monthly",
    top1_weight: float = 0.70,
    buy_cost: float = 0.0005,
    sell_cost: float = 0.0005,
) -> tuple[pd.Series, pd.DataFrame]:
    prices = prices.copy().sort_index()

    signal_assets = ["QQQ", "SPY", "SOXX"]
    trade_map = {
        "QQQ": "TQQQ_MIX",
        "SPY": "UPRO_MIX",
        "SOXX": "SOXL_MIX",
    }
    defensive = "SGOV_MIX"

    required = signal_assets + list(trade_map.values()) + [defensive]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"missing required columns for branch5a: {missing}")

    returns = prices[list(trade_map.values()) + [defensive]].pct_change().fillna(0.0)

    signal_rets = prices[signal_assets].pct_change()
    roll_mean = signal_rets.rolling(lookback).mean()
    roll_std = signal_rets.rolling(lookback).std().replace(0.0, np.nan)
    score = roll_mean / roll_std

    reb_dates = set(compute_rebalance_dates(prices.index, rebalance))

    h_cur: dict[str, float] = {defensive: 1.0}
    equity = 1.0
    curve: list[float] = []
    logs: list[dict[str, Any]] = []

    top2_weight = 1.0 - float(top1_weight)
    idx = prices.index

    for i, dt in enumerate(idx):
        daily_ret = 0.0
        for t, w in h_cur.items():
            if t in returns.columns:
                daily_ret += float(returns.loc[dt, t]) * float(w)

        equity *= (1.0 + daily_ret)

        selected: list[str] = []
        if dt in reb_dates:
            s = score.loc[dt, signal_assets].dropna().sort_values(ascending=False)
            s = s[s > 0.0]

            if len(s) == 0:
                h_des = {defensive: 1.0}
            elif len(s) == 1:
                selected = [str(s.index[0])]
                h_des = {trade_map[selected[0]]: 1.0}
            else:
                selected = [str(s.index[0]), str(s.index[1])]
                h_des = {
                    trade_map[selected[0]]: float(top1_weight),
                    trade_map[selected[1]]: float(top2_weight),
                }

            if i < len(idx) - 1:
                cost_frac = turnover_cost_frac(h_cur, h_des, buy_cost, sell_cost)
                if cost_frac > 0:
                    equity *= (1.0 - cost_frac)
                h_cur = h_des.copy()

        curve.append(equity)
        logs.append(
            {
                "date": str(dt.date()),
                "rebalance_today": bool(dt in reb_dates),
                "selected_1": selected[0] if len(selected) > 0 else "",
                "selected_2": selected[1] if len(selected) > 1 else "",
                "equity": float(equity),
            }
        )

    return pd.Series(curve, index=prices.index, name="equity"), pd.DataFrame(logs)


def build_hybrid_equity(
    meta_eq: pd.Series,
    branch_eq: pd.Series,
    core_weight: float = 0.70,
    satellite_weight: float = 0.30,
) -> pd.Series:
    idx = meta_eq.index.intersection(branch_eq.index)
    meta_eq = meta_eq.loc[idx].astype(float)
    branch_eq = branch_eq.loc[idx].astype(float)

    meta_ret = meta_eq.pct_change().fillna(0.0)
    branch_ret = branch_eq.pct_change().fillna(0.0)

    hybrid_ret = core_weight * meta_ret + satellite_weight * branch_ret
    hybrid_eq = (1.0 + hybrid_ret).cumprod()
    hybrid_eq.name = "equity"
    return hybrid_eq


def prepare_mode_price_frames(
    prices_csv: Path,
    proxy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    signal_symbols = ["QQQ", "SPY", "SOXX"]
    trade_cols = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]
    missing = [c for c in signal_symbols + trade_cols if c not in prices.columns]
    if missing:
        raise ValueError(f"prices csv missing columns: {missing}")

    proxy = proxy_df.copy()
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy = proxy.set_index("date").sort_index()

    common_idx = prices.index.intersection(proxy.index)
    common_idx = common_idx[
        (common_idx >= pd.Timestamp(start_date)) & (common_idx <= pd.Timestamp(end_date))
    ]

    if len(common_idx) < 200:
        raise ValueError(f"too few common dates: {len(common_idx)}")

    actual_prices = prices.loc[common_idx].copy()

    # mode1: D-1 official close로 신호 계산 -> D일 종가 체결
    # 현재 엔진은 "그 날짜에 계산해서 그 날짜 종가에 리밸런싱" 구조라
    # 신호용 종가를 1일 shift 해서 넣으면 실행시점이 D+1 의미가 된다.
    mode1_prices = actual_prices.copy()
    for s in signal_symbols:
        mode1_prices[s] = actual_prices[s].shift(1)

    # mode2: D일 15:50 proxy close로 신호 계산 -> D일 종가 체결
    mode2_prices = actual_prices.copy()
    for s in signal_symbols:
        mode2_prices[s] = proxy.loc[common_idx, s].astype(float).values

    # 첫날은 mode1 shift 때문에 NaN이 생기므로 제거
    valid_idx = mode1_prices.dropna(subset=signal_symbols).index
    valid_idx = valid_idx.intersection(mode2_prices.dropna(subset=signal_symbols).index)

    actual_prices = actual_prices.loc[valid_idx].copy()
    mode1_prices = mode1_prices.loc[valid_idx].copy()
    mode2_prices = mode2_prices.loc[valid_idx].copy()

    return actual_prices, mode1_prices, mode2_prices


def save_series(series: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-csv", required=True)
    parser.add_argument("--meta-config-yaml", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--out-dir", default="out/compare_signal_mode_1v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["QQQ", "SPY", "SOXX"]
    proxy_df = build_proxy_wide_from_snapshot_root(
        snapshot_root=Path(args.snapshot_root),
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    with open(args.meta_config_yaml, "r", encoding="utf-8") as f:
        meta_cfg = unwrap_singletons(yaml.safe_load(f))

    actual_prices, mode1_signal_prices, mode2_signal_prices = prepare_mode_price_frames(
        prices_csv=Path(args.prices_csv),
        proxy_df=proxy_df,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # 메타/브랜치 모두 trade return은 actual_prices의 trade columns를 사용하고
    # signal columns(QQQ/SPY/SOXX)만 mode1/mode2에 맞게 다르게 준다.
    meta_prices_1 = actual_prices.copy()
    meta_prices_2 = actual_prices.copy()
    for s in symbols:
        meta_prices_1[s] = mode1_signal_prices[s]
        meta_prices_2[s] = mode2_signal_prices[s]

    branch_prices_1 = actual_prices[["QQQ", "SPY", "SOXX", "TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]].copy()
    branch_prices_2 = branch_prices_1.copy()
    for s in symbols:
        branch_prices_1[s] = mode1_signal_prices[s]
        branch_prices_2[s] = mode2_signal_prices[s]

    # Meta
    meta_eq_1, meta_log_1, _, _, _ = run_meta_portfolio(meta_prices_1, meta_cfg)
    meta_eq_2, meta_log_2, _, _, _ = run_meta_portfolio(meta_prices_2, meta_cfg)

    # Branch5A
    branch_eq_1, branch_log_1 = run_branch5a(
        prices=branch_prices_1,
        lookback=126,
        rebalance="monthly",
        top1_weight=0.70,
        buy_cost=0.0005,
        sell_cost=0.0005,
    )
    branch_eq_2, branch_log_2 = run_branch5a(
        prices=branch_prices_2,
        lookback=126,
        rebalance="monthly",
        top1_weight=0.70,
        buy_cost=0.0005,
        sell_cost=0.0005,
    )

    # Hybrid 70/30
    hybrid_eq_1 = build_hybrid_equity(meta_eq_1, branch_eq_1, core_weight=0.70, satellite_weight=0.30)
    hybrid_eq_2 = build_hybrid_equity(meta_eq_2, branch_eq_2, core_weight=0.70, satellite_weight=0.30)

    summary_rows: list[dict[str, Any]] = []
    for strategy_name, eq1, eq2 in [
        ("meta_mode1_nextday_vs_mode2_sameday", meta_eq_1, meta_eq_2),
        ("branch5a_mode1_nextday_vs_mode2_sameday", branch_eq_1, branch_eq_2),
        ("hybrid_70_30_mode1_nextday_vs_mode2_sameday", hybrid_eq_1, hybrid_eq_2),
    ]:
        m1 = compute_metrics(eq1)
        m2 = compute_metrics(eq2)
        summary_rows.append(
            {
                "strategy": strategy_name,
                "mode1_execution": "official_close_signal_nextday_trade",
                "mode2_execution": "proxy_1550_signal_sameday_trade",
                "mode1_cagr": m1["cagr"],
                "mode1_mdd": m1["mdd"],
                "mode1_seed_multiple": m1["seed_multiple"],
                "mode1_recovery_count": m1["recovery_count"],
                "mode1_avg_recovery_days": m1["avg_recovery_days"],
                "mode1_median_recovery_days": m1["median_recovery_days"],
                "mode1_p90_recovery_days": m1["p90_recovery_days"],
                "mode1_max_recovery_days": m1["max_recovery_days"],
                "mode2_cagr": m2["cagr"],
                "mode2_mdd": m2["mdd"],
                "mode2_seed_multiple": m2["seed_multiple"],
                "mode2_recovery_count": m2["recovery_count"],
                "mode2_avg_recovery_days": m2["avg_recovery_days"],
                "mode2_median_recovery_days": m2["median_recovery_days"],
                "mode2_p90_recovery_days": m2["p90_recovery_days"],
                "mode2_max_recovery_days": m2["max_recovery_days"],
                "delta_cagr": m2["cagr"] - m1["cagr"],
                "delta_mdd": m2["mdd"] - m1["mdd"],
                "delta_avg_recovery_days": m2["avg_recovery_days"] - m1["avg_recovery_days"],
                "delta_max_recovery_days": m2["max_recovery_days"] - m1["max_recovery_days"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_compare_1v2.csv", index=False)

    save_series(meta_eq_1, out_dir / "meta_mode1_equity.csv")
    save_series(meta_eq_2, out_dir / "meta_mode2_equity.csv")
    save_series(branch_eq_1, out_dir / "branch5a_mode1_equity.csv")
    save_series(branch_eq_2, out_dir / "branch5a_mode2_equity.csv")
    save_series(hybrid_eq_1, out_dir / "hybrid_mode1_equity.csv")
    save_series(hybrid_eq_2, out_dir / "hybrid_mode2_equity.csv")

    meta_log_1.to_csv(out_dir / "meta_mode1_log.csv", index=False)
    meta_log_2.to_csv(out_dir / "meta_mode2_log.csv", index=False)
    branch_log_1.to_csv(out_dir / "branch5a_mode1_log.csv", index=False)
    branch_log_2.to_csv(out_dir / "branch5a_mode2_log.csv", index=False)

    signal_debug = pd.DataFrame(
        {
            "date": actual_prices.index,
            "QQQ_mode1_signal": mode1_signal_prices["QQQ"].values,
            "QQQ_mode2_signal": mode2_signal_prices["QQQ"].values,
            "SPY_mode1_signal": mode1_signal_prices["SPY"].values,
            "SPY_mode2_signal": mode2_signal_prices["SPY"].values,
            "SOXX_mode1_signal": mode1_signal_prices["SOXX"].values,
            "SOXX_mode2_signal": mode2_signal_prices["SOXX"].values,
        }
    )
    signal_debug.to_csv(out_dir / "signal_price_debug.csv", index=False)

    info = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows_proxy": int(len(proxy_df)),
        "rows_actual_prices": int(len(actual_prices)),
        "rows_mode1": int(len(mode1_signal_prices)),
        "rows_mode2": int(len(mode2_signal_prices)),
        "mode1_definition": "official close signal shifted by 1 trading day, executed at current day close",
        "mode2_definition": "15:50 ET proxy signal same day, executed at current day close",
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print(f"saved: {out_dir / 'summary_compare_1v2.csv'}")


if __name__ == "__main__":
    main()