from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRADE_COLS = ["TQQQ_MIX", "UPRO_MIX", "SOXL_MIX", "SGOV_MIX"]
CASH = "CASH"


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

    peak = equity.cummax()
    underwater = equity < peak

    durations: list[int] = []
    cur = 0
    for flag in underwater:
        if bool(flag):
            cur += 1
        else:
            if cur > 0:
                durations.append(cur)
                cur = 0

    if cur > 0:
        durations.append(cur)

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


def normalize_target(row: pd.Series) -> dict[str, float]:
    tgt = {}
    for c in TRADE_COLS:
        v = float(row.get(c, 0.0) or 0.0)
        if v < 0:
            v = 0.0
        tgt[c] = v

    s = sum(tgt.values())
    if s > 1.000001:
        for k in tgt:
            tgt[k] /= s
        s = 1.0

    tgt[CASH] = max(0.0, 1.0 - s)
    return tgt


def apply_returns(weights: dict[str, float], returns_row: pd.Series) -> float:
    port_ret = 0.0
    for k, w in weights.items():
        if k == CASH:
            continue
        port_ret += float(w) * float(returns_row.get(k, 0.0))
    return port_ret


def rebalance_full(
    weights: dict[str, float],
    target: dict[str, float],
    buy_cost: float,
    sell_cost: float,
) -> tuple[dict[str, float], float]:
    keys = set(weights.keys()) | set(target.keys())
    buy_turn = 0.0
    sell_turn = 0.0

    for k in keys:
        cur = float(weights.get(k, 0.0))
        tgt = float(target.get(k, 0.0))
        d = tgt - cur
        if d > 0:
            buy_turn += d
        elif d < 0:
            sell_turn += -d

    cost = buy_turn * buy_cost + sell_turn * sell_cost
    return dict(target), float(cost)


def rebalance_sell_only(
    weights: dict[str, float],
    target: dict[str, float],
    sell_cost: float,
) -> tuple[dict[str, float], float]:
    out = dict(weights)
    out.setdefault(CASH, 0.0)
    sell_turn = 0.0

    for k in TRADE_COLS:
        cur = float(out.get(k, 0.0))
        tgt = float(target.get(k, 0.0))
        if cur > tgt:
            delta = cur - tgt
            out[k] = tgt
            out[CASH] += delta
            sell_turn += delta

    cost = sell_turn * sell_cost
    out[CASH] = max(0.0, out.get(CASH, 0.0) - cost)
    total = sum(out.values())
    if total > 0:
        for k in out:
            out[k] /= total
    return out, float(cost)


def rebalance_buy_only(
    weights: dict[str, float],
    target: dict[str, float],
    buy_cost: float,
) -> tuple[dict[str, float], float]:
    out = dict(weights)
    out.setdefault(CASH, 0.0)

    need_total = 0.0
    needs: dict[str, float] = {}
    for k in TRADE_COLS:
        cur = float(out.get(k, 0.0))
        tgt = float(target.get(k, 0.0))
        if tgt > cur:
            need = tgt - cur
            needs[k] = need
            need_total += need

    cash_avail = float(out.get(CASH, 0.0))
    if cash_avail <= 0 or need_total <= 0:
        return out, 0.0

    gross_buy_cap = cash_avail / (1.0 + buy_cost)
    buy_turn = min(need_total, gross_buy_cap)

    if buy_turn <= 0:
        return out, 0.0

    scale = buy_turn / need_total
    for k, need in needs.items():
        alloc = need * scale
        out[k] = float(out.get(k, 0.0)) + alloc

    cost = buy_turn * buy_cost
    out[CASH] = cash_avail - buy_turn - cost
    out[CASH] = max(0.0, out[CASH])

    total = sum(out.values())
    if total > 0:
        for k in out:
            out[k] /= total

    return out, float(cost)


def prepare_inputs(prices_csv: str, targets_csv: str, start_date: str | None, end_date: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True).sort_index()
    prices.index = pd.to_datetime(prices.index)

    targets = pd.read_csv(targets_csv, parse_dates=["date"]).sort_values("date")
    targets["date"] = pd.to_datetime(targets["date"])
    targets = targets.set_index("date")

    missing_prices = [c for c in TRADE_COLS if c not in prices.columns]
    if missing_prices:
        raise ValueError(f"prices missing columns: {missing_prices}")

    missing_targets = [c for c in TRADE_COLS if c not in targets.columns]
    if missing_targets:
        raise ValueError(f"targets missing columns: {missing_targets}")

    idx = prices.index.intersection(targets.index)
    if start_date:
        idx = idx[idx >= pd.Timestamp(start_date)]
    if end_date:
        idx = idx[idx <= pd.Timestamp(end_date)]

    prices = prices.loc[idx, TRADE_COLS].copy()
    targets = targets.loc[idx, TRADE_COLS].copy()

    if len(idx) < 20:
        raise ValueError(f"too few overlapping dates: {len(idx)}")

    return prices, targets


def simulate_mode1_full_next_day(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    idx = prices.index

    first_target = normalize_target(targets.iloc[0])
    weights = dict(first_target)

    curve = []
    logs = []

    for i, dt in enumerate(idx):
        daily_ret = apply_returns(weights, returns.loc[dt])
        equity_prev = curve[-1] if curve else 1.0
        equity = equity_prev * (1.0 + daily_ret)

        executed = False
        cost = 0.0
        target_date = None

        if i >= 1:
            signal_dt = idx[i - 1]
            target = normalize_target(targets.loc[signal_dt])
            weights, cost = rebalance_full(weights, target, buy_cost, sell_cost)
            equity *= (1.0 - cost)
            executed = True
            target_date = str(signal_dt.date())

        curve.append(equity)
        logs.append(
            {
                "date": str(dt.date()),
                "executed": executed,
                "signal_date_used": target_date or "",
                "cost_frac": cost,
                **{f"w_{k}": weights.get(k, 0.0) for k in [*TRADE_COLS, CASH]},
            }
        )

    return pd.Series(curve, index=idx, name="equity"), pd.DataFrame(logs)


def simulate_mode2_sell_then_buy(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
) -> tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    idx = prices.index

    first_target = normalize_target(targets.iloc[0])
    weights = dict(first_target)

    curve = []
    logs = []

    for i, dt in enumerate(idx):
        daily_ret = apply_returns(weights, returns.loc[dt])
        equity_prev = curve[-1] if curve else 1.0
        equity = equity_prev * (1.0 + daily_ret)

        sell_cost_frac = 0.0
        buy_cost_frac = 0.0
        sell_signal_date = ""
        buy_signal_date = ""

        if i >= 1:
            signal_dt_for_sell = idx[i - 1]
            target_sell = normalize_target(targets.loc[signal_dt_for_sell])
            weights, sell_cost_frac = rebalance_sell_only(weights, target_sell, sell_cost)
            equity *= (1.0 - sell_cost_frac)
            sell_signal_date = str(signal_dt_for_sell.date())

        if i >= 2:
            signal_dt_for_buy = idx[i - 2]
            target_buy = normalize_target(targets.loc[signal_dt_for_buy])
            weights, buy_cost_frac = rebalance_buy_only(weights, target_buy, buy_cost)
            equity *= (1.0 - buy_cost_frac)
            buy_signal_date = str(signal_dt_for_buy.date())

        curve.append(equity)
        logs.append(
            {
                "date": str(dt.date()),
                "sell_signal_date_used": sell_signal_date,
                "buy_signal_date_used": buy_signal_date,
                "sell_cost_frac": sell_cost_frac,
                "buy_cost_frac": buy_cost_frac,
                **{f"w_{k}": weights.get(k, 0.0) for k in [*TRADE_COLS, CASH]},
            }
        )

    return pd.Series(curve, index=idx, name="equity"), pd.DataFrame(logs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-csv", required=True)
    parser.add_argument("--targets-csv", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--buy-cost", type=float, default=0.0020)
    parser.add_argument("--sell-cost", type=float, default=0.0020)
    parser.add_argument("--out-dir", default="out/compare_execution_modes")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, targets = prepare_inputs(
        prices_csv=args.prices_csv,
        targets_csv=args.targets_csv,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
    )

    eq1, log1 = simulate_mode1_full_next_day(
        prices=prices,
        targets=targets,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
    )
    eq2, log2 = simulate_mode2_sell_then_buy(
        prices=prices,
        targets=targets,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
    )

    m1 = compute_metrics(eq1)
    m2 = compute_metrics(eq2)

    summary = pd.DataFrame(
        [
            {"mode": "mode1_next_day_full", **m1},
            {"mode": "mode2_next_day_sell_then_next_day_buy", **m2},
            {
                "mode": "delta_mode2_minus_mode1",
                "cagr": m2["cagr"] - m1["cagr"],
                "mdd": m2["mdd"] - m1["mdd"],
                "seed_multiple": m2["seed_multiple"] - m1["seed_multiple"],
                "recovery_count": m2["recovery_count"] - m1["recovery_count"],
                "avg_recovery_days": m2["avg_recovery_days"] - m1["avg_recovery_days"],
                "median_recovery_days": m2["median_recovery_days"] - m1["median_recovery_days"],
                "p90_recovery_days": m2["p90_recovery_days"] - m1["p90_recovery_days"],
                "max_recovery_days": m2["max_recovery_days"] - m1["max_recovery_days"],
            },
        ]
    )

    eq1.to_csv(out_dir / "equity_mode1.csv", header=True)
    eq2.to_csv(out_dir / "equity_mode2.csv", header=True)
    log1.to_csv(out_dir / "log_mode1.csv", index=False)
    log2.to_csv(out_dir / "log_mode2.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "prices_csv": args.prices_csv,
                "targets_csv": args.targets_csv,
                "rows": int(len(prices)),
                "start_date": str(prices.index.min().date()),
                "end_date": str(prices.index.max().date()),
                "buy_cost": args.buy_cost,
                "sell_cost": args.sell_cost,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(summary.to_string(index=False))
    print(f"saved: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()