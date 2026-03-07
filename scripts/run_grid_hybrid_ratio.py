from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def compute_max_recovery_days(equity: pd.Series) -> int:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return 0

    running_peak = equity.cummax()
    underwater = equity < running_peak

    max_len = 0
    cur_len = 0
    for flag in underwater:
        if bool(flag):
            cur_len += 1
            max_len = max(max_len, cur_len)
        else:
            cur_len = 0
    return int(max_len)


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return {
            "cagr": float("nan"),
            "mdd": float("nan"),
            "max_recovery_days": 0,
            "seed_multiple": float("nan"),
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
        "max_recovery_days": compute_max_recovery_days(equity),
        "seed_multiple": seed_multiple,
    }


def compute_recent_10y_metrics(equity: pd.Series) -> dict:
    equity = equity.astype(float).dropna()
    if equity.empty:
        return {
            "cagr": float("nan"),
            "mdd": float("nan"),
            "max_recovery_days": 0,
            "seed_multiple": float("nan"),
        }

    end_dt = equity.index[-1]
    start_dt = end_dt - pd.DateOffset(years=10)
    sub = equity.loc[equity.index >= start_dt].copy()
    if sub.empty:
        sub = equity.copy()
    return compute_metrics(sub)


def load_equity_curve(csv_path: str | Path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df.shape[1] == 1:
        s = df.iloc[:, 0].copy()
    else:
        s = df["equity"].copy() if "equity" in df.columns else df.iloc[:, 0].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().astype(float).dropna()
    s.name = "equity"
    return s


def build_hybrid_equity(
    core_equity: pd.Series,
    satellite_equity: pd.Series,
    core_weight: float,
    satellite_weight: float,
) -> pd.Series:
    if abs((core_weight + satellite_weight) - 1.0) > 1e-9:
        raise ValueError("core_weight + satellite_weight must equal 1.0")

    idx = core_equity.index.intersection(satellite_equity.index)
    if len(idx) < 2:
        raise ValueError("두 equity curve의 겹치는 날짜가 너무 적음")

    core_eq = core_equity.loc[idx].copy()
    sat_eq = satellite_equity.loc[idx].copy()

    core_ret = core_eq.pct_change().fillna(0.0)
    sat_ret = sat_eq.pct_change().fillna(0.0)

    hybrid_ret = core_weight * core_ret + satellite_weight * sat_ret
    hybrid_eq = (1.0 + hybrid_ret).cumprod()
    hybrid_eq.name = "equity"
    return hybrid_eq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-equity-csv", required=True)
    parser.add_argument("--satellite-equity-csv", required=True)
    parser.add_argument("--grid-yaml", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.grid_yaml).read_text(encoding="utf-8"))
    core_weights = cfg.get("core_weight", [0.8])
    sat_weights = cfg.get("satellite_weight", None)

    if sat_weights is None:
        sat_weights = [round(1.0 - float(w), 10) for w in core_weights]

    if len(core_weights) != len(sat_weights):
        raise ValueError("core_weight와 satellite_weight 길이가 같아야 함")

    core_equity = load_equity_curve(args.core_equity_csv)
    satellite_equity = load_equity_curve(args.satellite_equity_csv)

    all_rows = []

    for core_w, sat_w in zip(core_weights, sat_weights):
        core_w = float(core_w)
        sat_w = float(sat_w)

        if core_w < 0 or sat_w < 0:
            raise ValueError("비중은 음수가 될 수 없음")
        if abs((core_w + sat_w) - 1.0) > 1e-9:
            raise ValueError(f"비중 합이 1이 아님: core={core_w}, sat={sat_w}")

        run_name = f"core_{str(core_w).replace('.', 'p')}__sat_{str(sat_w).replace('.', 'p')}"
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        hybrid_eq = build_hybrid_equity(
            core_equity=core_equity,
            satellite_equity=satellite_equity,
            core_weight=core_w,
            satellite_weight=sat_w,
        )

        full = compute_metrics(hybrid_eq)
        recent10 = compute_recent_10y_metrics(hybrid_eq)

        row = {
            "core_weight": core_w,
            "satellite_weight": sat_w,
            "cagr": full["cagr"],
            "mdd": full["mdd"],
            "max_recovery_days": full["max_recovery_days"],
            "seed_multiple": full["seed_multiple"],
            "cagr_10y": recent10["cagr"],
            "mdd_10y": recent10["mdd"],
            "max_recovery_10y_days": recent10["max_recovery_days"],
            "seed_multiple_10y": recent10["seed_multiple"],
        }
        all_rows.append(row)

        hybrid_eq.to_csv(run_dir / "equity_curve.csv", header=True)
        pd.DataFrame([row]).to_csv(run_dir / "summary.csv", index=False)
        (run_dir / "metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    merged = pd.DataFrame(all_rows).sort_values(
        by=["cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, True],
    )
    merged.to_csv(out_dir / "summary_merged.csv", index=False)

    if not merged.empty:
        best = merged.iloc[0].to_dict()
        (out_dir / "best.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(merged)


if __name__ == "__main__":
    main()