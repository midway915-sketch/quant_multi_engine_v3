from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def rank_summary(
    summary: pd.DataFrame,
    method: str,
    balanced_mdd_weight: float,
    balanced_recovery_weight: float,
) -> pd.DataFrame:
    method = str(method).lower().strip()
    df = summary.copy()

    required = {"cagr", "mdd", "max_recovery_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary csv missing required columns: {sorted(missing)}")

    if method == "cagr":
        return df.sort_values(
            by=["cagr", "mdd", "max_recovery_days"],
            ascending=[False, False, True],
        )

    if method == "recovery":
        return df.sort_values(
            by=["max_recovery_days", "mdd", "cagr"],
            ascending=[True, False, False],
        )

    if method == "balanced":
        df["score_balanced"] = (
            df["cagr"].astype(float)
            - float(balanced_mdd_weight) * df["mdd"].astype(float).abs()
            - float(balanced_recovery_weight) * df["max_recovery_days"].astype(float)
        )
        return df.sort_values(
            by=["score_balanced", "cagr", "mdd", "max_recovery_days"],
            ascending=[False, False, False, True],
        )

    raise ValueError(f"unsupported method: {method}")


def find_equity_path_from_run_dir(run_dir: Path) -> Path:
    candidates = [
        run_dir / "equity_curve.csv",
        run_dir / "equity.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = list(run_dir.rglob("equity_curve.csv"))
    if matches:
        return matches[0]

    matches = list(run_dir.rglob("equity.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"equity curve file not found under: {run_dir}")


def find_best_run_dir_by_summary_file(runs_root: Path, best_row: dict) -> Path:
    best_cagr = float(best_row["cagr"])
    best_mdd = float(best_row["mdd"])
    best_rec = int(best_row["max_recovery_days"])

    summary_files = list(runs_root.rglob("summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No summary.csv files found under {runs_root}")

    for sf in summary_files:
        try:
            df = pd.read_csv(sf)
        except Exception:
            continue
        if df.empty:
            continue

        row = df.iloc[0].to_dict()
        if "cagr" not in row or "mdd" not in row or "max_recovery_days" not in row:
            continue

        cagr_ok = abs(float(row["cagr"]) - best_cagr) < 1e-12
        mdd_ok = abs(float(row["mdd"]) - best_mdd) < 1e-12
        rec_ok = int(row["max_recovery_days"]) == best_rec

        if cagr_ok and mdd_ok and rec_ok:
            return sf.parent

    raise FileNotFoundError(
        "Could not match best row from merged summary to any run summary.csv under runs_root"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--run-dir-column", default="run_dir")
    parser.add_argument("--method", default="cagr", choices=["cagr", "recovery", "balanced"])
    parser.add_argument("--balanced-mdd-weight", type=float, default=0.35)
    parser.add_argument("--balanced-recovery-weight", type=float, default=0.00015)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise ValueError("summary csv is empty")

    ranked = rank_summary(
        summary=summary,
        method=args.method,
        balanced_mdd_weight=args.balanced_mdd_weight,
        balanced_recovery_weight=args.balanced_recovery_weight,
    )
    best = ranked.iloc[0].to_dict()

    runs_root = Path(args.runs_root)
    run_dir_col = args.run_dir_column

    if run_dir_col in best and pd.notna(best[run_dir_col]) and str(best[run_dir_col]).strip():
        run_dir = runs_root / str(best[run_dir_col])
    else:
        run_dir = find_best_run_dir_by_summary_file(runs_root, best)

    equity_path = find_equity_path_from_run_dir(run_dir)

    payload = {
        "selection_method": args.method,
        "balanced_mdd_weight": float(args.balanced_mdd_weight),
        "balanced_recovery_weight": float(args.balanced_recovery_weight),
        "best_params": best,
        "run_dir": str(run_dir),
        "equity_path": str(equity_path),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()