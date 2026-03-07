from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
    """
    run_dir 컬럼이 없을 때 fallback:
    runs_root 아래 모든 summary.csv를 뒤져서
    cagr/mdd/max_recovery_days가 best_row와 일치하는 run 폴더를 찾는다.
    """
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
    parser.add_argument("--summary-csv", required=True, help="기존 멀티엔진 merged summary csv")
    parser.add_argument("--runs-root", required=True, help="run 결과 폴더 루트")
    parser.add_argument("--out-json", required=True, help="best equity path json 출력 경로")
    parser.add_argument("--run-dir-column", default="run_dir", help="summary csv 안 run 폴더 컬럼명")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if summary.empty:
        raise ValueError("summary csv is empty")

    best = summary.sort_values(
        by=["cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()

    runs_root = Path(args.runs_root)
    run_dir_col = args.run_dir_column

    if run_dir_col in best and pd.notna(best[run_dir_col]) and str(best[run_dir_col]).strip():
        run_dir = runs_root / str(best[run_dir_col])
    else:
        run_dir = find_best_run_dir_by_summary_file(runs_root, best)

    equity_path = find_equity_path_from_run_dir(run_dir)

    payload = {
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