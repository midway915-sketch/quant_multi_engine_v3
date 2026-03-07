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

    # 혹시 하위 폴더에 있는 경우까지 탐색
    matches = list(run_dir.rglob("equity_curve.csv"))
    if matches:
        return matches[0]

    matches = list(run_dir.rglob("equity.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"equity curve file not found under: {run_dir}")


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

    # 기존 정렬 기준 그대로
    best = summary.sort_values(
        by=["cagr", "mdd", "max_recovery_days"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()

    run_dir_col = args.run_dir_column
    runs_root = Path(args.runs_root)

    if run_dir_col in best and pd.notna(best[run_dir_col]):
        run_dir = runs_root / str(best[run_dir_col])
    else:
        # run_dir 컬럼이 없는 경우를 대비해 가장 흔한 패턴들 추정
        # summary에 아래 컬럼이 있으면 조합명 생성
        if {"lookback", "rebalance", "top1_weight"}.issubset(summary.columns):
            run_name = f"lb_{int(best['lookback'])}__reb_{best['rebalance']}__w1_{str(float(best['top1_weight'])).replace('.', 'p')}"
            run_dir = runs_root / run_name
        else:
            # 마지막 fallback: runs_root 아래에서 summary 정렬 1등과 대응되는 첫 equity 파일 탐색
            matches = list(runs_root.rglob("equity_curve.csv"))
            if not matches:
                raise FileNotFoundError(f"run_dir column '{run_dir_col}' not found and no equity_curve.csv under {runs_root}")
            run_dir = matches[0].parent

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