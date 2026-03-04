from __future__ import annotations
import argparse
import datetime as dt
import os
import json
import yaml

from src.live.calendar_us import is_last_trading_day_today, is_first_trading_day_today
from src.live.data_yf import download_daily_closes
from src.live.signal import compute_state, pick_trend_top1, build_targets
from src.live.notify_telegram import send_telegram

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/live.yml")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (optional, for testing)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    mode = cfg.get("mode", "dry_run")

    if args.date:
        today = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        today = dt.date.today()

    when = cfg["schedule"]["when"]
    if when == "last_trading_day_close":
        rebalance_today = is_last_trading_day_today(today)
    elif when == "first_trading_day_open":
        rebalance_today = is_first_trading_day_today(today)
    else:
        raise ValueError(f"Unknown schedule.when: {when}")

    # 데이터: 룩백만큼만 받기 위해 start를 넉넉히 잡음(대략 1년+)
    # ma=150, mom=126, crash=12이므로 400영업일 정도면 충분 -> 2년 정도 확보
    start = (today - dt.timedelta(days=900)).strftime("%Y-%m-%d")
    tickers = cfg["data"]["tickers"]
    prices = download_daily_closes(tickers, start=start)

    # 스모크 체크: 최신 날짜 확인
    last_dt = prices.index.max().date()
    # 오늘이 거래일이 아닐 수도 있으니(주말/휴장) last_dt가 today보다 작아도 허용.
    # 대신 너무 오래되면 위험 -> 경고/중단
    stale_days = (today - last_dt).days
    if stale_days > 7:
        msg = f"[LIVE] DATA STALE: today={today} last_price_date={last_dt} stale_days={stale_days} -> ABORT"
        print(msg)
        send_telegram(cfg, msg)
        return

    # state + targets
    state = compute_state(cfg, prices, today)
    top1 = None
    targets = None
    if state == "BULL":
        top1 = pick_trend_top1(cfg, prices)
        targets = build_targets(cfg, state, top1)
    else:
        # bear/crash
        top1 = None
        targets = build_targets(cfg, state, top_universe="SPY")  # ignored in bear/crash

    out_root = cfg.get("paths", {}).get("out_dir", "out/live_runs")
    ensure_dir(out_root)
    out_dir = os.path.join(out_root, str(today))
    ensure_dir(out_dir)

    payload = {
        "date": str(today),
        "last_price_date": str(last_dt),
        "rebalance_today": bool(rebalance_today),
        "state": state,
        "top1_universe": top1 or "",
        "targets": targets,
        "mode": mode,
        "order_type": cfg.get("execution", {}).get("order_type", "MOC"),
    }

    with open(os.path.join(out_dir, "live_payload.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 텔레그램: 매일은 스팸일 수 있으니 리밸런스 날에만 알림 기본
    if rebalance_today:
        msg = (
            f"[LIVE] Rebalance Day ✅ {today}\n"
            f"state={state}\n"
            f"top1={payload['top1_universe']}\n"
            f"targets={payload['targets']}\n"
            f"order={payload['order_type']}\n"
            f"last_price_date={last_dt}"
        )
        print(msg)
        send_telegram(cfg, msg)
    else:
        print(f"[LIVE] {today} no rebalance. state={state} last_price_date={last_dt}")

    # TODO: mode == "live" 이면 여기서 KIS 주문 실행(REST) 붙이기
    # - MOC(33)로만 단순하게 시작
    # - 유량 제한/재시도/중복방지(run_lock) 추가

if __name__ == "__main__":
    main()