import argparse
import datetime as dt
import yaml

from src.live.notify_telegram import send_telegram

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def is_us_trading_day(d: dt.date) -> bool:
    # 최소 버전: 주말만 제외 (정확한 휴장일은 나중에 캘린더로 보강)
    return d.weekday() < 5

def first_trading_day_of_month(d: dt.date) -> dt.date:
    cur = dt.date(d.year, d.month, 1)
    while not is_us_trading_day(cur):
        cur += dt.timedelta(days=1)
    return cur

def last_trading_day_of_month(d: dt.date) -> dt.date:
    # 다음달 1일 - 1일에서 주말만 보정
    if d.month == 12:
        nxt = dt.date(d.year + 1, 1, 1)
    else:
        nxt = dt.date(d.year, d.month + 1, 1)
    cur = nxt - dt.timedelta(days=1)
    while not is_us_trading_day(cur):
        cur -= dt.timedelta(days=1)
    return cur

def should_rebalance(cfg: dict, today: dt.date) -> bool:
    sched = cfg.get("schedule", {})
    if sched.get("rebalance") != "monthly":
        raise ValueError("Only monthly supported for now")

    when = sched.get("when", "first_trading_day_open")
    if when == "first_trading_day_open":
        return today == first_trading_day_of_month(today)
    if when == "last_trading_day_close":
        return today == last_trading_day_of_month(today)
    raise ValueError(f"Unknown schedule.when: {when}")

def pick_assets(cfg: dict) -> dict:
    """
    TODO: 여기서 네 기존 코어 로직을 호출해:
    - 레짐(bull/bear/crash)
    - trend top_n 선정
    - defensive(SGOV) 처리
    - allocator로 목표 비중 산출

    지금은 골격만: trend 100%로 QQQ/SPY/SOXX 중 1개를 임시로 선택
    """
    candidates = cfg["universe"]["candidates"]
    chosen = candidates[0]  # TODO: mom 계산으로 top 1 고르기
    return {chosen: 1.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/live.yml")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (optional)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if args.date:
        today = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        # 서버 로컬 날짜
        today = dt.date.today()

    if not should_rebalance(cfg, today):
        print(f"[LIVE] {today} -> no rebalance day")
        return

    targets = pick_assets(cfg)
    mode = cfg.get("mode", "dry_run")

    msg = (
        f"[LIVE] Rebalance day: {today}\n"
        f"mode={mode}\n"
        f"targets={targets}\n"
        f"order_type={cfg.get('execution', {}).get('rebalance_order', 'MOC')}"
    )
    print(msg)
    send_telegram(cfg, msg)

    # TODO: mode가 live면 여기서 KIS 주문 실행 (MOC 등)
    # - REST rate limit/재시도/백오프
    # - 웹소켓은 나중(너가 올린 경고처럼 무한 재연결 금지)

if __name__ == "__main__":
    main()