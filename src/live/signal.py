from __future__ import annotations
import datetime as dt
from typing import Dict, Tuple
import pandas as pd

from src.live.state_store import load_state, save_state, get_last_state, get_last_change_date, set_state

def _normalize(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in w.items()}

def _compute_bull_flag(px_spy: pd.Series, ma_days: int) -> pd.Series:
    ma = px_spy.rolling(ma_days).mean()
    return (px_spy > ma)

def _apply_min_hold(today: dt.date, desired: str, prev: str | None, prev_change: dt.date | None, min_hold_days: int) -> Tuple[str, dt.date]:
    if prev is None or prev_change is None:
        return desired, today
    if desired == prev:
        return prev, prev_change
    # 상태 변경하려면 min_hold_days 경과 필요
    days = (today - prev_change).days
    if days < min_hold_days:
        return prev, prev_change
    return desired, today

def compute_state(cfg: dict, prices: pd.DataFrame, today: dt.date) -> str:
    st_cfg = cfg["strategy"]
    base = st_cfg["base_ticker"]
    ma_days = int(st_cfg["ma_days"])
    min_hold_days = int(st_cfg["min_hold_days"])

    crash_cfg = st_cfg["crash"]
    crash_lb = int(crash_cfg["lookback_days"])
    crash_th = float(crash_cfg["threshold"])

    if base not in prices.columns:
        raise ValueError(f"base_ticker not in prices: {base}")

    spy = prices[base].dropna()
    if len(spy) < max(ma_days + 5, crash_lb + 5):
        raise ValueError("Not enough price history for state")

    # 오늘(가장 최근 거래일) 기준으로 판단
    bull_flag = _compute_bull_flag(spy, ma_days).iloc[-1]

    # crash: 최근 crash_lb 수익률
    r_crash = float(spy.pct_change(crash_lb).iloc[-1])
    desired = "BULL" if bool(bull_flag) else "BEAR"
    if r_crash <= crash_th:
        desired = "CRASH"

    # min_hold 적용
    state_dir = cfg.get("paths", {}).get("state_dir", "out/live_state")
    st = load_state(state_dir)
    prev = get_last_state(st)
    prev_change = get_last_change_date(st)

    final, final_change = _apply_min_hold(today, desired, prev, prev_change, min_hold_days)
    st = set_state(st, final, final_change)
    save_state(state_dir, st)
    return final

def pick_trend_top1(cfg: dict, prices: pd.DataFrame) -> str:
    te = cfg["strategy"]["trend_engine"]
    lb = int(te["mom_lookback_days"])
    candidates = list(cfg["universe"]["candidates"])

    mom = prices[candidates].pct_change(lb).iloc[-1].dropna()
    if mom.empty:
        raise ValueError("Momentum scores are empty (not enough history or NaNs).")
    top = mom.sort_values(ascending=False).index[0]
    return str(top)

def build_targets(cfg: dict, state: str, top_universe: str) -> Dict[str, float]:
    alloc = cfg["allocator"]
    df = cfg["universe"]["defensive"]
    trade_map = cfg["universe"]["trade_map"]

    if state == "BULL":
        w_tr = float(alloc["bull"]["trend"])
        w_df = float(alloc["bull"]["defensive"])
        trade_ticker = trade_map[top_universe]
        tgt = {trade_ticker: w_tr, df: w_df}
    elif state == "BEAR":
        tgt = {df: 1.0}
    elif state == "CRASH":
        tgt = {df: 1.0}
    else:
        raise ValueError(f"Unknown state: {state}")

    tgt = _normalize(tgt)
    # sanity
    s = sum(tgt.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Target weights do not sum to 1: {s}")
    if any((v < 0) or pd.isna(v) for v in tgt.values()):
        raise ValueError(f"Bad target weights: {tgt}")
    return tgt