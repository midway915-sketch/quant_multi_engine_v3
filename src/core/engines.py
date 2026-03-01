from __future__ import annotations
import pandas as pd
import numpy as np


def trend_pick(prices: pd.DataFrame, cfg) -> tuple[str, float]:
    uni = cfg["trend_engine"]["universe"]
    lb = int(cfg["trend_engine"]["mom_lookback_days"])
    # momentum at t uses t-1 via shift in caller; 여기서는 raw 모멘텀만 계산
    mom = prices[uni].pct_change(lb)
    return mom


def run_trend_engine(prices: pd.DataFrame, cfg, state_df: pd.DataFrame):
    """
    Returns:
      trend_equity: pd.Series
      picks_log: pd.DataFrame (weekly rebal dates)
    Execution: proxy 3x tickers
    """
    uni = cfg["trend_engine"]["universe"]
    lb = int(cfg["trend_engine"]["mom_lookback_days"])

    # signal (look-ahead safe)
    scores = prices[uni].pct_change(lb).shift(1)

    # weekly rebal dates
    rebal = _weekly_dates(prices.index)

    # mapping base->proxy traded
    mp = {"QQQ": "TQQQ_PROXY", "SPY": "UPRO_PROXY", "SOXX": "SOXL_PROXY"}

    held = None
    equity = 1.0
    curve = []

    rows = []
    rets = prices.pct_change().fillna(0.0)

    for d in prices.index:
        if d in rebal:
            row = scores.loc[d]
            if row.isna().all():
                held = None
            else:
                base_top = row.idxmax()
                held = mp.get(base_top, base_top)  # proxy series exists in data.py

            rows.append({
                "date": d,
                "state": state_df.loc[d, "state"],
                "base_top1": "" if held is None else _reverse_map(mp, held),
                "traded_top1": "" if held is None else held,
                "score": float(row.max()) if not row.isna().all() else np.nan
            })

        daily = 0.0
        if held and held in rets.columns:
            daily = float(rets.loc[d, held])
        equity *= (1.0 + daily)
        curve.append(equity)

    eq = pd.Series(curve, index=prices.index)
    picks = pd.DataFrame(rows).set_index("date")
    return eq, picks


def run_defensive_engine(prices: pd.DataFrame, cfg, state_df: pd.DataFrame):
    """
    Defensive engine equity:
      mode=SHY -> hold SHY returns
      mode=CASH -> 0% return
    """
    mode = cfg["defensive_engine"]["mode"]
    rets = prices.pct_change().fillna(0.0)

    equity = 1.0
    curve = []

    for d in prices.index:
        daily = 0.0
        if mode == "SHY":
            daily = float(rets.loc[d, "SHY"]) if "SHY" in rets.columns else 0.0
        elif mode == "CASH":
            daily = 0.0
        equity *= (1.0 + daily)
        curve.append(equity)

    return pd.Series(curve, index=prices.index)


def run_meanrev_engine(prices: pd.DataFrame, cfg, state_df: pd.DataFrame):
    """
    Mean reversion on SPY:
      if lookback return <= threshold -> enter next day, hold N days or TP/SL
    Execution uses SPY_1P5_PROXY series for leverage research mode.
    """
    base = cfg["meanrev_engine"]["base_ticker"]
    lb = int(cfg["meanrev_engine"]["lookback_days"])
    thr = float(cfg["meanrev_engine"]["drop_threshold"])
    hold_days = int(cfg["meanrev_engine"]["hold_days"])
    tp = float(cfg["meanrev_engine"]["take_profit"])
    sl = float(cfg["meanrev_engine"]["stop_loss"])

    traded = "SPY_1P5_PROXY"  # built in data.py
    if traded not in prices.columns:
        raise ValueError("SPY_1P5_PROXY missing; check data.py")

    rets = prices.pct_change().fillna(0.0)
    sig = prices[base].pct_change(lb).shift(1) <= thr

    equity = 1.0
    curve = []
    in_pos = False
    entry_eq = None
    days_in = 0

    for d in prices.index:
        # exit logic
        if in_pos:
            days_in += 1
            # current PnL since entry (on equity)
            pnl = equity / entry_eq - 1.0 if entry_eq else 0.0
            if pnl >= tp or pnl <= sl or days_in >= hold_days:
                in_pos = False
                entry_eq = None
                days_in = 0

        # entry logic (next bar effect approximated by using shifted signal)
        if (not in_pos) and bool(sig.loc[d]):
            in_pos = True
            entry_eq = equity
            days_in = 0

        daily = float(rets.loc[d, traded]) if in_pos else 0.0
        equity *= (1.0 + daily)
        curve.append(equity)

    return pd.Series(curve, index=prices.index)


def _weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=index)
    w = s.resample("W-FRI").last().dropna().index
    w = w[w.isin(index)]
    return w


def _reverse_map(mp: dict, traded: str) -> str:
    for k, v in mp.items():
        if v == traded:
            return k
    return traded