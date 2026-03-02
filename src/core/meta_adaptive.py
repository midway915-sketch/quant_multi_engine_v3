from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.risk_off import risk_off_weights
from src.core.state_adaptive import compute_state_flags_adaptive
from src.core.adaptive_rules import choose_bull_weights, filter_trend_picks


def _risk_off_weights(mode: str) -> dict:
    """Backward-compatible wrapper around src.core.risk_off.risk_off_weights."""
    try:
        w = risk_off_weights(mode)
        if isinstance(w, dict) and len(w) > 0:
            return {str(k): float(v) for k, v in w.items()}
    except Exception:
        pass
    return {"SHY": 1.0}


def _trend_universe_to_trade_col(ticker: str) -> str:
    # Trend는 3x MIX
    if ticker == "QQQ":
        return "TQQQ_MIX"
    if ticker == "SPY":
        return "UPRO_MIX"
    if ticker == "SOXX":
        return "SOXL_MIX"
    return ticker


def _meanrev_universe_to_trade_col(ticker: str) -> str:
    # MeanRev은 2x MIX (예: 1.5x 대신 2x mix를 쓰는 구조)
    if ticker == "QQQ":
        return "QLD_MIX"
    if ticker == "SPY":
        return "SSO_MIX"
    if ticker == "SOXX":
        return "USD_MIX"
    return ticker


def _week_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return "week end trading days" (W-FRI end) that actually exist in idx.
    """
    df = pd.DataFrame(index=idx)
    df["week_end"] = df.index.to_period("W-FRI").end_time.normalize()
    wk = pd.DatetimeIndex(df["week_end"].unique()).sort_values()
    wk = wk[wk.isin(idx)]
    return wk


def _safe_get_return(returns: pd.DataFrame, dt, ticker_col: str) -> float:
    if ticker_col not in returns.columns:
        return 0.0
    r = returns.at[dt, ticker_col]
    if pd.isna(r):
        return 0.0
    return float(r)


def _align_to_last_trading_day(px_index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    """
    Given target timestamp ts, return the last trading day in px_index that is <= ts.
    If none exists, return None.
    """
    ts = pd.Timestamp(ts).normalize()
    if len(px_index) == 0:
        return None

    # fast path
    if ts in px_index:
        return ts

    pos = px_index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return px_index[pos]


def _calc_week_ret(px: pd.DataFrame, week_end: pd.Timestamp, ticker_col: str) -> float:
    """
    Robust weekly return:
    - week_end might be a holiday (not in index); align to last trading day <= week_end.
    - previous day is the trading day immediately before that aligned day.
    """
    if ticker_col not in px.columns:
        return np.nan

    we = _align_to_last_trading_day(px.index, pd.Timestamp(week_end))
    if we is None:
        return np.nan

    loc = px.index.get_loc(we)
    if isinstance(loc, slice):
        loc = loc.stop - 1

    if loc <= 0:
        return np.nan

    prev = px.iloc[loc - 1][ticker_col]
    cur = px.iloc[loc][ticker_col]
    if pd.isna(prev) or prev == 0 or pd.isna(cur):
        return np.nan

    return float(cur / prev - 1.0)


def run_meta_portfolio(prices: pd.DataFrame, cfg: dict):
    """
    Lookahead-free:
      - Today's return uses yesterday's confirmed holdings
      - Signals/picks computed at date dt are applied starting next trading day

    Returns:
      equity: pd.Series
      engine_choice_log: list[dict]
      picks_top2_weekly: pd.DataFrame
      holdings_daily: pd.DataFrame  (date,ticker,weight,state,vol_regime)
      holdings_weekly: pd.DataFrame (week_end_trade,ticker,avg_weight,week_ret,contrib)
    """
    prices = prices.copy()
    returns = prices.pct_change().fillna(0.0)

    state_df = compute_state_flags_adaptive(prices, cfg)

    # ---- Trend config ----
    trend_cfg = cfg.get("trend_engine", {}) or {}
    mom_lb = int(trend_cfg.get("mom_lookback_days", 168))
    candidates = trend_cfg.get("candidates", ["QQQ", "SPY", "SOXX"])
    top_n = int(trend_cfg.get("top_n", 1))

    mom = prices.pct_change(mom_lb)

    # ---- MeanRev config ----
    mr_cfg = cfg.get("meanrev_engine", {}) or {}
    mr_lb = int(mr_cfg.get("lookback_days", 20))
    mr_drop = float(mr_cfg.get("drop_threshold", -0.12))
    mr_hold = int(mr_cfg.get("hold_days", 5))
    mr_tp = float(mr_cfg.get("take_profit", 0.10))
    mr_sl = float(mr_cfg.get("stop_loss", -0.08))
    mr_candidates = mr_cfg.get("candidates", candidates)

    mr_base = mr_cfg.get("base", mr_cfg.get("base_ticker", "SPY"))
    if isinstance(mr_base, list) and len(mr_base) > 0:
        mr_base = mr_base[0]
    if not isinstance(mr_base, str):
        mr_base = "SPY"

    # ---- Allocator / defensive ----
    alloc = cfg["allocator"]
    risk_off_mode = (cfg.get("risk_off", {}) or {}).get("mode", "SHY_100")

    reb_dates = prices.index[prices.index.isin(_week_end_index(prices.index))]

    # --- Trend picks state ---
    current_trend_tradecols: list[str] = []
    picks_rows: list[dict] = []

    # --- MeanRev position state ---
    mr_active = False
    mr_entry_price = None
    mr_days = 0
    mr_trade_col = None

    # --- Holdings (lookahead-free) ---
    h_cur = {"SHY": 1.0}
    h_next = h_cur.copy()

    equity = 1.0
    curve: list[dict] = []
    engine_choice_log: list[dict] = []
    holdings_rows: list[dict] = []

    buy_cost = float((cfg.get("costs", {}) or {}).get("buy", 0.0))
    sell_cost = float((cfg.get("costs", {}) or {}).get("sell", 0.0))

    for dt in prices.index:
        # ---- apply today's return based on yesterday-confirmed holdings (h_cur) ----
        day_ret = 0.0
        for tcol, w in h_cur.items():
            day_ret += w * _safe_get_return(returns, dt, tcol)

        equity *= (1.0 + day_ret)
        curve.append({"date": dt, "equity": equity})

        st = str(state_df.at[dt, "state"])
        vol_reg = str(state_df.at[dt, "vol_regime"]) if "vol_regime" in state_df.columns else "NORMAL"

        # ---- compute signals / desired holdings for TOMORROW ----
        if dt in reb_dates:
            m = mom.loc[dt, candidates]
            ranked = m.dropna().sort_values(ascending=False)
            if ranked.empty:
                top = []
                top1, top2 = None, None
                current_trend_tradecols = []
            else:
                top = list(ranked.index[:top_n])
                top = filter_trend_picks(top, m, cfg, vol_reg)
                top1 = top[0] if len(top) > 0 else None
                top2 = top[1] if len(top) > 1 else (ranked.index[1] if len(ranked) > 1 else None)
                current_trend_tradecols = [_trend_universe_to_trade_col(t) for t in top if t is not None]

            picks_rows.append(
                {
                    "week_end_trade": str(pd.Timestamp(dt).date()),
                    "top1": top1,
                    "top2": top2,
                    "mom_lb": mom_lb,
                    "vol_regime": vol_reg,
                }
            )

        # ---- MeanRev signal ----
        mr_signal = False
        mr_exit = False

        if mr_base in prices.columns:
            base_px = prices[mr_base]
            loc = base_px.index.get_loc(dt)
            if isinstance(loc, slice):
                loc = loc.stop - 1
            if loc >= mr_lb:
                prev = base_px.iloc[loc - mr_lb]
                cur = base_px.iloc[loc]
                if pd.notna(prev) and prev != 0 and pd.notna(cur):
                    drop = float(cur / prev - 1.0)
                    mr_signal = (drop <= mr_drop)

        if mr_active:
            mr_days += 1
            if mr_trade_col and mr_trade_col in prices.columns:
                curp = float(prices.at[dt, mr_trade_col])
                if mr_entry_price and mr_entry_price != 0:
                    pnl = curp / mr_entry_price - 1.0
                    if pnl >= mr_tp or pnl <= mr_sl:
                        mr_exit = True
            if mr_days >= mr_hold:
                mr_exit = True

        st_key = st.lower()

        # Adaptive bull weights
        if (
            st == "BULL"
            and bool((cfg.get("adaptive", {}) or {}).get("enabled", True))
            and bool((cfg.get("adaptive", {}) or {}).get("override_bull_allocator", True))
        ):
            plan = choose_bull_weights(vol_reg, cfg)
            w_tr, w_mr, w_df = float(plan.trend), float(plan.meanrev), float(plan.defensive)
        else:
            w_tr = float(alloc[st_key]["trend"])
            w_mr = float(alloc[st_key]["meanrev"])
            w_df = float(alloc[st_key]["defensive"])

        df_w = _risk_off_weights(risk_off_mode)
        h_des: dict[str, float] = {}

        # Trend slice
        if w_tr > 0 and len(current_trend_tradecols) > 0:
            per = w_tr / len(current_trend_tradecols)
            for tcol in current_trend_tradecols:
                h_des[tcol] = h_des.get(tcol, 0.0) + per

        # MeanRev slice
        if w_mr > 0:
            if (not mr_active) and mr_signal:
                best = None
                if dt in reb_dates:
                    mm = mom.loc[dt, mr_candidates]
                    rr = mm.dropna().sort_values(ascending=False)
                    best = rr.index[0] if len(rr) > 0 else None
                if best is None:
                    best = "QQQ"
                mr_trade_col = _meanrev_universe_to_trade_col(str(best))
                mr_active = True
                mr_days = 0
                if mr_trade_col in prices.columns:
                    mr_entry_price = float(prices.at[dt, mr_trade_col])

            if mr_active and mr_trade_col is not None:
                h_des[mr_trade_col] = h_des.get(mr_trade_col, 0.0) + w_mr

            if mr_active and mr_exit:
                mr_active = False
                mr_entry_price = None
                mr_days = 0
                mr_trade_col = None

        # Defensive slice
        if w_df > 0:
            for k, v in df_w.items():
                h_des[k] = h_des.get(k, 0.0) + float(v) * w_df

        # turnover cost
        turnover = 0.0
        for k in set(h_cur.keys()) | set(h_des.keys()):
            turnover += abs(h_des.get(k, 0.0) - h_cur.get(k, 0.0))
        cost = (turnover * 0.5) * (buy_cost + sell_cost)

        h_next = h_des.copy()

        engine_choice_log.append(
            {
                "date": str(dt.date()),
                "state": st,
                "vol_regime": vol_reg,
                "w_trend": w_tr,
                "w_meanrev": w_mr,
                "w_defensive": w_df,
                "risk_off_mode": risk_off_mode,
                "turnover": turnover,
                "cost": cost,
            }
        )

        for tcol, w in h_cur.items():
            holdings_rows.append(
                {"date": str(dt.date()), "ticker": tcol, "weight": float(w), "state": st, "vol_regime": vol_reg}
            )

        equity *= (1.0 - cost)
        h_cur = h_next

    equity_series = pd.Series([x["equity"] for x in curve], index=[x["date"] for x in curve])
    equity_series.index = pd.to_datetime(equity_series.index)

    picks_top2_weekly = pd.DataFrame(picks_rows)
    holdings_daily = pd.DataFrame(holdings_rows)

    # ---- weekly holdings aggregation (week_end_trade) ----
    if holdings_daily.empty:
        holdings_weekly = pd.DataFrame(columns=["week_end_trade", "ticker", "avg_weight", "week_ret", "contrib"])
    else:
        hd = holdings_daily.copy()
        hd["date"] = pd.to_datetime(hd["date"])

        # compute the week_end "target" date, then align to last trading day
        targets = hd["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

        # vectorized align: use searchsorted over sorted prices.index
        idx = prices.index
        pos = idx.searchsorted(targets.values.astype("datetime64[ns]"), side="right") - 1
        pos = np.clip(pos, 0, len(idx) - 1)
        week_end_trade = idx[pos]
        hd["week_end_trade"] = week_end_trade

        grp = (
            hd.groupby(["week_end_trade", "ticker"], as_index=False)["weight"]
            .mean()
            .rename(columns={"weight": "avg_weight"})
        )

        wk_rows = []
        for _, r in grp.iterrows():
            we = pd.Timestamp(r["week_end_trade"])
            t = str(r["ticker"])
            w = float(r["avg_weight"])
            wret = _calc_week_ret(prices, we, t)
            contrib = w * (0.0 if pd.isna(wret) else float(wret))
            wk_rows.append(
                {"week_end_trade": str(we.date()), "ticker": t, "avg_weight": w, "week_ret": wret, "contrib": contrib}
            )
        holdings_weekly = pd.DataFrame(wk_rows)

    return equity_series, engine_choice_log, picks_top2_weekly, holdings_daily, holdings_weekly