from __future__ import annotations
import pandas as pd
import numpy as np

from src.core.state import compute_state_flags


def _risk_off_weights(mode: str) -> dict:
    # Cash-like / short-duration
    if mode == "SHY_100":
        return {"SHY": 1.0}
    if mode == "BIL_100":
        return {"BIL_MIX": 1.0}
    if mode == "SGOV_100":
        return {"SGOV_MIX": 1.0}

    # Gold mixes
    if mode == "SHY_GLD_50_50":
        return {"SHY": 0.5, "GLD": 0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY": 0.7, "GLD": 0.3}
    if mode == "GLD_100":
        return {"GLD": 1.0}

    # Inverse equity (MIX = proxy pre-listing, real after listing)
    if mode == "SH_100":
        return {"SH_MIX": 1.0}
    if mode == "PSQ_100":
        return {"PSQ_MIX": 1.0}

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
    # MeanRev는 2x MIX
    if ticker == "QQQ":
        return "QLD_MIX"
    if ticker == "SPY":
        return "SSO_MIX"
    if ticker == "SOXX":
        return "USD_MIX"
    return ticker


def _week_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("W-FRI").to_timestamp("W-FRI")


def _weekly_close(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("W-FRI").last()


def _normalize_weights(h: dict) -> dict:
    s = float(sum(h.values()))
    if s <= 0:
        return {}
    return {k: float(v / s) for k, v in h.items()}


def run_meta_portfolio(prices: pd.DataFrame, cfg: dict):
    """
    Lookahead-free:
      - Today's return uses yesterday's confirmed holdings
      - Signals/picks computed at date dt are applied starting next trading day

    Returns:
      equity: pd.Series
      engine_choice_log: list[dict]
      picks_top2_weekly: pd.DataFrame
      holdings_daily: pd.DataFrame  (date,ticker,weight,state)
      holdings_weekly: pd.DataFrame (week_end,ticker,avg_weight,week_ret,contrib)
    """
    prices = prices.copy()
    returns = prices.pct_change().fillna(0.0)

    state_df = compute_state_flags(prices, cfg)

    # ---- Trend config (safe fallback) ----
    trend_cfg = cfg.get("trend_engine", {}) or {}
    mom_lb = int(trend_cfg.get("mom_lookback_days", 168))
    candidates = trend_cfg.get("candidates", ["QQQ", "SPY", "SOXX"])

    sel_cfg = cfg.get("selection", {}) or {}
    top_n = int(sel_cfg.get("top_n", trend_cfg.get("top_n", 1)))

    # Momentum (do NOT fillna with 0 to avoid early tie noise)
    mom = prices.pct_change(mom_lb)

    # ---- MeanRev config ----
    mr_cfg = cfg.get("meanrev_engine", {}) or {}
    mr_lb = int(mr_cfg.get("lookback_days", 20))
    mr_drop = float(mr_cfg.get("drop_threshold", -0.12))
    mr_hold = int(mr_cfg.get("hold_days", 5))
    mr_tp = float(mr_cfg.get("take_profit", 0.08))
    mr_sl = float(mr_cfg.get("stop_loss", -0.08))
    mr_candidates = mr_cfg.get("candidates", ["QQQ", "SPY", "SOXX"])
    mr_base = mr_cfg.get("base", "QQQ")
    if mr_base not in mr_candidates:
        mr_base = "QQQ"

    # ---- Allocator / defensive ----
    alloc = cfg["allocator"]
    risk_off_mode = (cfg.get("risk_off", {}) or {}).get("mode", "SHY_100")

    # Weekly rebalance dates (Fridays present in price index)
    week_end = _week_end_index(prices.index)
    reb_dates = prices.index[prices.index.isin(week_end.unique())]

    # Weekly closes for weekly return reporting
    wclose = _weekly_close(prices)

    def get_week_ret(ticker_col: str, wk_end: pd.Timestamp) -> float:
        if ticker_col not in wclose.columns or wk_end not in wclose.index:
            return np.nan
        loc = wclose.index.get_loc(wk_end)
        if loc == 0:
            return np.nan
        prev = wclose.iloc[loc - 1][ticker_col]
        cur = wclose.iloc[loc][ticker_col]
        if pd.isna(prev) or prev == 0 or pd.isna(cur):
            return np.nan
        return float(cur / prev - 1.0)

    # --- State for Trend picks ---
    current_trend_tradecols = []
    picks_rows = []

    # --- MeanRev position state (decision made at dt, applied dt+1) ---
    mr_active = False
    mr_entry_price = None
    mr_days = 0
    mr_trade_col = None

    # --- Holdings (lookahead-free): h_cur applies today, h_next applies tomorrow ---
    h_cur = {"SHY": 1.0}   # start parked in SHY
    h_next = h_cur.copy()

    equity = 1.0
    curve = []
    engine_choice_log = []
    holdings_daily_rows = []

    idx = prices.index

    for i, dt in enumerate(idx):
        st = state_df.loc[dt, "state"]

        # 1) ---- APPLY TODAY'S RETURN USING h_cur (yesterday-confirmed) ----
        # log today's holdings
        for t, w in h_cur.items():
            holdings_daily_rows.append({
                "date": str(dt.date()),
                "ticker": t,
                "weight": float(w),
                "state": st
            })

        daily_ret = 0.0
        for t, w in h_cur.items():
            if t in returns.columns:
                daily_ret += float(returns.loc[dt, t]) * float(w)

        equity *= (1.0 + daily_ret)
        curve.append(equity)

        # 2) ---- COMPUTE SIGNALS / DESIRED HOLDINGS FOR TOMORROW (h_next) ----
        # (a) Trend weekly picks decided on rebalance dates (Fridays)
        if dt in reb_dates:
            m = mom.loc[dt, candidates]
            ranked = m.dropna().sort_values(ascending=False)
            if ranked.empty:
                top1, top2 = None, None
                current_trend_tradecols = []
            else:
                top = list(ranked.index[:top_n])
                top1 = top[0] if len(top) > 0 else None
                top2 = top[1] if len(top) > 1 else (ranked.index[1] if len(ranked) > 1 else None)
                current_trend_tradecols = [_trend_universe_to_trade_col(t) for t in top if t is not None]

            wk = _week_end_index(pd.DatetimeIndex([dt]))[0]
            row = {
                "week_end": str(wk.date()),
                "rank1": top1,
                "rank2": top2,
                "rank1_trade": _trend_universe_to_trade_col(top1) if top1 else None,
                "rank2_trade": _trend_universe_to_trade_col(top2) if top2 else None,
                "score1_mom": float(ranked.iloc[0]) if len(ranked) > 0 else np.nan,
                "score2_mom": float(ranked.iloc[1]) if len(ranked) > 1 else np.nan,
            }
            row["rank1_week_ret"] = get_week_ret(row["rank1_trade"], wk) if row["rank1_trade"] else np.nan
            row["rank2_week_ret"] = get_week_ret(row["rank2_trade"], wk) if row["rank2_trade"] else np.nan
            picks_rows.append(row)

        # (b) MeanRev entry/exit decisions at dt (already uses shift(1) for signal)
        mr_price_under = prices[mr_base] if mr_base in prices.columns else None
        if mr_price_under is None or mr_price_under.isna().all():
            mr_active = False
            mr_trade_col = None
            mr_entry_price = None
            mr_days = 0
        else:
            r_lb = mr_price_under.pct_change(mr_lb).shift(1).loc[dt]

            if (not mr_active) and pd.notna(r_lb) and (float(r_lb) <= mr_drop):
                mr_active = True
                mr_trade_col = _meanrev_universe_to_trade_col(mr_base)
                mr_entry_price = prices[mr_trade_col].loc[dt] if mr_trade_col in prices.columns else None
                mr_days = 0

            if mr_active:
                mr_days += 1
                cur_px = prices[mr_trade_col].loc[dt] if (mr_trade_col in prices.columns) else np.nan
                if mr_entry_price is None or pd.isna(mr_entry_price) or pd.isna(cur_px):
                    mr_active = False
                    mr_trade_col = None
                    mr_entry_price = None
                    mr_days = 0
                else:
                    pnl = float(cur_px / mr_entry_price - 1.0)
                    if (mr_days >= mr_hold) or (pnl >= mr_tp) or (pnl <= mr_sl):
                        mr_active = False
                        mr_trade_col = None
                        mr_entry_price = None
                        mr_days = 0

        # (c) Build desired holdings for TOMORROW based on today's state + decisions
        st_key = st.lower()
        w_tr = float(alloc[st_key]["trend"])
        w_mr = float(alloc[st_key]["meanrev"])
        w_df = float(alloc[st_key]["defensive"])
        df_w = _risk_off_weights(risk_off_mode)

        h_des = {}

        # Trend slice (equal weight)
        if w_tr > 0 and len(current_trend_tradecols) > 0:
            per = w_tr / len(current_trend_tradecols)
            for tcol in current_trend_tradecols:
                h_des[tcol] = h_des.get(tcol, 0.0) + per

        # MeanRev slice (inactive -> SHY)
        if w_mr > 0:
            if mr_active and (mr_trade_col in returns.columns):
                h_des[mr_trade_col] = h_des.get(mr_trade_col, 0.0) + w_mr
            else:
                h_des["SHY"] = h_des.get("SHY", 0.0) + w_mr

        # Defensive slice
        if w_df > 0:
            for t, w in df_w.items():
                h_des[t] = h_des.get(t, 0.0) + (w_df * float(w))

        h_des = _normalize_weights(h_des)
        if not h_des:
            h_des = {"SHY": 1.0}

        # 3) ---- COMMIT TOMORROW HOLDINGS (lookahead-free) ----
        # only if there *is* a next trading day
        if i < len(idx) - 1:
            h_next = h_des
            h_cur = h_next

        # log engine choice for dt (decision date)
        engine_choice_log.append({
            "date": str(dt.date()),
            "state": st,
            "w_trend": w_tr,
            "w_meanrev": w_mr,
            "w_defensive": w_df,
            "meanrev_active": bool(mr_active),
            "meanrev_ticker": mr_trade_col if mr_active else "",
            "rebalance_today": bool(dt in reb_dates),
        })

    equity_series = pd.Series(curve, index=prices.index, name="equity")
    picks_df = pd.DataFrame(picks_rows)
    holdings_daily = pd.DataFrame(holdings_daily_rows)

    # holdings_weekly (없는 날=0 포함 평균)
    if not holdings_daily.empty:
        hd = holdings_daily.copy()
        hd["date"] = pd.to_datetime(hd["date"])
        mat = (
            hd.pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum")
            .fillna(0.0)
        )
        mat["week_end"] = _week_end_index(mat.index)
        wk_mean = mat.groupby("week_end").mean().drop(columns=["week_end"], errors="ignore")
        wk_avg = wk_mean.reset_index().melt(id_vars=["week_end"], var_name="ticker", value_name="avg_weight")
        wk_avg = wk_avg[wk_avg["avg_weight"] > 0]
    else:
        wk_avg = pd.DataFrame(columns=["week_end", "ticker", "avg_weight"])

    def wk_ret_col(ticker: str, wk_end_ts: pd.Timestamp) -> float:
        if ticker not in wclose.columns or wk_end_ts not in wclose.index:
            return np.nan
        loc = wclose.index.get_loc(wk_end_ts)
        if loc == 0:
            return np.nan
        prev = wclose.iloc[loc - 1][ticker]
        cur = wclose.iloc[loc][ticker]
        if pd.isna(prev) or prev == 0 or pd.isna(cur):
            return np.nan
        return float(cur / prev - 1.0)

    if not wk_avg.empty:
        wk_avg["week_ret"] = wk_avg.apply(lambda r: wk_ret_col(r["ticker"], pd.Timestamp(r["week_end"])), axis=1)
        wk_avg["contrib"] = wk_avg["avg_weight"] * wk_avg["week_ret"]
        wk_avg["week_end"] = wk_avg["week_end"].astype(str)

    holdings_weekly = wk_avg
    return equity_series, engine_choice_log, picks_df, holdings_daily, holdings_weekly