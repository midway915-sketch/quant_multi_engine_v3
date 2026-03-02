from __future__ import annotations
import pandas as pd
import numpy as np

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
    # MeanRev은 1.5x (QQQ/SPY/SOXX 대응)
    if ticker == "QQQ":
        return "QLD_MIX"
    if ticker == "SPY":
        return "SSO_MIX"
    if ticker == "SOXX":
        return "USD_MIX"
    return ticker


def _week_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # pick Fridays that exist in index
    df = pd.DataFrame(index=idx)
    df["w"] = df.index.to_period("W-FRI").end_time.normalize()
    # map each date to its week_end, then take unique week_end that exist in idx
    wk = df["w"].unique()
    wk = pd.DatetimeIndex(wk).sort_values()
    wk = wk[wk.isin(idx)]
    return wk


def _safe_get_return(returns: pd.DataFrame, dt, ticker_col: str) -> float:
    if ticker_col not in returns.columns:
        return 0.0
    r = returns.at[dt, ticker_col]
    if pd.isna(r):
        return 0.0
    return float(r)


def _calc_week_ret(px: pd.DataFrame, week_end: pd.Timestamp, ticker_col: str) -> float:
    if ticker_col not in px.columns:
        return np.nan
    loc = px.index.get_loc(week_end)
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
      holdings_daily: pd.DataFrame  (date,ticker,weight,state)
      holdings_weekly: pd.DataFrame (week_end,ticker,avg_weight,week_ret,contrib)
    """
    prices = prices.copy()
    returns = prices.pct_change().fillna(0.0)

    state_df = compute_state_flags_adaptive(prices, cfg)

    # ---- Trend config (safe fallback) ----
    trend_cfg = cfg.get("trend_engine", {}) or {}
    mom_lb = int(trend_cfg.get("mom_lookback_days", 168))
    candidates = trend_cfg.get("candidates", ["QQQ", "SPY", "SOXX"])

    sel_cfg = cfg.get("selection", {}) or {}
    top_n = int(sel_cfg.get("top_n", trend_cfg.get("top_n", 1)))

    # Momentum
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

    # Weekly rebalance dates (Fridays present in price index)
    week_end = _week_end_index(prices.index)
    reb_dates = prices.index[prices.index.isin(week_end)]

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
    holdings_rows = []

    buy_cost = float((cfg.get("costs", {}) or {}).get("buy", 0.0))
    sell_cost = float((cfg.get("costs", {}) or {}).get("sell", 0.0))

    for dt in prices.index:
        # ---- apply today's return based on yesterday-confirmed holdings (h_cur) ----
        day_ret = 0.0
        for tcol, w in h_cur.items():
            r = _safe_get_return(returns, dt, tcol)
            day_ret += w * r

        # apply costs when holdings changed yesterday (h_cur came from h_next of prev day)
        # meta.py original: cost is applied when we set h_next; here we keep the same behavior:
        # we apply cost on transition at the moment of change (when setting h_next below).
        equity *= (1.0 + day_ret)
        curve.append({"date": dt, "equity": equity})

        st = str(state_df.at[dt, "state"])
        vol_reg = str(state_df.at[dt, "vol_regime"]) if "vol_regime" in state_df.columns else "NORMAL"

        # ---- compute signals / desired holdings for TOMORROW (h_next) ----
        # (a) Trend weekly picks decided on rebalance dates (Fridays)
        if dt in reb_dates:
            m = mom.loc[dt, candidates]
            ranked = m.dropna().sort_values(ascending=False)
            if ranked.empty:
                top1, top2 = None, None
                current_trend_tradecols = []
            else:
                top = list(ranked.index[:top_n])
                # Adaptive: SOXX conditional allowance (and other future filters)
                top = filter_trend_picks(top, m, cfg, vol_reg)
                top1 = top[0] if len(top) > 0 else None
                top2 = top[1] if len(top) > 1 else (ranked.index[1] if len(ranked) > 1 else None)
                current_trend_tradecols = [_trend_universe_to_trade_col(t) for t in top if t is not None]

            wk = _week_end_index(pd.DatetimeIndex([dt]))[0]
            row = {
                "week_end": str(wk.date()),
                "top1": top1,
                "top2": top2,
                "mom_lb": mom_lb,
                "vol_regime": vol_reg,
            }
            picks_rows.append(row)

        # (b) MeanRev signal (daily): if base drops >= threshold over lookback, enter MR basket next day
        mr_signal = False
        mr_exit = False

        if mr_base in prices.columns:
            base_px = prices[mr_base]
            if dt in base_px.index:
                loc = base_px.index.get_loc(dt)
                if isinstance(loc, slice):
                    loc = loc.stop - 1
                if loc >= mr_lb:
                    prev = base_px.iloc[loc - mr_lb]
                    cur = base_px.iloc[loc]
                    if pd.notna(prev) and prev != 0 and pd.notna(cur):
                        drop = float(cur / prev - 1.0)
                        mr_signal = (drop <= mr_drop)

        # manage MR position
        if mr_active:
            mr_days += 1
            # check TP/SL using today's close vs entry
            if mr_trade_col and mr_trade_col in prices.columns:
                curp = float(prices.at[dt, mr_trade_col])
                if mr_entry_price and mr_entry_price != 0:
                    pnl = curp / mr_entry_price - 1.0
                    if pnl >= mr_tp:
                        mr_exit = True
                    if pnl <= mr_sl:
                        mr_exit = True
            # time exit
            if mr_days >= mr_hold:
                mr_exit = True

        # state today -> weights
        st_key = st.lower()

        # Adaptive bull weights
        if st == "BULL" and bool((cfg.get("adaptive", {}) or {}).get("enabled", True)) and bool((cfg.get("adaptive", {}) or {}).get("override_bull_allocator", True)):
            plan = choose_bull_weights(vol_reg, cfg)
            w_tr, w_mr, w_df = float(plan.trend), float(plan.meanrev), float(plan.defensive)
        else:
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

        # MeanRev slice
        if w_mr > 0:
            if (not mr_active) and mr_signal:
                # enter MR basket tomorrow
                best = None
                if dt in reb_dates:
                    # use same ranked list but MR candidates
                    mm = mom.loc[dt, mr_candidates]
                    rr = mm.dropna().sort_values(ascending=False)
                    best = rr.index[0] if len(rr) > 0 else None
                if best is None:
                    best = "QQQ"
                mr_trade_col = _meanrev_universe_to_trade_col(str(best))
                mr_active = True
                mr_days = 0
                # entry price is tomorrow close; approximate with today close for decision bookkeeping
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

        # Costs on changes (applied when setting h_next)
        # simple turnover cost approximation
        turnover = 0.0
        for k in set(h_cur.keys()) | set(h_des.keys()):
            turnover += abs(h_des.get(k, 0.0) - h_cur.get(k, 0.0))
        # split buy/sell half-half
        cost = (turnover * 0.5) * (buy_cost + sell_cost)

        # Desired holdings become tomorrow's confirmed holdings
        h_next = h_des.copy()

        # log engine choice
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

        # record today's holdings
        for tcol, w in h_cur.items():
            holdings_rows.append({"date": str(dt.date()), "ticker": tcol, "weight": float(w), "state": st, "vol_regime": vol_reg})

        # apply cost on equity at end of day (cost for the rebalance decided today)
        equity *= (1.0 - cost)

        # advance holdings: tomorrow holdings become today's in next loop iteration
        h_cur = h_next

    equity_series = pd.Series([x["equity"] for x in curve], index=[x["date"] for x in curve])
    equity_series.index = pd.to_datetime(equity_series.index)

    picks_top2_weekly = pd.DataFrame(picks_rows)

    holdings_daily = pd.DataFrame(holdings_rows)

    # weekly holdings aggregation (avg weight per week_end)
    if holdings_daily.empty:
        holdings_weekly = pd.DataFrame(columns=["week_end", "ticker", "avg_weight", "week_ret", "contrib"])
    else:
        hd = holdings_daily.copy()
        hd["date"] = pd.to_datetime(hd["date"])
        hd["week_end"] = hd["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
        grp = hd.groupby(["week_end", "ticker"], as_index=False)["weight"].mean().rename(columns={"weight": "avg_weight"})
        # compute week_ret and contribution
        wk_rows = []
        for _, r in grp.iterrows():
            we = pd.Timestamp(r["week_end"])
            t = str(r["ticker"])
            w = float(r["avg_weight"])
            wret = _calc_week_ret(prices, we, t)
            contrib = w * (0.0 if pd.isna(wret) else float(wret))
            wk_rows.append({"week_end": str(we.date()), "ticker": t, "avg_weight": w, "week_ret": wret, "contrib": contrib})
        holdings_weekly = pd.DataFrame(wk_rows)

    return equity_series, engine_choice_log, picks_top2_weekly, holdings_daily, holdings_weekly