from __future__ import annotations
import pandas as pd
import numpy as np

from src.core.state import compute_state_flags


def _risk_off_weights(mode: str) -> dict:
    if mode == "SHY_100":
        return {"SHY": 1.0}
    if mode == "SHY_GLD_50_50":
        return {"SHY": 0.5, "GLD": 0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY": 0.7, "GLD": 0.3}
    if mode == "GLD_100":
        return {"GLD": 1.0}
    return {"SHY": 1.0}


def _trend_universe_to_trade_col(ticker: str) -> str:
    # Trend는 3x (실물 있으면 실물, 없으면 proxy) -> *_MIX 사용
    if ticker == "QQQ":
        return "TQQQ_MIX"
    if ticker == "SPY":
        return "UPRO_MIX"
    if ticker == "SOXX":
        return "SOXL_MIX"
    return ticker


def _meanrev_universe_to_trade_col(ticker: str) -> str:
    # MeanRev는 2x (실물 있으면 실물, 없으면 proxy) -> *_MIX 사용
    if ticker == "QQQ":
        return "QLD_MIX"
    if ticker == "SPY":
        return "SSO_MIX"
    if ticker == "SOXX":
        return "USD_MIX"
    return ticker


def _week_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # 주간은 금요일(week_end) 기준
    return idx.to_period("W-FRI").to_timestamp("W-FRI")


def _weekly_close(prices: pd.DataFrame) -> pd.DataFrame:
    # 금요일 종가로 주간 close 생성
    return prices.resample("W-FRI").last()


def run_meta_portfolio(prices: pd.DataFrame, cfg: dict):
    """
    Returns:
      equity: pd.Series
      engine_choice_log: list[dict]
      picks_top2_weekly: pd.DataFrame
      holdings_daily: pd.DataFrame  (date,ticker,weight)
      holdings_weekly: pd.DataFrame (week_end,ticker,avg_weight,week_ret,contrib)
    """
    prices = prices.copy()
    returns = prices.pct_change().fillna(0.0)

    state_df = compute_state_flags(prices, cfg)

    # Trend ranking (6m momentum like, configurable days)
    mom_lb = int(cfg["trend_engine"]["mom_lookback_days"])
    mom = prices.pct_change(mom_lb).fillna(0.0)

    # Trend candidates are cfg["trend_engine"]["candidates"] (or fallback)
    candidates = cfg.get("trend_engine", {}).get("candidates", ["QQQ", "SPY", "SOXX"])
    top_n = int(cfg["selection"]["top_n"])

    # MeanRev config
    mr_lb = int(cfg["meanrev_engine"]["lookback_days"])
    mr_drop = float(cfg["meanrev_engine"]["drop_threshold"])
    mr_hold = int(cfg["meanrev_engine"]["hold_days"])
    mr_tp = float(cfg["meanrev_engine"]["take_profit"])
    mr_sl = float(cfg["meanrev_engine"]["stop_loss"])

    # MeanRev underlying universe (same set)
    mr_candidates = cfg.get("meanrev_engine", {}).get("candidates", ["QQQ", "SPY", "SOXX"])

    # Allocator
    alloc = cfg["allocator"]
    risk_off_mode = cfg.get("risk_off", {}).get("mode", "SHY_100")

    # Weekly rebalance points
    week_end = _week_end_index(prices.index)
    is_reb = week_end != week_end.shift(1)  # first day of each weekly bucket
    # We want decisions at week_end itself, so mark the actual Fridays that exist
    reb_dates = prices.index[prices.index.isin(week_end.unique())]

    # Track trend picks (rebalanced weekly)
    current_trend = []
    picks_rows = []

    # MeanRev position tracking
    mr_active = False
    mr_entry_date = None
    mr_entry_price = None
    mr_days = 0
    mr_ticker = None  # underlying ticker (QQQ/SPY/SOXX)
    mr_trade_col = None

    equity = 1.0
    curve = []
    engine_choice_log = []

    # Holdings logs
    holdings_daily_rows = []

    # Precompute weekly closes for reporting weekly returns of picked tickers
    wclose = _weekly_close(prices)

    def get_week_ret(ticker_col: str, wk_end: pd.Timestamp) -> float:
        # 주간 수익률: (이번 주 금요일 close / 전 주 금요일 close) - 1
        if ticker_col not in wclose.columns:
            return np.nan
        if wk_end not in wclose.index:
            return np.nan
        loc = wclose.index.get_loc(wk_end)
        if loc == 0:
            return np.nan
        prev = wclose.iloc[loc - 1][ticker_col]
        cur = wclose.iloc[loc][ticker_col]
        if pd.isna(prev) or prev == 0 or pd.isna(cur):
            return np.nan
        return float(cur / prev - 1.0)

    for dt in prices.index:
        st = state_df.loc[dt, "state"]

        # -------- weekly trend selection on rebalance dates (Fridays) --------
        if dt in reb_dates:
            # rank by momentum among candidates (use underlying 1x universe to rank)
            m = mom.loc[dt, candidates]
            # larger is better
            ranked = m.sort_values(ascending=False)
            top = list(ranked.index[:top_n])
            # store both top1/top2 for logs (even if top_n=1)
            top1 = top[0] if len(top) > 0 else None
            top2 = top[1] if len(top) > 1 else (ranked.index[1] if len(ranked) > 1 else None)

            # trade columns (3x mix)
            trade_top = [_trend_universe_to_trade_col(t) for t in top if t is not None]
            current_trend = trade_top

            # weekly reporting row
            wk = _week_end_index(pd.DatetimeIndex([dt]))[0]
            row = {
                "week_end": str(wk.date()),
                "rank1": top1,
                "rank2": top2,
                "rank1_trade": _trend_universe_to_trade_col(top1) if top1 else None,
                "rank2_trade": _trend_universe_to_trade_col(top2) if top2 else None,
                "score1_6m": float(ranked.iloc[0]) if len(ranked) > 0 else np.nan,
                "score2_6m": float(ranked.iloc[1]) if len(ranked) > 1 else np.nan,
            }
            # 종목별 "주간 수익률"(해당 trade_col의 금요일 close 기준)
            if row["rank1_trade"]:
                row["rank1_week_ret"] = get_week_ret(row["rank1_trade"], wk)
            else:
                row["rank1_week_ret"] = np.nan
            if row["rank2_trade"]:
                row["rank2_week_ret"] = get_week_ret(row["rank2_trade"], wk)
            else:
                row["rank2_week_ret"] = np.nan

            picks_rows.append(row)

        # -------- MeanReversion signal & position update (2x mix) --------
        # signal uses underlying (1x) prices of a chosen base candidate:
        # choose the same as top1 underlying if available, else QQQ
        mr_base = None
        if len(current_trend) > 0:
            # current_trend is 3x trade cols; map back to underlying for meanrev choice
            # simplest: align meanrev to QQQ by default unless configured
            pass
        mr_base = cfg.get("meanrev_engine", {}).get("base", "QQQ")
        if mr_base not in mr_candidates:
            mr_base = "QQQ"

        # meanrev underlying close
        mr_price = prices[mr_base] if mr_base in prices.columns else None

        if mr_price is None or mr_price.isna().all():
            # can't run meanrev; force inactive
            mr_active = False
            mr_ticker = None
            mr_trade_col = None
            mr_entry_date = None
            mr_entry_price = None
            mr_days = 0
        else:
            # entry condition: 20d return <= drop threshold (lookahead-safe by using shift(1))
            r_lb = mr_price.pct_change(mr_lb).shift(1).loc[dt]
            if (not mr_active) and pd.notna(r_lb) and (float(r_lb) <= mr_drop):
                mr_active = True
                mr_ticker = mr_base
                mr_trade_col = _meanrev_universe_to_trade_col(mr_base)
                mr_entry_date = dt
                mr_entry_price = prices[mr_trade_col].loc[dt] if mr_trade_col in prices.columns else None
                mr_days = 0

            if mr_active:
                mr_days += 1
                cur_px = prices[mr_trade_col].loc[dt] if (mr_trade_col in prices.columns) else np.nan
                if mr_entry_price is None or pd.isna(mr_entry_price) or pd.isna(cur_px):
                    # bad data -> exit
                    mr_active = False
                    mr_ticker = None
                    mr_trade_col = None
                    mr_entry_date = None
                    mr_entry_price = None
                    mr_days = 0
                else:
                    pnl = float(cur_px / mr_entry_price - 1.0)
                    # exit rules: hold_days OR TP OR SL
                    if (mr_days >= mr_hold) or (pnl >= mr_tp) or (pnl <= mr_sl):
                        mr_active = False
                        mr_ticker = None
                        mr_trade_col = None
                        mr_entry_date = None
                        mr_entry_price = None
                        mr_days = 0

        # -------- Portfolio weights by state --------
        w_tr = float(alloc[st.lower()]["trend"])
        w_mr = float(alloc[st.lower()]["meanrev"])
        w_df = float(alloc[st.lower()]["defensive"])

        # Defensive holdings
        df_w = _risk_off_weights(risk_off_mode)

        # Build today's holdings (ticker -> weight)
        h = {}

        # Trend holdings: equally among current_trend
        if w_tr > 0 and len(current_trend) > 0:
            per = w_tr / len(current_trend)
            for tcol in current_trend:
                h[tcol] = h.get(tcol, 0.0) + per

        # MeanRev holdings: if active, 100% of meanrev slice into 2x mix, else park in SHY
        if w_mr > 0:
            if mr_active and mr_trade_col in returns.columns:
                h[mr_trade_col] = h.get(mr_trade_col, 0.0) + w_mr
            else:
                # 실전 구현 쉬운 “대기자산” 처리
                h["SHY"] = h.get("SHY", 0.0) + w_mr

        # Defensive holdings
        if w_df > 0:
            for t, w in df_w.items():
                h[t] = h.get(t, 0.0) + (w_df * float(w))

        # Normalize tiny numeric drift
        s = sum(h.values())
        if s > 0:
            for k in list(h.keys()):
                h[k] = float(h[k] / s)

        # log holdings daily
        for t, w in h.items():
            holdings_daily_rows.append({"date": str(dt.date()), "ticker": t, "weight": w, "state": st})

        # compute daily portfolio return (simple weighted sum)
        daily_ret = 0.0
        for t, w in h.items():
            if t in returns.columns:
                daily_ret += float(returns.loc[dt, t]) * float(w)

        equity *= (1.0 + daily_ret)
        curve.append(equity)

        engine_choice_log.append({
            "date": str(dt.date()),
            "state": st,
            "w_trend": w_tr,
            "w_meanrev": w_mr,
            "w_defensive": w_df,
            "meanrev_active": bool(mr_active),
            "meanrev_ticker": mr_trade_col if mr_active else "",
        })

    equity_series = pd.Series(curve, index=prices.index, name="equity")

    picks_df = pd.DataFrame(picks_rows)

    holdings_daily = pd.DataFrame(holdings_daily_rows)

    # Weekly holdings: average weight per week_end
    if not holdings_daily.empty:
        holdings_daily["date"] = pd.to_datetime(holdings_daily["date"])
        holdings_daily["week_end"] = _week_end_index(pd.DatetimeIndex(holdings_daily["date"])).values

        wk_avg = (
            holdings_daily
            .groupby(["week_end", "ticker"], as_index=False)["weight"]
            .mean()
            .rename(columns={"weight": "avg_weight"})
        )
    else:
        wk_avg = pd.DataFrame(columns=["week_end", "ticker", "avg_weight"])

    # Add weekly returns and contribution
    if not wk_avg.empty:
        wk_avg["week_ret"] = wk_avg.apply(lambda r: get_week_ret(r["ticker"], pd.Timestamp(r["week_end"])), axis=1)
        wk_avg["contrib"] = wk_avg["avg_weight"] * wk_avg["week_ret"]
        wk_avg["week_end"] = wk_avg["week_end"].astype(str)
    holdings_weekly = wk_avg

    return equity_series, engine_choice_log, picks_df, holdings_daily, holdings_weekly