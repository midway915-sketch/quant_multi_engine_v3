from __future__ import annotations
import pandas as pd
import numpy as np

from .state import compute_state_flags
from .engines import run_trend_engine, run_meanrev_engine, run_defensive_engine


def run_meta_portfolio(prices: pd.DataFrame, cfg):
    """
    Returns:
      equity_curve: pd.Series
      engine_choice_log: pd.DataFrame
      picks_top2_weekly: pd.DataFrame (trend picks + realized return to next rebalance)
    """
    state_df = compute_state_flags(prices, cfg)

    eq_trend, picks = run_trend_engine(prices, cfg, state_df)
    eq_mr = run_meanrev_engine(prices, cfg, state_df)
    eq_def = run_defensive_engine(prices, cfg, state_df)

    # convert engine equity -> daily returns
    r_tr = eq_trend.pct_change().fillna(0.0)
    r_mr = eq_mr.pct_change().fillna(0.0)
    r_df = eq_def.pct_change().fillna(0.0)

    equity = 1.0
    curve = []
    rows = []

    for d in prices.index:
        st = state_df.loc[d, "state"]

        w = cfg["allocator"][st.lower()]
        wt = float(w["trend"]); wm = float(w["meanrev"]); wd = float(w["defensive"])
        # weights should sum to 1 already

        daily = wt * float(r_tr.loc[d]) + wm * float(r_mr.loc[d]) + wd * float(r_df.loc[d])

        equity *= (1.0 + daily)
        curve.append(equity)

        rows.append({
            "date": d,
            "state": st,
            "w_trend": wt,
            "w_meanrev": wm,
            "w_defensive": wd,
        })

    eq = pd.Series(curve, index=prices.index)
    choice_log = pd.DataFrame(rows).set_index("date")

    # add realized return between rebalance dates to picks
    picks_out = _attach_realized(eq, picks)

    return eq, choice_log.reset_index(), picks_out


def _attach_realized(eq: pd.Series, picks: pd.DataFrame) -> pd.DataFrame:
    if picks is None or picks.empty:
        return pd.DataFrame()

    picks = picks.copy()
    reb_idx = picks.index.sort_values()
    next_dates = list(reb_idx[1:]) + [pd.NaT]

    realized = []
    for d, nd in zip(reb_idx, next_dates):
        if pd.isna(nd):
            realized.append(np.nan)
        else:
            realized.append(float(eq.loc[nd] / eq.loc[d] - 1.0))

    picks["next_rebalance_date"] = next_dates
    picks["realized_return_to_next_rebalance"] = realized
    return picks.reset_index()