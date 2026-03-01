
import pandas as pd
import numpy as np
from .regime import compute_regime
from .risk_off import risk_off_weights

def momentum_scores(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    # look-ahead safe if shifted by 1 later
    mom = prices.pct_change(int(lookback_days))
    return mom

def _weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # W-FRI aligned to trading days: take last trading day each week
    s = pd.Series(index=index, data=1)
    weekly = s.resample("W-FRI").last().dropna().index
    # keep only those that are in original index (should be, but safe)
    weekly = weekly[weekly.isin(index)]
    return weekly

def run_backtest(prices: pd.DataFrame, cfg, save_picks_path: str | None = None) -> pd.Series:
    returns = prices.pct_change().fillna(0.0)

    lookback = int(cfg["selection"]["lookback_days"])
    top_n = int(cfg["selection"]["top_n"])

    # signals computed on day t, applied on day t+1 (shift(1))
    scores = momentum_scores(prices, lookback).shift(1)

    # regime on market proxy: first ticker column (e.g., QQQ)
    regime = compute_regime(prices.iloc[:, 0], cfg).shift(1)

    # weekly rebalance schedule
    if cfg.get("portfolio", {}).get("rebalance", "weekly") == "weekly":
        reb_dates = _weekly_dates(prices.index)
    else:
        reb_dates = prices.index  # fallback daily

    equity = 1.0
    curve = []
    current_assets = None
    current_weights = None
    current_lev = 1.0

    picks_rows = []

    for i, date in enumerate(prices.index):
        # set holdings on rebalance dates (decision uses signals at 'date', already shifted => uses date-1)
        if date in reb_dates:
            state = regime.loc[date] if date in regime.index else np.nan
            if pd.isna(state):
                current_assets = None
                current_weights = None
                current_lev = 1.0
            elif cfg.get("risk_off", {}).get("enabled", True) and state == "WEAK":
                # WEAK treated as risk-off in this simple v3 template (can be refined later)
                w = risk_off_weights(cfg["risk_off"]["mode"])
                current_assets = list(w.keys())
                current_weights = [float(w[k]) for k in current_assets]
                current_lev = 1.0
            else:
                row = scores.loc[date]
                if row.isna().all():
                    current_assets = None
                    current_weights = None
                    current_lev = 1.0
                else:
                    top = row.nlargest(top_n).index.tolist()
                    current_assets = top
                    current_weights = [1.0 / len(top)] * len(top)
                    if cfg.get("leverage", {}).get("enabled", True):
                        current_lev = float(cfg["leverage"]["strong"]) if state == "STRONG" else float(cfg["leverage"]["weak"])
                    else:
                        current_lev = 1.0

            # record pick snapshot + next-week realized return (computed later)
            if save_picks_path is not None:
                r1 = current_assets[0] if current_assets and len(current_assets) > 0 else ""
                r2 = current_assets[1] if current_assets and len(current_assets) > 1 else ""
                s1 = float(scores.loc[date, r1]) if r1 and r1 in scores.columns else np.nan
                s2 = float(scores.loc[date, r2]) if r2 and r2 in scores.columns else np.nan
                picks_rows.append({
                    "date": date,
                    "state": str(regime.loc[date]) if date in regime.index else "",
                    "top1": r1,
                    "top1_score": s1,
                    "top2": r2,
                    "top2_score": s2,
                    "leverage": current_lev
                })

        # compute daily return based on current holdings (applies from next_close notion approximately via shifted signals)
        daily_ret = 0.0
        if current_assets:
            for a, w in zip(current_assets, current_weights):
                if a in returns.columns:
                    daily_ret += float(returns.loc[date, a]) * float(w)
            daily_ret *= current_lev

        equity *= (1.0 + daily_ret)
        curve.append(equity)

    curve_s = pd.Series(curve, index=prices.index)

    # attach realized next-week return to picks (week-to-week based on equity curve)
    if save_picks_path is not None and picks_rows:
        picks_df = pd.DataFrame(picks_rows).set_index("date")
        # next period return from rebalance date to next rebalance date (using equity curve)
        reb_idx = picks_df.index.sort_values()
        next_dates = list(reb_idx[1:]) + [pd.NaT]
        realized = []
        for d, nd in zip(reb_idx, next_dates):
            if pd.isna(nd):
                realized.append(np.nan)
                continue
            realized.append(float(curve_s.loc[nd] / curve_s.loc[d] - 1.0))
        picks_df["next_rebalance_date"] = next_dates
        picks_df["realized_return_to_next_rebalance"] = realized
        picks_df.reset_index().to_csv(save_picks_path, index=False)

    return curve_s
