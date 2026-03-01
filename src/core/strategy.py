import pandas as pd
import numpy as np
from .regime import compute_regime
from .risk_off import risk_off_weights
from .leverage_map import map_to_leveraged


def momentum_scores(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    return prices.pct_change(int(lookback_days))


def _weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=1)
    weekly = s.resample("W-FRI").last().dropna().index
    weekly = weekly[weekly.isin(index)]
    return weekly


def run_backtest(prices: pd.DataFrame, cfg, save_picks_path: str | None = None) -> pd.Series:
    returns = prices.pct_change().fillna(0.0)

    lookback = int(cfg["selection"]["lookback_days"])
    top_n = int(cfg["selection"]["top_n"])

    # Look-ahead safe signals
    scores = momentum_scores(prices, lookback).shift(1)

    # Market proxy for regime (prefer QQQ if exists)
    mkt = prices["QQQ"] if "QQQ" in prices.columns else prices.iloc[:, 0]
    regime = compute_regime(mkt, cfg).shift(1)

    reb_dates = (
        _weekly_dates(prices.index)
        if cfg.get("portfolio", {}).get("rebalance", "weekly") == "weekly"
        else prices.index
    )

    equity = 1.0
    curve = []

    # These are "traded" tickers (may be 3x mapped)
    current_assets = None
    current_weights = None

    picks_rows = []

    for date in prices.index:
        if date in reb_dates:
            state = regime.loc[date] if date in regime.index else np.nan

            base_assets = []
            traded_assets = []
            traded_weights = []

            if pd.isna(state):
                current_assets = None
                current_weights = None
            else:
                # WEAK => risk-off allocation (default stays 1x by config)
                if cfg.get("risk_off", {}).get("enabled", True) and state == "WEAK":
                    w = risk_off_weights(cfg["risk_off"]["mode"])
                    base_assets = list(w.keys())
                    traded_assets = [map_to_leveraged(t, cfg, state) for t in base_assets]
                    traded_weights = [float(w[k]) for k in base_assets]
                else:
                    row = scores.loc[date]
                    if row.isna().all():
                        current_assets = None
                        current_weights = None
                    else:
                        base_assets = row.nlargest(top_n).index.tolist()
                        traded_assets = [map_to_leveraged(t, cfg, state) for t in base_assets]
                        traded_weights = [1.0 / len(traded_assets)] * len(traded_assets)

            current_assets = traded_assets if traded_assets else None
            current_weights = traded_weights if traded_assets else None

            if save_picks_path is not None:
                b1 = base_assets[0] if len(base_assets) > 0 else ""
                b2 = base_assets[1] if len(base_assets) > 1 else ""
                t1 = traded_assets[0] if len(traded_assets) > 0 else ""
                t2 = traded_assets[1] if len(traded_assets) > 1 else ""

                s1 = float(scores.loc[date, b1]) if b1 and b1 in scores.columns else np.nan
                s2 = float(scores.loc[date, b2]) if b2 and b2 in scores.columns else np.nan

                picks_rows.append({
                    "date": date,
                    "state": str(state),
                    "base_top1": b1,
                    "base_top1_score": s1,
                    "base_top2": b2,
                    "base_top2_score": s2,
                    "traded_top1": t1,
                    "traded_top2": t2,
                })

        # Daily return based on *traded* tickers
        daily_ret = 0.0
        if current_assets:
            for a, w in zip(current_assets, current_weights):
                if a in returns.columns:
                    daily_ret += float(returns.loc[date, a]) * float(w)

        equity *= (1.0 + daily_ret)
        curve.append(equity)

    curve_s = pd.Series(curve, index=prices.index)

    # Attach realized return from rebalance date -> next rebalance date
    if save_picks_path is not None and picks_rows:
        picks_df = pd.DataFrame(picks_rows).set_index("date")
        reb_idx = picks_df.index.sort_values()
        next_dates = list(reb_idx[1:]) + [pd.NaT]

        realized = []
        for d, nd in zip(reb_idx, next_dates):
            if pd.isna(nd):
                realized.append(np.nan)
            else:
                realized.append(float(curve_s.loc[nd] / curve_s.loc[d] - 1.0))

        picks_df["next_rebalance_date"] = next_dates
        picks_df["realized_return_to_next_rebalance"] = realized
        picks_df.reset_index().to_csv(save_picks_path, index=False)

    return curve_s