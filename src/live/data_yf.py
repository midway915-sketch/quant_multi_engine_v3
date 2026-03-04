from __future__ import annotations
import time
import datetime as dt
from typing import List
import pandas as pd

def download_daily_closes(tickers: List[str], start: str, max_retries: int = 5) -> pd.DataFrame:
    import yfinance as yf

    # yfinance는 end가 포함되지 않는 경우가 많아서 여유를 둔다
    end = (dt.date.today() + dt.timedelta(days=2)).strftime("%Y-%m-%d")

    last_err = None
    for k in range(max_retries):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="column",
            )
            if df is None or len(df) == 0:
                raise RuntimeError("Downloaded data empty")

            # Adj Close 우선
            if isinstance(df.columns, pd.MultiIndex):
                if "Adj Close" in df.columns.levels[0]:
                    px = df["Adj Close"].copy()
                elif "Close" in df.columns.levels[0]:
                    px = df["Close"].copy()
                else:
                    px = df.xs(df.columns.levels[0][0], level=0, axis=1).copy()
            else:
                # 단일 티커 케이스 (거의 안 씀)
                if "Adj Close" in df.columns:
                    px = df[["Adj Close"]].copy()
                    px.columns = [tickers[0]]
                elif "Close" in df.columns:
                    px = df[["Close"]].copy()
                    px.columns = [tickers[0]]
                else:
                    raise RuntimeError("Neither Adj Close nor Close found")

            px = px.sort_index().ffill().dropna(how="all")
            # 컬럼 정리
            px = px[[c for c in tickers if c in px.columns]]
            if px.empty:
                raise RuntimeError("Prices empty after column filter")
            return px

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** k, 20))

    raise RuntimeError(f"yfinance download failed: {last_err}")