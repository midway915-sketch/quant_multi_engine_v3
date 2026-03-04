from __future__ import annotations
import datetime as dt
from typing import Optional

def _try_get_nyse_calendar():
    try:
        import pandas_market_calendars as mcal  # type: ignore
        return mcal.get_calendar("XNYS")
    except Exception:
        return None

def is_trading_day(d: dt.date) -> bool:
    cal = _try_get_nyse_calendar()
    if cal is None:
        return d.weekday() < 5  # fallback: weekend 제외
    start = dt.datetime.combine(d, dt.time(0, 0))
    end = dt.datetime.combine(d, dt.time(23, 59))
    sched = cal.schedule(start_date=start, end_date=end)
    return len(sched.index) > 0

def last_trading_day_of_month(d: dt.date) -> dt.date:
    # find last day of month then move backward to last trading day
    if d.month == 12:
        nxt = dt.date(d.year + 1, 1, 1)
    else:
        nxt = dt.date(d.year, d.month + 1, 1)
    cur = nxt - dt.timedelta(days=1)
    while not is_trading_day(cur):
        cur -= dt.timedelta(days=1)
    return cur

def first_trading_day_of_month(d: dt.date) -> dt.date:
    cur = dt.date(d.year, d.month, 1)
    while not is_trading_day(cur):
        cur += dt.timedelta(days=1)
    return cur

def is_last_trading_day_today(today: dt.date) -> bool:
    return today == last_trading_day_of_month(today)

def is_first_trading_day_today(today: dt.date) -> bool:
    return today == first_trading_day_of_month(today)