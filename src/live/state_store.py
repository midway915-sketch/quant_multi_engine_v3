from __future__ import annotations
import json
import os
import datetime as dt
from typing import Optional

def _path(state_dir: str) -> str:
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, "state.json")

def load_state(state_dir: str) -> dict:
    p = _path(state_dir)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state_dir: str, state: dict) -> None:
    p = _path(state_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def get_last_state(state: dict) -> Optional[str]:
    return state.get("last_state")

def get_last_change_date(state: dict) -> Optional[dt.date]:
    s = state.get("last_change_date")
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def set_state(state: dict, new_state: str, change_date: dt.date) -> dict:
    state["last_state"] = new_state
    state["last_change_date"] = change_date.strftime("%Y-%m-%d")
    return state