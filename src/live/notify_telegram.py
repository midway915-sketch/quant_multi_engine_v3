import os
import requests

def tg_enabled(cfg: dict) -> bool:
    return bool(cfg.get("notify", {}).get("telegram", {}).get("enabled", False))

def send_telegram(cfg: dict, text: str) -> None:
    if not tg_enabled(cfg):
        return

    tg = cfg["notify"]["telegram"]
    token = os.environ.get(tg.get("bot_token_env", "TG_BOT_TOKEN"), "").strip()
    chat_id = os.environ.get(tg.get("chat_id_env", "TG_CHAT_ID"), "").strip()

    if not token or not chat_id:
        raise RuntimeError("Telegram env missing: set TG_BOT_TOKEN and TG_CHAT_ID (or config env keys).")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
    r.raise_for_status()