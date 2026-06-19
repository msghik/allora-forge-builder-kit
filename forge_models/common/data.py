"""Data-source resolution shared by the topic scripts.

Prefer the Allora/Tiingo source (it matches how topics are scored); fall back
to Binance klines when no API key is available.
"""
from __future__ import annotations

import os


def load_api_key() -> str:
    """Read ALLORA_API_KEY from the env, then from common dotfiles."""
    api_key = os.environ.get("ALLORA_API_KEY", "").strip()
    if api_key:
        return api_key
    for p in (".allora_api_key", "notebooks/.allora_api_key"):
        if os.path.exists(p):
            return open(p).read().strip()
    return ""


def resolve_data_source(
    allora_tickers: list[str],
    binance_tickers: list[str],
) -> tuple[str, list[str], dict]:
    """Return (data_source, tickers, data_manager_kwargs).

    DATA_SOURCE=allora|binance forces a source; otherwise we use allora when an
    API key is present and fall back to binance when it is not. The two ticker
    lists let each competition spell its symbol per source (e.g. ``btcusd`` vs
    ``BTCUSDT``).
    """
    forced = os.environ.get("DATA_SOURCE", "").strip().lower()
    api_key = load_api_key()

    if forced == "binance" or (not api_key and forced != "allora"):
        reason = "forced" if forced == "binance" else "no ALLORA_API_KEY found"
        print(f"Data source: binance ({reason})")
        return "binance", list(binance_tickers), {}

    if not api_key:
        raise SystemExit(
            "ALLORA_API_KEY required for DATA_SOURCE=allora — "
            "get one free at https://developer.allora.network"
        )
    print("Data source: allora (Tiingo via Atlas)")
    return "allora", list(allora_tickers), {"api_key": api_key}
