#!/usr/bin/env python3
"""Topic 66 — 7d SOL/USD log-return (updated hourly).

Target: log_return = ln(price_{t+7d} / price_t). Submit the log-return itself.
Same 7d/hourly design as topic 68 — see its docstring and
forge_models.common.runner. SOL is a volatile alt: confirm Tiingo history depth
on the first backfill.

    export ALLORA_API_KEY=UP-...
    python forge_models/train_topic_66.py        # -> models/predict_topic_66.pkl
    PREDICT_PKL=models/predict_topic_66.pkl TOPIC_ID=66 python notebooks/deploy_worker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_models.common import TopicConfig, run, SIGMA_WINDOW

FEATURE_KW_7D = dict(
    sigma_window=SIGMA_WINDOW,
    ret_lags=(1, 2, 3, 4, 6, 12, 24, 48, 96, 168, 336),     # 1h ... 14d
    vol_windows=(6, 24, 96, 336),                            # keep 6/24/96 for rv_ratio
)

TOPIC = TopicConfig(
    topic_id=66,
    asset="SOL",
    horizon_label="7d",
    interval="1h",
    target_bars=168,                # 168 bars ahead at 1h = 7 days
    allora_tickers=["solusd"],
    binance_tickers=["SOLUSDT"],
    max_plausible_logret=1.0,       # volatile alt
    input_bars=384,                 # >= max(ret_lags)+1 = 337
    days_of_history=1400,
    half_life_days=365,
    live_cache_ttl=300,
    feature_kw=FEATURE_KW_7D,       # vol_scale defaults to sqrt(168)
)

if __name__ == "__main__":
    run(TOPIC)
