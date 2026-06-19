#!/usr/bin/env python3
"""Topic 73 — 1h ETH/USD log-return.

Target: log_return = ln(price_{t+1h} / price_t). Submit the log-return itself.
Train/evaluate/calibrate/export machinery lives in
``forge_models.common.runner``; this file is just the topic's config.

    export ALLORA_API_KEY=UP-...
    python forge_models/train_topic_73.py        # -> models/predict_topic_73.pkl
    PREDICT_PKL=models/predict_topic_73.pkl TOPIC_ID=73 python notebooks/deploy_worker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_models.common import TopicConfig, run

TOPIC = TopicConfig(
    topic_id=73,
    asset="ETH",
    horizon_label="1h",
    interval="1h",
    target_bars=1,                  # 1 bar ahead at 1h = 1 hour
    allora_tickers=["ethusd"],
    binance_tickers=["ETHUSDT"],
    max_plausible_logret=0.2,
    input_bars=128,                 # >=101 so sigma100 fits
    days_of_history=1000,
    half_life_days=270,
)

if __name__ == "__main__":
    run(TOPIC)
