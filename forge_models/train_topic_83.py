#!/usr/bin/env python3
"""Topic 83 — 8h BTC/USD log-return.

Target: log_return = ln(price_{t+8h} / price_t). Submit the log-return itself.

Horizon mapping: interval="8h", target_bars=1 — the bar IS the 8h horizon, so
sigma100 (and the runner's vol_scale=sqrt(1)=1) equals the topic's ZPTAE
reference std over the last 100 non-overlapping 8h windows. Prefer
DATA_SOURCE=allora; the Binance fallback mis-handles hour intervals.

    export ALLORA_API_KEY=UP-...
    python forge_models/train_topic_83.py        # -> models/predict_topic_83.pkl
    PREDICT_PKL=models/predict_topic_83.pkl TOPIC_ID=83 python notebooks/deploy_worker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_models.common import TopicConfig, run

TOPIC = TopicConfig(
    topic_id=83,
    asset="BTC",
    horizon_label="8h",
    interval="8h",
    target_bars=1,                  # 1 bar ahead at 8h = 8 hours
    allora_tickers=["btcusd"],
    binance_tickers=["BTCUSDT"],
    max_plausible_logret=0.3,
    input_bars=128,                 # >=101 so sigma100 fits
    days_of_history=1500,           # 8h bars are ~3/day; pull more calendar history
    half_life_days=270,
)

if __name__ == "__main__":
    run(TOPIC)
