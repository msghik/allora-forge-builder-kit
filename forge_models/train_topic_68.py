#!/usr/bin/env python3
"""Topic 68 — 7d ETH/USD log-return (updated hourly).

Target: log_return = ln(price_{t+7d} / price_t). Submit the log-return itself.

Horizon mapping: interval="1h", target_bars=168 (=7 days). Base resolution stays
1h so live features — and submissions — refresh hourly as the topic requires.
The runner sets vol_scale=sqrt(168) so the target is vol-normalized at the 7d
horizon (else Z_CLIP would shred z = r_7d / sigma_1h ~ N(0, ~13)), passes the
extended feature geometry below to compact_features in both training and
predict(), and uses gap=target_bars so test targets never overlap train
features.

Effective sample size note: hourly samples of a 7d return overlap heavily
(~167/168 shared), so effective independent obs ~ calendar_days / 7.

    export ALLORA_API_KEY=UP-...
    python forge_models/train_topic_68.py        # -> models/predict_topic_68.pkl
    PREDICT_PKL=models/predict_topic_68.pkl TOPIC_ID=68 python notebooks/deploy_worker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_models.common import TopicConfig, run, SIGMA_WINDOW

# Feature reach extended to 14 days. Keep 6/24/96 in vol_windows so
# compact_features' hardcoded rv_ratio_6_96 / rv_ratio_24_96 block stays valid.
FEATURE_KW_7D = dict(
    sigma_window=SIGMA_WINDOW,
    ret_lags=(1, 2, 3, 4, 6, 12, 24, 48, 96, 168, 336),     # 1h ... 14d
    vol_windows=(6, 24, 96, 336),                            # ... 14d
)

TOPIC = TopicConfig(
    topic_id=68,
    asset="ETH",
    horizon_label="7d",
    interval="1h",
    target_bars=168,                # 168 bars ahead at 1h = 7 days
    allora_tickers=["ethusd"],
    binance_tickers=["ETHUSDT"],
    max_plausible_logret=0.7,
    input_bars=384,                 # >= max(ret_lags)+1 = 337
    days_of_history=1400,           # 7d targets need long calendar history
    half_life_days=365,
    live_cache_ttl=300,
    feature_kw=FEATURE_KW_7D,       # vol_scale defaults to sqrt(168)
)

if __name__ == "__main__":
    run(TOPIC)
