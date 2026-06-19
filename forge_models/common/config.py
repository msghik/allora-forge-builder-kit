"""Shared defaults for Allora Forge training scripts.

These are *cross-competition* constants — things tied to how the Forge scores
log-return topics rather than to any single asset/interval. Competition-specific
knobs (interval, target_bars, input_bars, tickers, half-life, ...) live in the
per-topic script, not here.
"""
from __future__ import annotations

from pathlib import Path

# --- repo layout ---------------------------------------------------------
# forge_models/common/config.py  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def model_path(topic_id: int | str, prefix: str = "predict") -> Path:
    """Canonical artifact path for a topic, e.g. models/predict_topic_72.pkl.

    The deploy scripts read PREDICT_PKL, so point them here:
        PREDICT_PKL=models/predict_topic_72.pkl TOPIC_ID=72 \
            python notebooks/deploy_worker.py
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{prefix}_topic_{topic_id}.pkl"


# --- scoring reference (shared across log-return topics) -----------------
# The Forge ZPTAE/WZPTAE reference std is computed over a rolling window of the
# last 100 ground-truth log-returns; keep the model's vol reference in sync.
SIGMA_WINDOW = 100
Z_CLIP = 6.0                       # clip vol-normalized targets' tails

# --- magnitude calibration (whitelist constraints) -----------------------
SHRINK_GRID = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3]
ASPECT_BOUND = 0.5                 # |log10(std(pred)/std(true))| < 0.5

# --- conservative LightGBM defaults shared by the topic scripts ----------
FIXED_COMMON = dict(
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
)
REG_EXTRA = dict(objective="huber", alpha=1.0)

# --- default compact-feature geometry ------------------------------------
RET_LAGS = (1, 2, 3, 4, 6, 12, 24, 48, 96)
VOL_WINDOWS = (6, 24, 96)
