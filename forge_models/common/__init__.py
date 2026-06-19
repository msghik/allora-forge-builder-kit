"""Reusable building blocks shared by the per-topic Forge training scripts."""
from __future__ import annotations

from .config import (
    REPO_ROOT, MODELS_DIR, model_path,
    SIGMA_WINDOW, Z_CLIP, SHRINK_GRID, ASPECT_BOUND,
    FIXED_COMMON, REG_EXTRA, RET_LAGS, VOL_WINDOWS,
)
from .data import load_api_key, resolve_data_source
from .features import matrices_from_workflow_df, compact_features
from .calibration import power_tanh, zptae_proxy, aspect_ratio, tune_shrink
from .export import export_predict

__all__ = [
    "REPO_ROOT", "MODELS_DIR", "model_path",
    "SIGMA_WINDOW", "Z_CLIP", "SHRINK_GRID", "ASPECT_BOUND",
    "FIXED_COMMON", "REG_EXTRA", "RET_LAGS", "VOL_WINDOWS",
    "load_api_key", "resolve_data_source",
    "matrices_from_workflow_df", "compact_features",
    "power_tanh", "zptae_proxy", "aspect_ratio", "tune_shrink",
    "export_predict",
]
