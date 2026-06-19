"""Stationary feature engineering shared across Forge log-return topics.

IMPORTANT (deployment): the functions here are called *inside* the exported
``predict()`` closure as well as during training. For the exported predict.pkl
to be self-contained, ``forge_models.common.export.export_predict`` registers
this module for pickle-by-value, so the worker process does NOT need
``forge_models`` installed. Keep the functions here pure (only numpy/pandas +
the constants below) so by-value pickling stays clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SIGMA_WINDOW, RET_LAGS, VOL_WINDOWS


def matrices_from_workflow_df(df: pd.DataFrame, n_bars: int):
    """Pull the kit's normalized OHLCV window back into (n, bars) matrices."""
    cols = lambda f: [f"feature_{f}_{i}" for i in range(n_bars)]
    C = df[cols("close")].to_numpy(dtype=float)
    H = df[cols("high")].to_numpy(dtype=float)
    L = df[cols("low")].to_numpy(dtype=float)
    V = df[cols("volume")].to_numpy(dtype=float)
    when = pd.to_datetime(df["open_time"], utc=True)
    return C, H, L, V, when


def compact_features(
    C, H, L, V, when,
    sigma_window: int = SIGMA_WINDOW,
    ret_lags=RET_LAGS,
    vol_windows=VOL_WINDOWS,
):
    """~30 stationary features per row + sigma (trailing hourly-return std).

    OHLC are normalized by the window's last close, so log-return and range
    features computed here equal those on raw prices (scale cancels). The
    keyword defaults are bound at definition time, so they travel with the
    function when it is pickled by value — train and inference stay identical
    as long as the caller uses the same arguments in both places.
    """
    eps = 1e-12
    B = C.shape[1]
    logc = np.log(np.maximum(C, eps))
    r1 = np.diff(logc, axis=1)                       # (n, B-1) per-bar log-returns
    sw = min(sigma_window, r1.shape[1])
    sigma = r1[:, -sw:].std(axis=1) + eps

    f: dict[str, np.ndarray] = {}
    for lag in ret_lags:
        if lag <= B - 1:
            f[f"ret_{lag}"] = logc[:, -1] - logc[:, -1 - lag]
    for w in vol_windows:
        if w <= r1.shape[1]:
            f[f"rv_{w}"] = r1[:, -w:].std(axis=1)
    if "rv_96" in f:
        f["rv_ratio_6_96"] = f["rv_6"] / (f["rv_96"] + eps)
        f["rv_ratio_24_96"] = f["rv_24"] / (f["rv_96"] + eps)
    for lag in (1, 6, 24):                            # scale-free momentum
        if f"ret_{lag}" in f:
            f[f"zret_{lag}"] = f[f"ret_{lag}"] / (sigma * np.sqrt(lag))
    g = np.clip(r1[:, -14:], 0, None).mean(axis=1)    # RSI-style balance
    l = np.clip(-r1[:, -14:], 0, None).mean(axis=1)
    f["rsi_14"] = 100.0 * g / (g + l + eps)
    hi24 = H[:, -24:].max(axis=1)
    lo24 = L[:, -24:].min(axis=1)
    f["hl_range_24"] = hi24 - lo24                    # already in last-close units
    f["range_pos_24"] = (C[:, -1] - lo24) / (hi24 - lo24 + eps)
    f["vol_z_24"] = (V[:, -1] - V[:, -24:].mean(axis=1)) / (V[:, -24:].std(axis=1) + eps)
    f["vol_trend"] = V[:, -6:].mean(axis=1) / (V[:, -48:].mean(axis=1) + eps)
    f["sigma_100"] = sigma
    hour = when.dt.hour.to_numpy()
    dow = when.dt.dayofweek.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return pd.DataFrame(f, index=when.index), sigma
