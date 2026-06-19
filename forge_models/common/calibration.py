"""ZPTAE-proxy scoring and magnitude (shrinkage) calibration.

These are training/evaluation-time utilities — they are NOT referenced inside
the exported ``predict()`` closure, so they don't need to be picklable. They
exist so every topic script scores and calibrates predictions the same way the
Forge whitelist criteria do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SIGMA_WINDOW, SHRINK_GRID, ASPECT_BOUND


def power_tanh(x: np.ndarray, p: float = 1.5) -> np.ndarray:
    """Smooth bounded transform — a stand-in for the competition's power-tanh.
    Only used to compare *relative* loss vs the zero baseline."""
    return np.tanh(np.abs(x)) ** p


def zptae_proxy(y_true: np.ndarray, y_pred: np.ndarray, window: int = SIGMA_WINDOW) -> float:
    """Mean power-tanh of |error| z-scored by the trailing std of the last
    ``window`` ground-truth log-returns (ref mean 0), like the topic's loss."""
    s = pd.Series(y_true)
    sigma = s.rolling(window, min_periods=window).std().shift(1).to_numpy()
    ok = np.isfinite(sigma) & (sigma > 0)
    if ok.sum() == 0:
        return float("nan")
    z = np.abs(y_pred[ok] - y_true[ok]) / sigma[ok]
    return float(np.mean(power_tanh(z)))


def aspect_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.log10((y_pred.std() + 1e-15) / (y_true.std() + 1e-15)))


def tune_shrink(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shrink_grid=SHRINK_GRID,
    aspect_bound: float = ASPECT_BOUND,
    strict_aspect: bool = False,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Pick a scale lambda for the predictions, balancing two pulls:

    * ZPTAE/WRMSE want weak signals scaled way down (the RMSE-optimal scale is
      lambda* = cov(pred, true)/var(pred), tiny when correlation is low);
    * the whitelist's |log10(std(pred)/std(true))| <= aspect_bound sets a
      *floor* on loudness — silence is not an option.

    Candidates: fixed grid + lambda* + the aspect-feasibility boundaries.
    Policy: take the best feasible lambda unless the unconstrained best beats
    it by more than 2pp of improvement, in which case prefer score (set
    strict_aspect=True to always stay in the whitelist band).
    Returns (lambda, zptae_improvement, aspect).
    """
    eps = 1e-15
    zp_zero = zptae_proxy(y_true, np.zeros_like(y_pred))
    sy, sp = y_true.std() + eps, y_pred.std() + eps
    lam_floor = (10.0 ** -aspect_bound) * sy / sp     # quietest feasible
    lam_ceil = (10.0 ** aspect_bound) * sy / sp       # loudest feasible
    lam_star = float(np.dot(y_pred, y_true) / (np.dot(y_pred, y_pred) + eps))
    cands = set(shrink_grid) | {lam_floor * 1.01, lam_ceil * 0.99}
    if lam_star > 0:
        cands |= {lam_star, min(max(lam_star, lam_floor * 1.01), lam_ceil * 0.99)}

    best, best_feasible = None, None
    for lam in sorted(c for c in cands if c > 0):
        zp = zptae_proxy(y_true, lam * y_pred)
        imp = 1.0 - zp / zp_zero if zp_zero and np.isfinite(zp) else float("-inf")
        asp = aspect_ratio(y_true, lam * y_pred)
        cand = (imp, lam, asp)
        if best is None or imp > best[0]:
            best = cand
        if abs(asp) <= aspect_bound and (best_feasible is None or imp > best_feasible[0]):
            best_feasible = cand
    chosen = best
    if best_feasible is not None and (strict_aspect or best_feasible[0] >= best[0] - 0.02):
        chosen = best_feasible
    elif verbose and best_feasible is not None:
        print(f"  note: taking lambda={best[1]:.3f} for score; aspect {best[2]:+.2f} "
              f"violates the +/-{aspect_bound} whitelist bound "
              f"(best feasible was {best_feasible[0]:+.2%} at lambda={best_feasible[1]:.3f};"
              f" strict_aspect=True forces compliance)")
    imp, lam, asp = chosen
    return lam, imp, asp
