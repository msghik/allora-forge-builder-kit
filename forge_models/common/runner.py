"""Shared train -> evaluate -> calibrate -> export runner for Forge log-return
topics. Each ``train_topic_<id>.py`` is reduced to a ``TopicConfig`` plus a
``run(TOPIC)`` call; everything below is identical across topics.

Design notes carried over from the per-topic scripts:

* Target. The kit's ``target_bars`` produces target = log(close.shift(-N)) -
  log(close), i.e. the N-bar-ahead log return. So ``interval``/``target_bars``
  together encode the horizon: 1h/1, 8h/1, or 1h/168 (=7d, hourly updates).
* Vol scale. ``compact_features`` returns ``sigma`` = per-BAR return std. The
  target spans ``target_bars`` bars, so the horizon std is ~ sigma *
  sqrt(target_bars). We vol-normalize by that ``sigma_h``. For target_bars=1
  (1h/8h topics) sqrt(1)=1, so sigma_h == sigma == the topic's ZPTAE reference
  std; for 7d it rescales the target to ~unit variance so Z_CLIP doesn't shred
  it. ``TopicConfig.vol_scale`` overrides the default only if you must.
* Feature reach. ``feature_kw`` (sigma_window/ret_lags/vol_windows) is passed
  verbatim to ``compact_features`` in BOTH training and predict(), so the
  feature set matches. Keep 6/24/96 in vol_windows so the rv_ratio block stays
  valid. None -> the compact_features defaults (good for 1h/8h).
* Leakage. Walk-forward folds use gap=target_bars so a test fold's targets
  never overlap train features (matters most at the 7d horizon).
* Self-contained export. predict() calls common.features helpers;
  export_predict registers that module pickle-by-value, so the artifact runs on
  a worker without forge_models installed.

Env overrides (applied on top of the config defaults): DAYS_OF_HISTORY,
INPUT_BARS, HALF_LIFE_DAYS, VOL_NORM_TARGET, FAMILIES, STRICT_ASPECT,
DATA_SOURCE, LIVE_CACHE_TTL, PREDICT_PKL.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import (
    Z_CLIP, ASPECT_BOUND, FIXED_COMMON, REG_EXTRA, model_path,
)
from .data import resolve_data_source
from .features import matrices_from_workflow_df, compact_features
from .calibration import aspect_ratio, tune_shrink
from .export import export_predict

# --- model search (small, conservative grid; shared by all topics) -------
N_SPLITS = 3
N_ESTIMATORS_MAX = 800
N_ESTIMATORS_CHECKPOINTS = [200, 400, 800]
LEARNING_RATES = [0.02, 0.05]
MAX_DEPTHS = [3, 5]
NUM_LEAVES = [15, 31]


@dataclass
class TopicConfig:
    """Everything that distinguishes one log-return topic from another."""
    topic_id: int
    asset: str                       # e.g. "BTC" — display only
    interval: str                    # workflow bar size, e.g. "1h", "8h"
    target_bars: int                 # bars ahead; horizon = interval * target_bars
    allora_tickers: list[str]
    binance_tickers: list[str]
    horizon_label: str               # e.g. "1h", "8h", "7d" — display only
    max_plausible_logret: float = 0.2   # warn-only guard on the live prediction
    input_bars: int = 128            # OHLCV window per sample (>= max feature reach)
    days_of_history: int = 1000
    half_life_days: float = 270.0
    live_cache_ttl: float = 90.0
    feature_kw: dict = field(default_factory=dict)   # -> compact_features kwargs
    vol_scale: float | None = None   # None -> sqrt(target_bars)

    def resolved_vol_scale(self) -> float:
        return float(np.sqrt(self.target_bars)) if self.vol_scale is None else float(self.vol_scale)

    def min_input_bars(self) -> int:
        lags = self.feature_kw.get("ret_lags")
        return (max(lags) + 1) if lags else 0


def run(cfg: TopicConfig) -> str:
    """Train, evaluate, calibrate, and export ``models/predict_topic_<id>.pkl``.
    Returns the artifact path. Honors the env overrides documented above."""
    # Heavy deps imported lazily so `import forge_models.common` stays light
    # (features/calibration/data/export don't need lightgbm or the kit).
    from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

    # --- env overrides on top of config defaults ---
    input_bars = int(os.environ.get("INPUT_BARS", cfg.input_bars))
    days_of_history = int(os.environ.get("DAYS_OF_HISTORY", cfg.days_of_history))
    half_life_days = float(os.environ.get("HALF_LIFE_DAYS", cfg.half_life_days))
    vol_norm = os.environ.get("VOL_NORM_TARGET", "1") != "0"
    families = tuple(os.environ.get("FAMILIES", "reg,clf").split(","))
    strict_aspect = os.environ.get("STRICT_ASPECT", "0") != "0"
    live_ttl = float(os.environ.get("LIVE_CACHE_TTL", cfg.live_cache_ttl))
    out_pkl = os.environ.get("PREDICT_PKL", str(model_path(cfg.topic_id)))

    interval = cfg.interval
    target_bars = cfg.target_bars
    feature_kw = dict(cfg.feature_kw)
    vol_scale = cfg.resolved_vol_scale()

    print("=" * 78)
    print(f"Topic {cfg.topic_id}: {cfg.horizon_label} {cfg.asset}/USD log-return "
          f"— train / evaluate / export (v2)")
    print("=" * 78)
    min_bars = cfg.min_input_bars()
    if input_bars < min_bars:
        sys.exit(f"INPUT_BARS={input_bars} too small for this feature set; need >= {min_bars}")
    print(f"history={days_of_history}d  input_bars={input_bars}  interval={interval}  "
          f"target_bars={target_bars}  vol_scale={vol_scale:.2f}  half_life={half_life_days}d")

    data_source, tickers, dm_kwargs = resolve_data_source(cfg.allora_tickers, cfg.binance_tickers)
    if data_source == "binance" and interval.endswith("h"):
        print("  WARNING: Binance fallback mis-handles hour intervals (reads 'Nh' as N min); "
              "set ALLORA_API_KEY and use DATA_SOURCE=allora for a correct target.")
    workflow = AlloraMLWorkflow(
        tickers=tickers,
        number_of_input_bars=input_bars,
        target_bars=target_bars,
        interval=interval,
        data_source=data_source,
        **dm_kwargs,
    )

    start_date = datetime.now(timezone.utc) - timedelta(days=days_of_history)
    print(f"\n[1/5] Backfilling {days_of_history} days of {interval} {tickers} ...")
    try:
        workflow.backfill(start=start_date)
    except Exception as e:  # cached parquet may still cover us
        print(f"  backfill warning: {e} — trying locally cached data")

    print("[2/5] Building features ...")
    df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
    df_all = df_all.dropna(subset=["target"]).reset_index(drop=True)

    C, H, L, V, when = matrices_from_workflow_df(df_all, input_bars)
    X, sigma_bar = compact_features(C, H, L, V, when, **feature_kw)
    sigma_h = sigma_bar * vol_scale        # per-bar std -> horizon std
    feature_cols = list(X.columns)
    y_raw = df_all["target"].to_numpy(dtype=float)
    y_train_space = np.clip(y_raw / sigma_h, -Z_CLIP, Z_CLIP) if vol_norm else y_raw

    # Recency weights: a sample half_life_days old counts half as much.
    age_days = (when.max() - when).dt.total_seconds().to_numpy() / 86400.0
    weights = 0.5 ** (age_days / half_life_days)

    if target_bars > 1:
        eff_n = days_of_history / (target_bars * _interval_hours(interval) / 24.0)
        extra = f", ~{eff_n:.0f} effective (non-overlapping) obs"
    else:
        extra = ""
    print(f"  {len(X):,} samples ({len(feature_cols)} features){extra} "
          f"({when.min()} → {when.max()})")

    y_sign = (y_raw > 0).astype(int)

    print("[3/5] Walk-forward grid search (selecting on calibrated ZPTAE-proxy improvement) ...")
    # gap=target_bars so a test fold's targets never overlap train features.
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=target_bars)
    evaluator = PerformanceEvaluator()
    results = []
    n = 0
    for family in families:
        for lr in LEARNING_RATES:
            for depth in MAX_DEPTHS:
                for leaves in NUM_LEAVES:
                    fold_models = []
                    for train_idx, test_idx in tscv.split(X):
                        if family == "clf":
                            m = LGBMClassifier(n_estimators=N_ESTIMATORS_MAX, learning_rate=lr,
                                               max_depth=depth, num_leaves=leaves, **FIXED_COMMON)
                            m.fit(X.iloc[train_idx], y_sign[train_idx],
                                  sample_weight=weights[train_idx])
                        else:
                            m = LGBMRegressor(n_estimators=N_ESTIMATORS_MAX, learning_rate=lr,
                                              max_depth=depth, num_leaves=leaves,
                                              **FIXED_COMMON, **REG_EXTRA)
                            m.fit(X.iloc[train_idx], y_train_space[train_idx],
                                  sample_weight=weights[train_idx])
                        fold_models.append((m, test_idx))
                    for n_est in N_ESTIMATORS_CHECKPOINTS:
                        n += 1
                        pred = np.full(len(X), np.nan)
                        for m, test_idx in fold_models:
                            if family == "clf":
                                p_up = m.predict_proba(X.iloc[test_idx], num_iteration=n_est)[:, 1]
                                pred[test_idx] = 2.0 * p_up - 1.0   # conviction in [-1, 1]
                            else:
                                pred[test_idx] = m.predict(X.iloc[test_idx], num_iteration=n_est)
                        mask = np.isfinite(pred)
                        # back to log-return units before scoring: classifier conviction is
                        # always vol-scaled, regressor only if trained in z-space. Use the
                        # horizon sigma (sigma_h), not the per-bar sigma.
                        scale = sigma_h[mask] if (family == "clf" or vol_norm) else 1.0
                        pred_lr = pred[mask] * scale
                        lam, imp, asp = tune_shrink(y_raw[mask], pred_lr, strict_aspect=strict_aspect)
                        da = float(np.mean(np.sign(pred_lr) == np.sign(y_raw[mask])))
                        results.append({"family": family, "n_estimators": n_est,
                                        "learning_rate": lr, "max_depth": depth,
                                        "num_leaves": leaves, "lambda": lam,
                                        "zptae_imp": imp, "aspect": asp,
                                        "da": da, "mask": mask, "pred_lr": pred_lr})
                        print(f"  [{family} {n:2d}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} -> "
                              f"zptae_imp={imp:+.2%} (lam={lam:.2f}, aspect={asp:+.2f}, DA={da:.4f})")

    results.sort(key=lambda r: (r["zptae_imp"], r["da"]), reverse=True)
    for family in families:
        fam_best = next(r for r in results if r["family"] == family)
        print(f"  best {family}: zptae_imp={fam_best['zptae_imp']:+.2%} DA={fam_best['da']:.4f} "
              f"aspect={fam_best['aspect']:+.2f}")
    best = results[0]
    lam = best["lambda"]
    print(f"\n[4/5] Best: family={best['family']} n={best['n_estimators']} "
          f"lr={best['learning_rate']} d={best['max_depth']} l={best['num_leaves']} lambda={lam:.2f}")
    print("  NOTE: lambda and config were chosen on the same OOS folds — expect the live")
    print("  numbers to be a bit weaker. The full 7-metric report on calibrated preds:")
    y_oos = y_raw[best["mask"]]
    # re-run calibration verbosely once, so any score-vs-whitelist tradeoff is shown
    tune_shrink(y_oos, best["pred_lr"], strict_aspect=strict_aspect, verbose=True)
    p_oos = lam * best["pred_lr"]
    report = evaluator.evaluate(y_true=pd.Series(y_oos), y_pred=pd.Series(p_oos))
    evaluator.print_report(report, detailed=False)
    print(f"  ZPTAE proxy improvement vs zero: {best['zptae_imp']:+.2%} (whitelist target > +20%)")
    print(f"  log-aspect ratio: {aspect_ratio(y_oos, p_oos):+.3f} (whitelist: within ±{ASPECT_BOUND})")
    if target_bars > 1:
        print("  (proxy uses a rolling std over overlapping samples; the official WZPTAE")
        print("   references non-overlapping windows — treat the proxy as relative.)")

    print("[5/5] Training production model on all data and exporting ...")
    hp = dict(n_estimators=best["n_estimators"], learning_rate=best["learning_rate"],
              max_depth=best["max_depth"], num_leaves=best["num_leaves"])
    family = best["family"]
    if family == "clf":
        final_model = LGBMClassifier(**hp, **FIXED_COMMON)
        final_model.fit(X, y_sign, sample_weight=weights)
    else:
        final_model = LGBMRegressor(**hp, **FIXED_COMMON, **REG_EXTRA)
        final_model.fit(X, y_train_space, sample_weight=weights)

    # --- values captured by the predict() closure (all picklable) ---
    ticker = tickers[0]
    n_bars = input_bars
    ds_name = data_source
    ds_tickers = list(tickers)
    asset = cfg.asset
    horizon = cfg.horizon_label
    max_plausible = cfg.max_plausible_logret
    topic_id = cfg.topic_id

    # The live workflow is rebuilt lazily inside predict: data managers hold
    # threads/locks (e.g. the Binance websocket client) that cannot be pickled,
    # and a fresh worker process needs its own live connection anyway. Only
    # plain config crosses the pickle boundary; the API key is re-read from the
    # runtime environment, never embedded in the artifact. _cache is a plain
    # dict (picklable, and cleared before export) — NOT a threading.Lock.
    _live: dict = {"wf": None}
    _cache: dict = {}

    def _live_workflow():
        if _live["wf"] is None:
            import os as _os
            from allora_forge_builder_kit import AlloraMLWorkflow as _Workflow
            kw = {}
            if ds_name == "allora":
                key = _os.environ.get("ALLORA_API_KEY", "").strip()
                if not key:
                    for p in (".allora_api_key", "notebooks/.allora_api_key"):
                        if _os.path.exists(p):
                            key = open(p).read().strip()
                            break
                if not key:
                    raise RuntimeError(
                        "ALLORA_API_KEY required at inference time for the allora data source")
                kw["api_key"] = key
            _live["wf"] = _Workflow(
                tickers=ds_tickers, number_of_input_bars=n_bars,
                target_bars=target_bars, interval=interval,
                data_source=ds_name, **kw)
        return _live["wf"]

    def predict(nonce: int | None = None) -> float:
        """Return the predicted log-return as a float (NOT a price).

        The SDK polls AND subscribes via websocket, so it can call predict twice
        for the same nonce. We memoize per nonce: the second call returns the
        identical number instantly instead of re-running a live fetch. This
        collapses the duplicate-submission race window (the root cause of the
        'signature verification failed' rejections) and avoids late submissions.
        A short TTL cache of the fetched bars keeps consecutive epochs cheap too.
        """
        import time as _t
        now = _t.time()
        # Per-nonce memo: same submission round -> same prediction, instantly.
        if nonce is not None and _cache.get("nonce") == nonce and "value" in _cache:
            return _cache["value"]

        live_row = _cache.get("data")
        if live_row is None or (now - _cache.get("data_ts", 0.0)) > live_ttl:
            live_row = _live_workflow().get_live_features(ticker=ticker)
            if live_row is None or len(live_row) == 0:
                raise ValueError("could not fetch live features")
            live_row = live_row.reset_index()
            if "open_time" not in live_row.columns:   # fall back to "now" for time feats
                live_row["open_time"] = pd.Timestamp.now(tz="UTC")
            _cache["data"] = live_row
            _cache["data_ts"] = now

        Cl, Hl, Ll, Vl, wl = matrices_from_workflow_df(live_row, n_bars)
        X_live, sigma_live_bar = compact_features(Cl, Hl, Ll, Vl, wl, **feature_kw)
        sigma_live_h = float(sigma_live_bar[0]) * vol_scale
        if family == "clf":
            p_up = float(final_model.predict_proba(X_live[feature_cols])[0, 1])
            log_ret = lam * (2.0 * p_up - 1.0) * sigma_live_h
        else:
            raw = float(final_model.predict(X_live[feature_cols])[0])
            log_ret = lam * raw * (sigma_live_h if vol_norm else 1.0)
        if abs(log_ret) > max_plausible:
            print(f"warning: implausible {horizon} log-return {log_ret:+.4f}")
        log_ret = float(log_ret)
        _cache["nonce"] = nonce
        _cache["value"] = log_ret
        print(f"{horizon} {asset} log-return prediction (nonce={nonce}): {log_ret:+.6f}")
        return log_ret

    print("  smoke-testing predict() against live data ...")
    test_val = predict()
    if not np.isfinite(test_val):
        sys.exit("predict() returned a non-finite value; not exporting")

    # Drop everything the smoke test created — the live connection (often
    # unpicklable) and the warm cache (a stale DataFrame the worker shouldn't
    # ship with). The worker rebuilds both on first call.
    _live["wf"] = None
    _cache.clear()

    out = export_predict(predict, out_pkl)
    print(f"\nSaved {out} (topic={topic_id}, family={family}, live data via '{ds_name}').")
    print("Deploy with your registered Forge wallet:")
    print(f"  PREDICT_PKL={out} TOPIC_ID={topic_id} python notebooks/deploy_worker.py")
    return str(out)


def _interval_hours(interval: str) -> float:
    """Hours per bar for '1h'/'8h'/'1d' style strings (best-effort, for display)."""
    iv = interval.strip().lower()
    if iv.endswith("h"):
        return float(iv[:-1] or 1)
    if iv.endswith("d"):
        return float(iv[:-1] or 1) * 24.0
    if iv.endswith("min") or iv.endswith("m"):
        num = iv[:-3] if iv.endswith("min") else iv[:-1]
        return float(num or 1) / 60.0
    return 1.0
