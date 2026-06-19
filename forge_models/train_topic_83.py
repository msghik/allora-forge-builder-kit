#!/usr/bin/env python3
"""Topic 83 — 8h BTC/USD log-return: train / evaluate / export predict.pkl.

Same task family as topic 72 (BTC/USD log-return) but at an **8-hour horizon**
instead of 1 hour. Target: log_return = ln(price_{t+8h} / price_t). The
exported ``predict(nonce)`` returns the predicted **log-return as a float** —
for log-return topics you submit the log-return itself, NOT a price.

Horizon mapping (the one real difference from topic 72):
    interval="8h", target_bars=1
We model the 8h return as a *single 8h bar* rather than 8 stacked 1h bars. This
keeps the v2 design's key invariant — ``sigma100`` (trailing std of the last
100 per-bar log-returns) equals the competition's ZPTAE reference std, which
for this topic is computed over the **last 100 non-overlapping 8h windows**.
With interval="1h"/target_bars=8 instead, sigma100 would measure 1h volatility
and mismatch that reference. Because the bar IS the horizon, all of
``forge_models/common`` (features, vol-normalization, calibration) is reused
unchanged.

Data source note: the Allora/Tiingo source resamples 1-minute data, so it
handles "8h" cleanly and is what the topic is scored on. The Binance fallback
has minute-assuming arithmetic for hour intervals (it mis-reads "8h" as 8
minutes) — prefer DATA_SOURCE=allora here; Binance is only a no-API-key escape
hatch and is unreliable at this horizon.

    export ALLORA_API_KEY=UP-...
    python forge_models/train_topic_83.py
    # -> writes models/predict_topic_83.pkl

Deploy:
    PREDICT_PKL=models/predict_topic_83.pkl TOPIC_ID=83 \
        python notebooks/deploy_worker.py

Env knobs: DAYS_OF_HISTORY (1500), INPUT_BARS (128), HALF_LIFE_DAYS (270),
VOL_NORM_TARGET (1; set 0 to A/B raw-target training), FAMILIES (reg,clf),
STRICT_ASPECT (0; set 1 to always keep the whitelist aspect band),
DATA_SOURCE (allora|binance), LIVE_CACHE_TTL (90), PREDICT_PKL.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

# Make `forge_models` importable whether run as a script or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator
from forge_models.common import (
    model_path, resolve_data_source,
    matrices_from_workflow_df, compact_features,
    zptae_proxy, aspect_ratio, tune_shrink, export_predict,
    SIGMA_WINDOW, Z_CLIP, ASPECT_BOUND, FIXED_COMMON, REG_EXTRA,
)

# --- competition identity ---
TOPIC_ID = 83
INTERVAL = "8h"                 # each bar is the 8h prediction horizon
TARGET_BARS = 1                 # 1 bar ahead at 8h interval = 8 hours
ASSET = "BTC"
HORIZON_LABEL = "8h"
ALLORA_TICKERS = ["btcusd"]
BINANCE_TICKERS = ["BTCUSDT"]
# 8h moves are larger than 1h moves; loosen the implausibility guard accordingly.
MAX_PLAUSIBLE_LOGRET = 0.3

# --- windowing / history ---
INPUT_BARS = int(os.environ.get("INPUT_BARS", "128"))   # >=101 so sigma100 fits
# 8h bars are 8x scarcer than 1h bars per calendar day (~3/day). Pull more
# calendar history so the model still sees plenty of 8h samples.
DAYS_OF_HISTORY = int(os.environ.get("DAYS_OF_HISTORY", "1500"))

# --- regime handling ---
VOL_NORM_TARGET = os.environ.get("VOL_NORM_TARGET", "1") != "0"
HALF_LIFE_DAYS = float(os.environ.get("HALF_LIFE_DAYS", "270"))

# --- model search (small, conservative grid) ---
N_SPLITS = 3
N_ESTIMATORS_MAX = 800
N_ESTIMATORS_CHECKPOINTS = [200, 400, 800]
LEARNING_RATES = [0.02, 0.05]
MAX_DEPTHS = [3, 5]
NUM_LEAVES = [15, 31]
FAMILIES = tuple(os.environ.get("FAMILIES", "reg,clf").split(","))

STRICT_ASPECT = os.environ.get("STRICT_ASPECT", "0") != "0"

OUT_PKL = os.environ.get("PREDICT_PKL", str(model_path(TOPIC_ID)))


def main() -> None:
    print("=" * 78)
    print(f"Topic {TOPIC_ID}: {HORIZON_LABEL} {ASSET}/USD log-return — train / evaluate / export (v2)")
    print("=" * 78)
    print(f"history={DAYS_OF_HISTORY}d  input_bars={INPUT_BARS}  interval={INTERVAL}  "
          f"vol_norm_target={VOL_NORM_TARGET}  half_life={HALF_LIFE_DAYS}d")

    data_source, tickers, dm_kwargs = resolve_data_source(ALLORA_TICKERS, BINANCE_TICKERS)
    if data_source == "binance":
        print("  WARNING: Binance fallback mis-handles hour intervals (reads '8h' as 8 min); "
              "set ALLORA_API_KEY and use DATA_SOURCE=allora for a correct 8h target.")
    workflow = AlloraMLWorkflow(
        tickers=tickers,
        number_of_input_bars=INPUT_BARS,
        target_bars=TARGET_BARS,
        interval=INTERVAL,
        data_source=data_source,
        **dm_kwargs,
    )

    start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
    print(f"\n[1/5] Backfilling {DAYS_OF_HISTORY} days of {INTERVAL} {tickers} ...")
    try:
        workflow.backfill(start=start_date)
    except Exception as e:  # cached parquet may still cover us
        print(f"  backfill warning: {e} — trying locally cached data")

    print("[2/5] Building features ...")
    df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
    df_all = df_all.dropna(subset=["target"]).reset_index(drop=True)

    C, H, L, V, when = matrices_from_workflow_df(df_all, INPUT_BARS)
    X, sigma100 = compact_features(C, H, L, V, when)
    feature_cols = list(X.columns)
    y_raw = df_all["target"].to_numpy(dtype=float)
    y_train_space = np.clip(y_raw / sigma100, -Z_CLIP, Z_CLIP) if VOL_NORM_TARGET else y_raw

    # Recency weights: a sample HALF_LIFE_DAYS old counts half as much.
    age_days = (when.max() - when).dt.total_seconds().to_numpy() / 86400.0
    weights = 0.5 ** (age_days / HALF_LIFE_DAYS)

    print(f"  {len(X):,} samples, {len(feature_cols)} features "
          f"({when.min()} → {when.max()})")

    y_sign = (y_raw > 0).astype(int)

    print("[3/5] Walk-forward grid search (selecting on calibrated ZPTAE-proxy improvement) ...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
    evaluator = PerformanceEvaluator()
    results = []
    n = 0
    for family in FAMILIES:
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
                        # back to log-return units before any scoring: the classifier's
                        # conviction is always vol-scaled, the regressor only if it was
                        # trained in z-space
                        scale = sigma100[mask] if (family == "clf" or VOL_NORM_TARGET) else 1.0
                        pred_lr = pred[mask] * scale
                        lam, imp, asp = tune_shrink(y_raw[mask], pred_lr, strict_aspect=STRICT_ASPECT)
                        da = float(np.mean(np.sign(pred_lr) == np.sign(y_raw[mask])))
                        results.append({"family": family, "n_estimators": n_est,
                                        "learning_rate": lr, "max_depth": depth,
                                        "num_leaves": leaves, "lambda": lam,
                                        "zptae_imp": imp, "aspect": asp,
                                        "da": da, "mask": mask, "pred_lr": pred_lr})
                        print(f"  [{family} {n:2d}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} -> "
                              f"zptae_imp={imp:+.2%} (lam={lam:.2f}, aspect={asp:+.2f}, DA={da:.4f})")

    results.sort(key=lambda r: (r["zptae_imp"], r["da"]), reverse=True)
    for family in FAMILIES:
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
    tune_shrink(y_oos, best["pred_lr"], strict_aspect=STRICT_ASPECT, verbose=True)
    p_oos = lam * best["pred_lr"]
    report = evaluator.evaluate(y_true=pd.Series(y_oos), y_pred=pd.Series(p_oos))
    evaluator.print_report(report, detailed=False)
    zp_imp = best["zptae_imp"]
    print(f"  ZPTAE proxy improvement vs zero: {zp_imp:+.2%} (whitelist target > +20%)")
    print(f"  log-aspect ratio: {aspect_ratio(y_oos, p_oos):+.3f} (whitelist: within ±{ASPECT_BOUND})")

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

    ticker = tickers[0]
    vol_norm = VOL_NORM_TARGET
    n_bars = INPUT_BARS
    ds_name = data_source
    ds_tickers = list(tickers)
    asset = ASSET
    horizon = HORIZON_LABEL
    max_plausible = MAX_PLAUSIBLE_LOGRET
    live_ttl = float(os.environ.get("LIVE_CACHE_TTL", "90"))

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
                target_bars=TARGET_BARS, interval=INTERVAL,
                data_source=ds_name, **kw)
        return _live["wf"]

    def predict(nonce: int | None = None) -> float:
        """Return the predicted 8h BTC/USD **log-return** (not a price).

        The SDK polls AND subscribes via websocket, so it can call predict twice
        for the same nonce. We memoize per nonce: the second call returns the
        identical number instantly instead of re-running a ~25s live fetch. This
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
        X_live, sigma_live = compact_features(Cl, Hl, Ll, Vl, wl)
        if family == "clf":
            p_up = float(final_model.predict_proba(X_live[feature_cols])[0, 1])
            log_ret = lam * (2.0 * p_up - 1.0) * float(sigma_live[0])
        else:
            raw = float(final_model.predict(X_live[feature_cols])[0])
            log_ret = lam * raw * (float(sigma_live[0]) if vol_norm else 1.0)
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

    out = export_predict(predict, OUT_PKL)
    print(f"\nSaved {out} (topic={TOPIC_ID}, family={family}, live data via '{ds_name}').")
    print("Deploy with your registered Forge wallet:")
    print(f"  PREDICT_PKL={out} TOPIC_ID={TOPIC_ID} python notebooks/deploy_worker.py")


if __name__ == "__main__":
    main()
