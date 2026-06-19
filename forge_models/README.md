# forge_models — unified Allora Forge training scripts

One `train_topic_<id>.py` per competition — but each is just a **`TopicConfig`
plus a `run(TOPIC)` call**. The entire train → evaluate → calibrate → export
flow lives once in `common/runner.py`; the scripts only carry what differs
between topics. Each run exports a **self-contained** `predict.pkl` into the
repo-level `models/` folder as `predict_topic_<id>.pkl`.

## Layout

```
forge_models/
  common/
    config.py        # cross-competition constants + model_path() helper
    data.py          # resolve_data_source(): allora (Tiingo) w/ binance fallback
    features.py      # matrices_from_workflow_df, compact_features (~30 stationary feats)
    calibration.py   # zptae_proxy, aspect_ratio, tune_shrink (shrinkage λ)
    export.py        # export_predict(): atomic, verified, self-contained .pkl
    runner.py        # TopicConfig + run(): the shared train/eval/export/predict flow
  train_topic_72.py  # 1h BTC/USD  (config only)
  ...
models/
  predict_topic_72.pkl   # written by the script (git-ignored)
```

## Topics

| Topic | Task | interval × target_bars | vol_scale |
|------:|------|------------------------|-----------|
| 72 | 1h BTC/USD | `1h` × 1   | 1.00 |
| 73 | 1h ETH/USD | `1h` × 1   | 1.00 |
| 71 | 8h NEAR/USD | `8h` × 1  | 1.00 |
| 83 | 8h BTC/USD | `8h` × 1   | 1.00 |
| 66 | 7d SOL/USD | `1h` × 168 | 12.96 |
| 67 | 7d BTC/USD | `1h` × 168 | 12.96 |
| 68 | 7d ETH/USD | `1h` × 168 | 12.96 |
| 70 | 7d NEAR/USD | `1h` × 168 | 12.96 |

`vol_scale = √target_bars`: the per-bar `sigma` from `compact_features` becomes
the horizon std. For `target_bars=1` it's 1.0 (so `sigma` already equals the
topic's 100-window ZPTAE reference std); for 7d it rescales the target to ~unit
variance so `Z_CLIP` doesn't shred it.

## Run

```bash
export ALLORA_API_KEY=UP-...                 # free at developer.allora.network
python forge_models/train_topic_72.py        # -> models/predict_topic_72.pkl
```

No API key? The scripts fall back to Binance klines automatically
(`DATA_SOURCE=binance` forces it).

## Deploy

The deploy scripts read `PREDICT_PKL` and `TOPIC_ID`:

```bash
PREDICT_PKL=models/predict_topic_72.pkl TOPIC_ID=72 \
    python notebooks/deploy_worker.py
```

## Self-contained artifacts (why `common/` is safe)

The exported `predict()` calls helpers from `common/features.py`. cloudpickle
would normally pickle those *by reference*, which would force the deployed
worker to have `forge_models` installed. `export_predict()` registers the
shared modules for **pickle-by-value** (`cloudpickle.register_pickle_by_value`),
so the `.pkl` carries the code with it — exactly as if everything lived in the
training script's `__main__`. The worker needs only the runtime deps
(`lightgbm`, `pandas`, `allora_forge_builder_kit`), not this package.

## Adding a new competition

1. Copy a similar `train_topic_<id>.py` (a 1h/8h one, or a 7d one if the horizon
   spans many bars) and change the `TopicConfig` fields: `topic_id`, `asset`,
   `horizon_label`, `interval`, `target_bars`, the ticker lists, and the
   warn-only `max_plausible_logret`.
2. Horizon = `interval` × `target_bars`. Keep `interval` at the topic's *update*
   cadence (so live features refresh that often) and set `target_bars` to span
   the prediction horizon — e.g. `1h`×168 for a 7d-ahead, hourly-updated topic.
   `vol_scale` is derived as `√target_bars` automatically.
3. Long horizons (target_bars ≫ 1): pass a `feature_kw` with longer
   `ret_lags`/`vol_windows` (keep `6,24,96` so the rv_ratio block stays valid)
   and bump `input_bars` ≥ `max(ret_lags)+1`. `run()` validates this.
4. That's it — `run(TOPIC)` handles backfill, walk-forward selection,
   calibration, the live smoke test, and the self-contained export. Only put
   genuinely new shared logic in `common/` (e.g. a non-log-return target would
   warrant a runner option), never copy the flow back into a script.

> Whitelist targets (log-return topics): DA > 0.55, Pearson r > 0.05,
> WRMSE improvement vs zero > 10%, WZPTAE improvement vs zero > 20%,
> `|log10(std_pred/std_true)| < 0.5`.
