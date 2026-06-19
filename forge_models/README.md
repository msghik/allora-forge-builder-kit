# forge_models — unified Allora Forge training scripts

One `train_topic_<id>.py` per competition. Each script builds, evaluates, and
exports a **self-contained** `predict.pkl` into the repo-level `models/` folder
as `predict_topic_<id>.pkl`. Shared machinery lives in `common/`, so the
per-topic scripts stay thin and bug fixes happen in one place.

## Layout

```
forge_models/
  common/
    config.py        # cross-competition constants + model_path() helper
    data.py          # resolve_data_source(): allora (Tiingo) w/ binance fallback
    features.py      # matrices_from_workflow_df, compact_features (~30 stationary feats)
    calibration.py   # zptae_proxy, aspect_ratio, tune_shrink (shrinkage λ)
    export.py        # export_predict(): atomic, verified, self-contained .pkl
  train_topic_72.py  # 1h BTC/USD log-return
  ...
models/
  predict_topic_72.pkl   # written by the script (git-ignored)
```

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

1. Copy `train_topic_72.py` to `train_topic_<id>.py`.
2. Update the **competition identity** block: `TOPIC_ID`, `INTERVAL`,
   `TARGET_BARS`, `ALLORA_TICKERS`, `BINANCE_TICKERS`.
3. Adjust the target/feature/predict logic for the topic's spec (e.g. if it
   asks for a price rather than a log-return, or a different horizon).
4. Reuse `common/` for features, calibration, data, and export. Promote any
   genuinely shared new helper into `common/` rather than copying it.

> Whitelist targets (log-return topics): DA > 0.55, Pearson r > 0.05,
> WRMSE improvement vs zero > 10%, WZPTAE improvement vs zero > 20%,
> `|log10(std_pred/std_true)| < 0.5`.
