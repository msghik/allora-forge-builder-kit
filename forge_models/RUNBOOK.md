# Runbook — train & deploy your Forge models

Exact, copy-paste steps. Commands are **PowerShell**, run from the **repo root**
(`allora-forge-builder-kit/`). A topic = one competition; you train one model per
topic and run one worker per topic.

---

## 0. One-time setup

1. **Install deps** (your conda/venv with the kit, lightgbm, pandas):
   ```powershell
   pip install -e .          # or: conda env create -f environment.yml
   ```
2. **Get an Allora API key** (free): https://developer.allora.network
3. **Wallet**: have your Allora wallet's 24-word **mnemonic** and its `allo1…`
   **address**. Fund it from the testnet faucet and register it to each topic in
   the Model Forge (use this same address everywhere).

---

## 1. Set your secrets (every new terminal session)

```powershell
$env:ALLORA_API_KEY = "UP-xxxxxxxx"              # needed for BOTH training and inference
$env:MNEMONIC       = "word1 word2 ... word24"   # your wallet seed phrase
$env:WALLET_ADDRESS = "allo1xxxxxxxx"            # required whenever MNEMONIC is set
```

> ⚠️ The mnemonic is a secret. Only paste it into your shell session. Never
> commit it to a file (`.env`, `.allora_key`, etc. are git-ignored).

---

## 2. Train all models

```powershell
python forge_models/train_all.py --skip-existing
```

- Writes one artifact per topic to `models/predict_topic_<id>.pkl`.
- `--skip-existing` skips topics already built; drop it to retrain everything.
- Other options: `--topics 72,73` (subset), `--list`, `--stop-on-error`.
- Each topic runs in its own process and prints a 5-step log; a summary table
  prints at the end. A topic is `ok` only if its `.pkl` was actually written.

Train just one while iterating:
```powershell
python forge_models/train_topic_72.py
```

---

## 3. Deploy a worker for every topic

With your secrets from step 1 still set:

```powershell
python forge_models/deploy_all.py
```

- Deploys one managed background worker per topic **that has a model** (missing
  ones are skipped with a note). Each uses your wallet (`MNEMONIC`/`WALLET_ADDRESS`).
- Preview first without deploying: `python forge_models/deploy_all.py --dry-run`
- Subset: `python forge_models/deploy_all.py --topics 72,73`
- It warns if `MNEMONIC` or `ALLORA_API_KEY` is missing (without `MNEMONIC` it
  would use an auto wallet, **not** yours).

Deploy one topic manually (equivalent of one loop step):
```powershell
$env:TOPIC_ID    = "72"
$env:PREDICT_PKL = "models/predict_topic_72.pkl"
python notebooks/deploy_worker.py
```

---

## 4. Monitor

```powershell
python -m allora_forge_builder_kit.workerctl dashboard     # all workers, one view
python -m allora_forge_builder_kit.web_dashboard           # browser dashboard
Get-Content "worker_logs/worker_72_<address>.log" -Wait    # tail one worker
```

---

## TL;DR (full run)

```powershell
$env:ALLORA_API_KEY = "UP-..."
$env:MNEMONIC       = "word1 ... word24"
$env:WALLET_ADDRESS = "allo1..."
python forge_models/train_all.py --skip-existing
python forge_models/deploy_all.py
python -m allora_forge_builder_kit.workerctl dashboard
```

---

## Gotchas (read once)

- **`ALLORA_API_KEY` must be set when you deploy**, not just when you train. The
  worker re-fetches live features at inference and reads the key from the
  environment — it is **not** baked into the `.pkl`. Setting it before
  `deploy_all.py` means each spawned worker inherits it.
- **Run from the repo root** so `models/predict_topic_<id>.pkl` resolves and all
  workers share one `worker_state.db` / `worker_logs/`.
- **Use the Allora data source for the 8h/7d topics.** The Binance fallback
  mis-handles hour intervals (reads `8h` as 8 min). With `ALLORA_API_KEY` set the
  source is Allora automatically; the scripts warn if they fall back.
- **First backfill is slow** — the Allora source resamples from 1-minute data, so
  1000–1400 days takes a while per topic. Subsequent runs reuse cached parquet.
- **Adding a new competition later**: drop a `train_topic_<id>.py` into
  `forge_models/` (copy a similar one, change the `TopicConfig`). `train_all.py`
  and `deploy_all.py` pick it up automatically — see `README.md` for the config
  fields and extension hooks.
