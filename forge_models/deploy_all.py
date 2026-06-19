#!/usr/bin/env python3
"""Deploy a worker for every Forge topic that has a trained model.

Discovers all ``train_topic_<id>.py`` scripts, then for each topic whose
``models/predict_topic_<id>.pkl`` exists, runs ``notebooks/deploy_worker.py``
with TOPIC_ID + PREDICT_PKL set. Topics missing a model are skipped (train them
with ``python forge_models/train_all.py`` first). Each deploy starts a managed
background worker; this script just launches them in turn and reports.

Your own wallet (recommended): set these before running so every worker uses it
(they pass through to deploy_worker.py via the inherited environment):

    # PowerShell
    $env:ALLORA_API_KEY = "UP-..."     # also needed at inference (live fetch)
    $env:MNEMONIC       = "word1 ... word24"
    $env:WALLET_ADDRESS = "allo1..."
    python forge_models/deploy_all.py

    python forge_models/deploy_all.py --topics 72,73   # just these
    python forge_models/deploy_all.py --dry-run        # show what would run
    python forge_models/deploy_all.py --list           # show discovered topics

Without MNEMONIC the kit falls back to a managed/auto wallet (NOT your own) —
the script warns about this. WALLET_ADDRESS is required whenever MNEMONIC is set.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_models.discovery import (   # noqa: E402
    REPO_ROOT, discover_topics, model_file, parse_subset,
)

DEPLOY_SCRIPT = REPO_ROOT / "notebooks" / "deploy_worker.py"


def _check_wallet_env() -> None:
    mnemonic = os.environ.get("MNEMONIC", "").strip()
    address = os.environ.get("WALLET_ADDRESS", "").strip()
    if mnemonic and not address:
        sys.exit("WALLET_ADDRESS must be set when MNEMONIC is provided.")
    if not mnemonic:
        print("  WARNING: MNEMONIC not set — workers will use a managed/auto wallet, "
              "NOT your own. Set MNEMONIC + WALLET_ADDRESS to deploy on your wallet.")
    else:
        print(f"  Wallet: {address} (from MNEMONIC)")
    if not os.environ.get("ALLORA_API_KEY", "").strip():
        print("  WARNING: ALLORA_API_KEY not set — workers need it at inference to fetch "
              "live features (the key is read from the environment, not the .pkl).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy a worker for every topic with a trained model.")
    ap.add_argument("--topics", help="comma-separated subset, e.g. 72,73")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, deploy nothing")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort on the first failure (default: continue and report)")
    ap.add_argument("--list", action="store_true", help="list discovered topics and exit")
    args = ap.parse_args()

    available = discover_topics()
    if not available:
        sys.exit("no train_topic_<id>.py scripts found in forge_models/")
    if args.list:
        print("Discovered topics:", ", ".join(map(str, available)))
        return
    if not DEPLOY_SCRIPT.exists():
        sys.exit(f"deploy script not found: {DEPLOY_SCRIPT}")
    topics = parse_subset(args.topics, available)

    print("Wallet / environment:")
    _check_wallet_env()
    print()

    results: list[tuple[int, str, float]] = []
    for t in topics:
        model = model_file(t)
        if not model.exists():
            print(f"=== topic {t}: SKIP (no model: {model.name}; train it first) ===\n")
            results.append((t, "skipped-no-model", 0.0))
            continue
        env = os.environ.copy()
        env["TOPIC_ID"] = str(t)
        env["PREDICT_PKL"] = str(model)        # absolute path -> cwd-independent
        if args.dry_run:
            print(f"=== topic {t}: DRY-RUN -> TOPIC_ID={t} PREDICT_PKL={model.name} "
                  f"python notebooks/deploy_worker.py ===\n")
            results.append((t, "dry-run", 0.0))
            continue
        print(f"{'=' * 78}\n=== topic {t}: deploying (model {model.name}) ===\n{'=' * 78}")
        start = time.time()
        proc = subprocess.run([sys.executable, str(DEPLOY_SCRIPT)], cwd=str(REPO_ROOT), env=env)
        dur = time.time() - start
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append((t, status, dur))
        print(f"\n=== topic {t}: {status} in {dur:.0f}s ===\n")
        if proc.returncode != 0 and args.stop_on_error:
            print("Stopping (--stop-on-error).")
            break

    _summary(results)
    if not args.dry_run:
        print("\nMonitor workers:")
        print("  python -m allora_forge_builder_kit.workerctl dashboard")
        print("  python -m allora_forge_builder_kit.web_dashboard")
    ok_states = ("ok", "dry-run", "skipped-no-model")
    sys.exit(0 if all(s in ok_states for _, s, _ in results) else 1)


def _summary(results: list[tuple[int, str, float]]) -> None:
    print("=" * 78)
    print("Deploy summary")
    print("-" * 78)
    for t, status, dur in results:
        mark = "ok " if status == "ok" else ("-- " if status.startswith(("skipped", "dry")) else "XX ")
        print(f"  {mark} topic {t:<4} {status:<22} {dur:>6.0f}s")
    n_ok = sum(s == "ok" for _, s, _ in results)
    n_skip = sum(s.startswith(("skipped", "dry")) for _, s, _ in results)
    n_fail = sum(not (s == "ok" or s.startswith(("skipped", "dry"))) for _, s, _ in results)
    print("-" * 78)
    print(f"  {n_ok} deployed, {n_skip} skipped, {n_fail} failed")
    print("=" * 78)


if __name__ == "__main__":
    main()
