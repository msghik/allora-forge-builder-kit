#!/usr/bin/env python3
"""Train every Forge topic model in one go.

Discovers all ``train_topic_<id>.py`` scripts in this folder and runs each in a
fresh subprocess (a clean process per topic — the kit's data managers hold
threads/sockets that shouldn't be shared across runs). Output streams live so
you see each topic's progress; failures are collected and reported at the end
instead of aborting the batch.

    export ALLORA_API_KEY=UP-...
    python forge_models/train_all.py                 # train all topics
    python forge_models/train_all.py --topics 72,73   # just these
    python forge_models/train_all.py --skip-existing  # skip ones already built
    python forge_models/train_all.py --list           # show discovered topics
    python forge_models/train_all.py --stop-on-error  # abort on first failure

Each topic writes models/predict_topic_<id>.pkl. Env knobs (DAYS_OF_HISTORY,
FAMILIES, ...) pass straight through to the per-topic scripts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_models.discovery import (   # noqa: E402
    SCRIPTS_DIR, REPO_ROOT, discover_topics, model_file, topic_script, parse_subset,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train all (or a subset of) Forge topic models.")
    ap.add_argument("--topics", help="comma-separated subset, e.g. 72,73")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip topics whose models/predict_topic_<id>.pkl already exists")
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
    topics = parse_subset(args.topics, available)

    print(f"Training {len(topics)} topic(s): {', '.join(map(str, topics))}\n")
    results: list[tuple[int, str, float]] = []
    for t in topics:
        script = topic_script(t)
        if args.skip_existing and model_file(t).exists():
            print(f"=== topic {t}: SKIP (model exists: {model_file(t).name}) ===\n")
            results.append((t, "skipped", 0.0))
            continue
        print(f"{'=' * 78}\n=== topic {t}: training ({script.name}) ===\n{'=' * 78}")
        start = time.time()
        proc = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT))
        dur = time.time() - start
        ok = proc.returncode == 0 and model_file(t).exists()
        status = "ok" if ok else f"FAILED (exit {proc.returncode})"
        results.append((t, status, dur))
        print(f"\n=== topic {t}: {status} in {dur:.0f}s ===\n")
        if not ok and args.stop_on_error:
            print("Stopping (--stop-on-error).")
            break

    _summary(results)
    sys.exit(0 if all(s == "ok" or s == "skipped" for _, s, _ in results) else 1)


def _summary(results: list[tuple[int, str, float]]) -> None:
    print("=" * 78)
    print("Training summary")
    print("-" * 78)
    for t, status, dur in results:
        mark = "ok " if status == "ok" else ("-- " if status == "skipped" else "XX ")
        print(f"  {mark} topic {t:<4} {status:<22} {dur:>6.0f}s")
    n_ok = sum(s == "ok" for _, s, _ in results)
    n_skip = sum(s == "skipped" for _, s, _ in results)
    n_fail = sum(s not in ("ok", "skipped") for _, s, _ in results)
    print("-" * 78)
    print(f"  {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    print("=" * 78)


if __name__ == "__main__":
    main()
