"""Topic discovery shared by train_all.py / deploy_all.py (and any future
tooling). Single source of truth for the ``train_topic_<id>.py`` convention and
the ``models/predict_topic_<id>.pkl`` artifact path. Stdlib-only so the
launchers stay dependency-light.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent          # forge_models/
REPO_ROOT = SCRIPTS_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
_TOPIC_RE = re.compile(r"^train_topic_(\d+)\.py$")


def discover_topics(scripts_dir: Path | None = None) -> list[int]:
    """All topic ids that have a ``train_topic_<id>.py`` script, sorted."""
    d = Path(scripts_dir) if scripts_dir else SCRIPTS_DIR
    ids = [int(m.group(1)) for f in d.glob("train_topic_*.py")
           if (m := _TOPIC_RE.match(f.name))]
    return sorted(set(ids))


def model_file(topic_id: int) -> Path:
    return MODELS_DIR / f"predict_topic_{topic_id}.pkl"


def topic_script(topic_id: int) -> Path:
    return SCRIPTS_DIR / f"train_topic_{topic_id}.py"


def parse_subset(arg: str | None, available: list[int]) -> list[int]:
    """Resolve a ``--topics 72,73`` string against the discovered topics."""
    if not arg:
        return available
    want = [int(x) for x in arg.replace(" ", "").split(",") if x]
    missing = [t for t in want if t not in available]
    if missing:
        sys.exit(f"no train_topic_{{{','.join(map(str, missing))}}}.py found "
                 f"(available: {', '.join(map(str, available))})")
    return want
