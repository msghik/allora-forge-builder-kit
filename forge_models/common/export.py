"""Self-contained, atomic export of a ``predict()`` closure to a .pkl artifact.

The exported ``predict`` calls helpers from ``forge_models.common.features``.
By default cloudpickle would pickle those *by reference*, which would force the
deployed worker to have ``forge_models`` importable. We instead register the
shared modules for **pickle-by-value** so the artifact is fully self-contained
— exactly as it would be if everything lived in the training script's
``__main__``. This is what lets the topic scripts stay thin without breaking
deployment.

Export is atomic and verified: a crash mid-dump must never leave a truncated
predict.pkl behind (a 0-byte artifact crashes the worker with EOFError).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import cloudpickle

from . import features as _features
from . import config as _config


def export_predict(predict_fn, out_path, extra_by_value_modules=()) -> Path:
    """Pickle ``predict_fn`` to ``out_path`` atomically, self-contained.

    ``extra_by_value_modules`` lets a caller register additional shared modules
    whose functions the predict closure calls (e.g. a custom feature builder);
    the feature/config helpers are always registered. ``None`` entries and the
    ``__main__``/``builtins`` modules are ignored — functions defined in the
    running script (``__main__``) are already captured by value by cloudpickle —
    and duplicates are de-duplicated so registration stays balanced.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_value = {}
    for mod in (_features, _config, *extra_by_value_modules):
        name = getattr(mod, "__name__", None)
        if name and name not in ("__main__", "builtins"):
            by_value.setdefault(name, mod)
    mods = list(by_value.values())

    for mod in mods:
        cloudpickle.register_pickle_by_value(mod)
    try:
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            cloudpickle.dump(predict_fn, f)
        with open(tmp_path, "rb") as f:
            pickle.load(f)              # reload check before promoting
        os.replace(tmp_path, out_path)
    finally:
        for mod in mods:
            try:
                cloudpickle.unregister_pickle_by_value(mod)
            except Exception:
                pass
    return out_path
