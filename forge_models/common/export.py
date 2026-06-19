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

    ``extra_by_value_modules`` lets a topic script register additional shared
    modules whose functions its predict closure calls (rarely needed — the
    feature helpers are registered automatically).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_value = [_features, _config, *extra_by_value_modules]
    for mod in by_value:
        cloudpickle.register_pickle_by_value(mod)
    try:
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            cloudpickle.dump(predict_fn, f)
        with open(tmp_path, "rb") as f:
            pickle.load(f)              # reload check before promoting
        os.replace(tmp_path, out_path)
    finally:
        for mod in by_value:
            cloudpickle.unregister_pickle_by_value(mod)
    return out_path
