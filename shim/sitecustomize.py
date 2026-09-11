"""Installed on PYTHONPATH by env.sh; Python imports it at startup.

This is how the deformable-aggregation shim reaches processes we do not launch
ourselves -- navsim's hydra entrypoints fan work out to ray workers, which are
fresh interpreters that never import our `reference` package. Without this,
`run_dataset_caching.py` dies inside a worker with

    ImportError: cannot import name 'deformable_aggregation_ext'

The hook is lazy (see reference/__init__.py), so this costs an import of
`reference` and nothing else -- no torch -- for processes that never touch the
extension. It is a no-op outside env.sh, since SPARSEDRIVE_TT_ROOT gates it.
"""

import os
import sys

_root = os.environ.get("SPARSEDRIVE_TT_ROOT")
if _root:
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        import reference
        reference.install()
    except Exception as exc:                                  # never break startup
        print(f"sitecustomize: SparseDrive shim not installed: {exc}",
              file=sys.stderr)
