"""PyTorch reference helpers for the SparseDriveV2 -> TT-NN port.

`navsim/agents/sparsedrive/ops/__init__.py` imports the compiled deformable
aggregation extension at module scope, and that extension is neither built nor
needed here (see daf_torch.py). `install()` supplies a pure-PyTorch stand-in.

It registers a meta-path finder rather than stuffing sys.modules directly, for
two reasons:

  - laziness. sitecustomize.py calls this at interpreter startup for every
    process under env.sh. Building the module eagerly would import torch into
    every `python -c`, ray worker bootstrap and hydra config parse. The finder
    imports torch only if something actually reaches for the extension.
  - reach. navsim's entrypoints fan out over ray workers, which are fresh
    interpreters that never import this package. They pick the shim up from
    sitecustomize, so the hook has to work from a cold start.
"""

import importlib.abc
import importlib.util
import sys
import types

_BASE = "navsim.agents.sparsedrive.ops"
_EXT = f"{_BASE}.deformable_aggregation_ext"
_EXT_DEPTH = f"{_BASE}.deformable_aggregation_with_depth_ext"


def _build(fullname):
    mod = types.ModuleType(fullname)
    if fullname == _EXT:
        from . import daf_torch
        mod.deformable_aggregation_forward = daf_torch.deformable_aggregation_forward
        mod.deformable_aggregation_backward = daf_torch.deformable_aggregation_backward
    else:
        # SparseDriveV2 always calls DAF with depth_prob=None, so the depth
        # variant is imported and never used. Fail loudly if that changes.
        def _no_depth(*_a, **_k):
            raise NotImplementedError(
                "depth-conditioned deformable aggregation has no PyTorch "
                "reference; SparseDriveV2 calls DAF with depth_prob=None"
            )

        mod.deformable_aggregation_with_depth_forward = _no_depth
        mod.deformable_aggregation_with_depth_backward = _no_depth
    mod.__spec__ = importlib.util.spec_from_loader(fullname, _Loader())
    return mod


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return _build(spec.name)

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    targets = (_EXT, _EXT_DEPTH)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.targets:
            return importlib.util.spec_from_loader(fullname, _Loader())
        return None


def install():
    """Idempotent; safe to call from sitecustomize and from scripts."""
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return
    # Ahead of the normal finders, which would find nothing and raise.
    sys.meta_path.insert(0, _Finder())
