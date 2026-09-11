"""PyTorch reference helpers for the SparseDriveV2 -> TT-NN port.

`install()` must run before anything imports navsim.agents.sparsedrive, because
navsim/agents/sparsedrive/ops/__init__.py imports the compiled extension at
module scope and that extension cannot be built (let alone run) on this box.
"""

import sys
import types


def install():
    """Register CPU stand-ins for the two deformable-aggregation extensions."""
    from . import daf_torch

    base = "navsim.agents.sparsedrive.ops"

    ext = types.ModuleType(f"{base}.deformable_aggregation_ext")
    ext.deformable_aggregation_forward = daf_torch.deformable_aggregation_forward
    ext.deformable_aggregation_backward = daf_torch.deformable_aggregation_backward
    sys.modules[ext.__name__] = ext

    # SparseDriveV2 never passes depth_prob, so the depth variant is imported
    # but never called. Fail loudly if that ever stops being true.
    def _no_depth(*_a, **_k):
        raise NotImplementedError(
            "depth-conditioned deformable aggregation has no CPU reference; "
            "SparseDriveV2 calls DAF with depth_prob=None"
        )

    depth_ext = types.ModuleType(f"{base}.deformable_aggregation_with_depth_ext")
    depth_ext.deformable_aggregation_with_depth_forward = _no_depth
    depth_ext.deformable_aggregation_with_depth_backward = _no_depth
    sys.modules[depth_ext.__name__] = depth_ext

    return ext, depth_ext
