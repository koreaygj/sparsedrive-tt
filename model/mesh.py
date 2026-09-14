"""Mesh helpers.

The DFA shards anchors across the mesh (see dfa.py). Everything else
replicates: the backbone sees 3 cameras, which do not divide by 2, and the
decoder's attention is cheap next to the DFA. A replicated op runs redundantly
on both chips -- no speedup, no loss, and no special cases at the boundaries.

A replicated tensor composed with ConcatMeshToTensor(dim=0) comes back with
both copies stacked. Callers already slice `[:T]` for tile padding, and that
same slice takes the first copy.
"""

import ttnn


def mesh_of(device):
    """-> (num_devices, replicate mapper or None, dim-0 composer or None)."""
    nd = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    if nd == 1:
        return 1, None, None
    return nd, ttnn.ReplicateTensorToMesh(device), ttnn.ConcatMeshToTensor(device, dim=0)
