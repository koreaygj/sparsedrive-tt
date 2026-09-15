"""Mesh helpers.

The DFA shards anchors across the mesh (see dfa.py). Everything else
replicates: the backbone sees 3 cameras, which do not divide by 2, and the
decoder's attention is cheap next to the DFA. A replicated op runs redundantly
on both chips -- no speedup, no loss, and no special cases at the boundaries.

A replicated tensor composed with ConcatMeshToTensor(dim=0) comes back with
both copies stacked. Callers already slice `[:T]` for tile padding, and that
same slice takes the first copy.
"""

import os
import pathlib

import ttnn


_FABRIC = False


def enable_fabric():
    """Turn on the 1D fabric, which is what makes ttnn.all_gather work.

    It is DISABLED by default, and an all_gather on a mesh without it does not
    fail -- it hangs the device until tt-smi -r. Call this BEFORE opening the
    mesh device; setting it afterwards does not reach an open device.

    Points TT_MESH_GRAPH_DESC_PATH at the stock n300 descriptor under
    TT_METAL_HOME when it is not already set, and refuses if neither is
    available -- the fabric is worse than useless without one. See the message.

    Set it in the SHELL if ttnn has already built its runtime options by the
    time this runs; the check below says which happened.

    Measured no cost: the compute grid stays 8x8 on both chips and a frame runs
    the same. What it buys is the anchor gather at the DFA's output, 3.25 ms of
    host round trip at 1024 anchors against 0.17 on device.
    """
    global _FABRIC
    if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        stock = (pathlib.Path(os.environ.get("TT_METAL_HOME", "")) / "tt_metal" /
                 "fabric" / "mesh_graph_descriptors" /
                 "n300_mesh_graph_descriptor.textproto")
        if stock.is_file():
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(stock)
    if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        raise RuntimeError(
            "TT_MESH_GRAPH_DESC_PATH is not set. Enabling the fabric without it "
            "kills the device: tt-metal only loads the stock mesh graph "
            "descriptor when there is more than one host rank, so a single "
            "process falls back to auto-discovery, auto-discovery does not know "
            "this motherboard, and the E/W routing planes never get registered. "
            "all_gather then logs 'Failed to discover available ethernet links' "
            "and the ethernet cores' mailboxes go bad a few hundred frames "
            "later. Point it at "
            "$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/"
            "n300_mesh_graph_descriptor.textproto")
    if hasattr(ttnn, "set_fabric_config"):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        _FABRIC = True


def fabric_on():
    """Did THIS process enable the fabric before opening its device?

    Deliberately a process-local flag and NOT ttnn.get_fabric_config(). That
    reports a setting which outlives the process: a fresh run which never
    called enable_fabric() still read FABRIC_1D, called all_gather on a device
    with no fabric, and hung until tt-smi -r. A wrong answer here is not an
    exception, it is a dead device, so the guard has to be something this
    process set itself.
    """
    return _FABRIC


def mesh_of(device):
    """-> (num_devices, replicate mapper or None, dim-0 composer or None)."""
    nd = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    if nd == 1:
        return 1, None, None
    return nd, ttnn.ReplicateTensorToMesh(device), ttnn.ConcatMeshToTensor(device, dim=0)
