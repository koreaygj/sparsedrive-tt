"""torchvision FeaturePyramidNetwork on TT-NN.

    inner_blocks[i]  1x1 conv, C_i -> 256      lateral
    layer_blocks[i]  3x3 conv, 256 -> 256      output

No BatchNorm here -- these are plain biased convolutions, so nothing to fold.

Top-down, exactly as torchvision orders it:

    last = inner[-1](x[-1]);  out[-1] = layer[-1](last)
    for i from n-2 down to 0:
        last = inner[i](x[i]) + upsample_nearest(last, to x[i]'s size)
        out[i] = layer[i](last)

The upsample is always a clean 2x here (8x16 -> 16x32 -> 32x64 -> 64x128), so
ttnn.upsample with scale 2 matches F.interpolate(size=..., mode="nearest")
without needing the general resize.
"""

import os

import ttnn

from .resnet34 import Conv2dOp

IN_CHANNELS = [64, 128, 256, 512]
OUT_CHANNELS = 256


def preprocess(sd: dict, prefix: str):
    out = {}
    for i in range(4):
        for kind, blk in (("inner", "inner_blocks"), ("layer", "layer_blocks")):
            w = sd[f"{prefix}{blk}.{i}.0.weight"].float()
            b = sd[f"{prefix}{blk}.{i}.0.bias"].float()
            out[f"{kind}{i}"] = {
                "weight": ttnn.from_torch(w, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            }
    return out


def _grid(level, rows):
    """Pin this conv to a rectangular core grid, or None to let ttnn choose.

    Left to itself, reshard_if_not_optimal picked 62 cores for layer0 -- 24576
    rows split 396/396/... over a grid that is not a rectangle, even though 64
    divides those rows exactly into 12 tiles each. A 1D weights multicast goes
    to a rectangle, so a 62-core set has to be covered in two sends, and every
    hang this port has seen sat in
    reader_writer_tiled_out_1d_mcast_receiver_conv_weights on one worker core of
    this conv, always the same core, waiting on a credit that never came.

    Pinning 8x8 keeps the multicast a single rectangle. It applies only where
    the rows divide evenly into 32-row tiles across 64 cores; the smaller levels
    do not, and they keep the chosen grid.

    Off unless TT_FPN_GRID is set to "<x>x<y>". 8x8 was measured and did NOT
    stop the hang, so a rectangle is not the cure; the knob stays because it
    also picks WHICH cores run the conv, and the same worker core has come up
    in every dump so far. It applies only where the rows divide evenly into
    32-row tiles across that grid.
    """
    spec = os.environ.get("TT_FPN_GRID", "")
    if spec:
        x, y = (int(v) for v in spec.split("x"))
        if rows % (x * y * 32) == 0:
            return (x, y)
    return None


class TtFPN:
    def __init__(self, params, device, batch_size, sizes):
        """sizes: [(h, w)] per level, highest resolution first."""
        self.device, self.batch_size, self.sizes = device, batch_size, sizes
        # Height sharded, and L1 slicing OFF.
        #
        # The slicing is the surprise. Conv2dL1FullSliceConfig reads like a
        # safety net -- "slice if it does not fit" -- and sparse4D-tt enables it
        # by default, but on layer0 (3x3, 256->256, 3x64x128 = 24576 rows) it is
        # what overflows L1. Measured on that conv alone:
        #
        #     batch shard   act_blk_h  slice_l1   result
        #         3 HEIGHT          0      True   L1 1694848 B
        #         3 HEIGHT          0     False   ok
        #         3 BLOCK           0      True   L1
        #         3 BLOCK          32      True   ok
        #
        # With slicing off every combination passes, so it is the one knob that
        # matters here; block sharding and the activation-block override only
        # help while slicing is on. Sharding alone never fixes layer0 -- it
        # fails under Height, Block and Width equally -- which is what sent the
        # first two attempts (block everywhere, then act_block_h_override) into
        # the ground with the error size unchanged to the byte.
        shard = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        self.inner, self.layer = [], []
        for i, (h, w) in enumerate(sizes):
            self.inner.append(Conv2dOp(params[f"inner{i}"], device, IN_CHANNELS[i],
                                       OUT_CHANNELS, (1, 1), (1, 1), (0, 0),
                                       batch_size, h, w, shard_layout=shard,
                                       slice_l1=False))
            self.layer.append(Conv2dOp(params[f"layer{i}"], device, OUT_CHANNELS,
                                       OUT_CHANNELS, (3, 3), (1, 1), (1, 1),
                                       batch_size, h, w, shard_layout=shard,
                                       slice_l1=False, core_grid=_grid(i, batch_size * h * w)))

    def __call__(self, feats):
        """feats: [(tensor, h, w, c)] from the backbone, highest res first."""
        n = len(feats)
        last, _, _ = self.inner[n - 1](feats[n - 1][0])
        outs = [None] * n
        o, h, w = self.layer[n - 1](last)
        outs[n - 1] = (o, h, w, OUT_CHANNELS)
        for i in range(n - 2, -1, -1):
            lat, lh, lw = self.inner[i](feats[i][0])
            ph, pw = self.sizes[i + 1]
            # upsample wants a tile-aligned input, and a conv output sharded in
            # TILE layout is not: its padded shape differs from its logical one.
            # Go through ROW_MAJOR in DRAM, where the [N,H,W,C] view is exact.
            u = ttnn.to_memory_config(ttnn.to_layout(last, ttnn.ROW_MAJOR_LAYOUT),
                                      ttnn.DRAM_MEMORY_CONFIG)
            u = ttnn.reshape(u, (self.batch_size, ph, pw, OUT_CHANNELS))
            up = ttnn.reshape(ttnn.upsample(u, scale_factor=2),
                              (1, 1, self.batch_size * lh * lw, OUT_CHANNELS))
            lat = ttnn.to_memory_config(ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT),
                                        ttnn.DRAM_MEMORY_CONFIG)
            last = ttnn.add(lat, up)
            o, oh, ow = self.layer[i](last)
            outs[i] = (o, oh, ow, OUT_CHANNELS)
        return outs
