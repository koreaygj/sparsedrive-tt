"""ResNet-34 backbone on TT-NN.

Same layer counts and planes as the ResNet-50 in sparse4D-tt ([3,4,6,3],
[64,128,256,512]) -- only the block differs. BasicBlock is two 3x3 convs with
no expansion, so every stage carries a quarter of the channels ResNet-50 does
and the L1 pressure that forced BLOCK_SHARDED there is much lower here.

BatchNorm is folded into the preceding convolution at preprocessing time. At
inference BN is a fixed affine map, so

    BN(Conv(x)) = Conv_folded(x)

with w' = w * gamma/sqrt(var+eps) and b' = (b - mean) * gamma/sqrt(var+eps) + beta.
That removes 36 ops from the graph and costs nothing in accuracy.
"""

from typing import Tuple

import torch
import os

import ttnn

LAYERS = [3, 4, 6, 3]
PLANES = [64, 128, 256, 512]
STRIDES = [1, 2, 2, 2]


def fold_bn_into_conv(w, gamma, beta, mean, var, eps=1e-5, bias=None):
    scale = gamma / torch.sqrt(var + eps)
    w_f = w * scale.view(-1, 1, 1, 1)
    b_f = (torch.zeros_like(mean) if bias is None else bias)
    b_f = (b_f - mean) * scale + beta
    return w_f, b_f


def preprocess(sd: dict, prefix: str, device):
    """state_dict -> {name: {weight, bias}} with BN folded and on device."""
    def g(k):
        return sd[prefix + k].float()

    def put(w, b):
        return {
            "weight": ttnn.from_torch(w, dtype=ttnn.bfloat16),
            "bias": ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
        }

    out = {}
    w, b = fold_bn_into_conv(g("conv1.weight"), g("bn1.weight"), g("bn1.bias"),
                             g("bn1.running_mean"), g("bn1.running_var"))
    # conv2d wants a channel count that tiles; pad RGB to 4
    out["conv1"] = put(torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 1)), b)

    for li, nb in enumerate(LAYERS, start=1):
        for bi in range(nb):
            p = f"layer{li}.{bi}."
            for ci in (1, 2):
                w, b = fold_bn_into_conv(
                    g(f"{p}conv{ci}.weight"), g(f"{p}bn{ci}.weight"),
                    g(f"{p}bn{ci}.bias"), g(f"{p}bn{ci}.running_mean"),
                    g(f"{p}bn{ci}.running_var"))
                out[f"{p}conv{ci}"] = put(w, b)
            if prefix + p + "downsample.0.weight" in sd:
                w, b = fold_bn_into_conv(
                    g(f"{p}downsample.0.weight"), g(f"{p}downsample.1.weight"),
                    g(f"{p}downsample.1.bias"), g(f"{p}downsample.1.running_mean"),
                    g(f"{p}downsample.1.running_var"))
                out[f"{p}downsample"] = put(w, b)
    return out


class Conv2dOp:
    """ttnn.conv2d with its weights and shapes bound once."""

    def __init__(self, params, device, in_channels, out_channels, kernel_size,
                 stride, padding, batch_size, input_height, input_width,
                 activation=None, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                 deallocate_activation=False, act_block_h_override=0,
                 math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True,
                 slice_l1=True, core_grid=None):
        self.device, self.in_channels, self.out_channels = device, in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.batch_size, self.input_height, self.input_width = batch_size, input_height, input_width
        self.weight, self.bias = params["weight"], params["bias"]
        if os.environ.get("TT_CONV_NO_FP32_ACC") == "1":
            fp32_dest_acc_en = False
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en, packer_l1_acc=False,
            math_approx_mode=False)
        if os.environ.get("TT_CONV_DST_FULL_SYNC") == "1":
            self.compute_config.dst_full_sync_en = True
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16, shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            reshard_if_not_optimal=True, activation=activation)
        if act_block_h_override:
            self.conv_config.act_block_h_override = act_block_h_override
        if core_grid is not None:
            self.conv_config.core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(core_grid[0] - 1, core_grid[1] - 1))])
            self.conv_config.override_sharding_config = True
        self.slice_config = ttnn.Conv2dL1FullSliceConfig if slice_l1 else None

    def __call__(self, x):
        kw = dict(input_tensor=x, weight_tensor=self.weight, bias_tensor=self.bias,
                  device=self.device, in_channels=self.in_channels,
                  out_channels=self.out_channels, input_height=self.input_height,
                  input_width=self.input_width, batch_size=self.batch_size,
                  kernel_size=self.kernel_size, stride=self.stride,
                  padding=self.padding, groups=1, conv_config=self.conv_config,
                  compute_config=self.compute_config, return_output_dim=True,
                  return_weights_and_bias=True)
        if self.slice_config is not None:
            kw["slice_config"] = self.slice_config
        [x, [h, w], [self.weight, self.bias]] = ttnn.conv2d(**kw)
        return x, h, w


class BasicBlock:
    """conv3x3 -> conv3x3 -> add identity -> relu. Stride sits on conv1."""

    expansion = 1

    def __init__(self, params, prefix, device, inplanes, planes, stride,
                 batch_size, in_h, in_w, shard):
        self.stride = stride
        self.shard = shard
        relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        self.conv1 = Conv2dOp(params[prefix + "conv1"], device, inplanes, planes,
                              (3, 3), (stride, stride), (1, 1), batch_size,
                              in_h, in_w, activation=relu, shard_layout=shard)
        out_h, out_w = in_h // stride, in_w // stride
        self.conv2 = Conv2dOp(params[prefix + "conv2"], device, planes, planes,
                              (3, 3), (1, 1), (1, 1), batch_size, out_h, out_w,
                              shard_layout=shard)
        key = prefix + "downsample"
        self.downsample = None
        if key in params:
            # Height sharded like the main path. shard_layout=None resolves to
            # INTERLEAVED, which conv2d rejects outright -- it takes Height,
            # Block or Width only.
            self.downsample = Conv2dOp(params[key], device, inplanes, planes,
                                       (1, 1), (stride, stride), (0, 0),
                                       batch_size, in_h, in_w, shard_layout=shard)

    def __call__(self, x):
        identity = x
        out, h, w = self.conv1(x)
        out, h, w = self.conv2(out)
        if self.downsample is not None:
            identity, _, _ = self.downsample(x)
        identity = ttnn.to_memory_config(identity, out.memory_config()) \
            if identity.memory_config() != out.memory_config() else identity
        out = ttnn.add(out, identity)
        return ttnn.relu(out), h, w


class TtResNet34:
    def __init__(self, params, device, batch_size=3, in_h=256, in_w=512,
                 out_indices=(0, 1, 2, 3)):
        self.device, self.batch_size = device, batch_size
        self.in_h, self.in_w, self.out_indices = in_h, in_w, out_indices
        relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        self.conv1 = Conv2dOp(params["conv1"], device, 4, 64, (7, 7), (2, 2), (3, 3),
                              batch_size, in_h, in_w, activation=relu,
                              act_block_h_override=64, deallocate_activation=True,
                              slice_l1=False)
        self.pool_h, self.pool_w = in_h // 4, in_w // 4
        self.layers, inplanes = [], 64
        h, w = self.pool_h, self.pool_w
        for i in range(4):
            planes, stride, nb = PLANES[i], STRIDES[i], LAYERS[i]
            # Height sharding holds to layer3. At layer4 the spatial map is
            # 8x16 and the channels are 512, so each core gets a handful of
            # rows carrying every channel's weights, and the circular buffers
            # grow to 1.82 MB against L1's 1.5 -- measured, not guessed.
            # Block sharding splits the channel axis as well.
            #
            # ResNet-34 carries a quarter of ResNet-50's channels, so the
            # switch lands one stage later than it does in sparse4D-tt, which
            # needs it from layer3.
            shard = (ttnn.TensorMemoryLayout.BLOCK_SHARDED if i == 3
                     else ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            blocks = []
            for b in range(nb):
                s = stride if b == 0 else 1
                blocks.append(BasicBlock(params, f"layer{i+1}.{b}.", device,
                                         inplanes, planes, s, batch_size, h, w, shard))
                if b == 0:
                    h, w = h // s, w // s
                inplanes = planes
            self.layers.append(blocks)

    def __call__(self, x):
        x, h, w = self.conv1(x)
        x = ttnn.max_pool2d(x, batch_size=self.batch_size, input_h=h, input_w=w,
                            channels=64, kernel_size=[3, 3], stride=[2, 2],
                            padding=[1, 1], dilation=[1, 1])
        outs = []
        last = max(self.out_indices)
        for i, blocks in enumerate(self.layers):
            for blk in blocks:
                x, h, w = blk(x)
            if i in self.out_indices:
                outs.append((x, h, w, PLANES[i]))
            if i == last:                      # stop early when testing a prefix
                break
        return outs
