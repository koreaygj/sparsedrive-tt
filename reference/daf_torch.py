"""Pure-PyTorch stand-in for the deformable-aggregation CUDA extension.

Why this exists: the `sparse` env ships torch 2.0.1+cu117, whose fat binary stops
at sm_86, and this box has an RTX 5060 Ti (sm_120). Every CUDA kernel raises
"no kernel image is available", so the upstream extension under
navsim/agents/sparsedrive/ops/src cannot run here even if it were built. The
PyTorch reference therefore runs on CPU, and this module supplies the op.

It is also the executable spec for the TT-NN port: the semantics below are read
off deformable_aggregation_cuda.cu, not guessed.

  mc_ms_feat        (B, C, F, E)      F = sum_s H_s*W_s, cams share the level list
  spatial_shape     (S, 2) int        [H, W] per level
  scale_start_index (S,)  int         offset of each level inside F
  sampling_location (B, P, C, 2)      normalised (x, y) in [0, 1]
  weights           (B, P, C, S, G)   softmaxed over (C, S, P) per group
  ->                (B, P, E)         E = G * group_dims

Two details that are easy to get wrong:

1. Bounds. The kernel skips a (point, camera) pair entirely unless
   0 < x < 1 and 0 < y < 1, strictly. That is not what grid_sample does on its
   own -- grid_sample would still blend the in-bounds corners of a point just
   outside. So the mask is applied explicitly, on the weights.
2. Pixel centres. h_im = y*H - 0.5 and w_im = x*W - 0.5 with zero padding is
   exactly grid_sample(align_corners=False, padding_mode="zeros") fed
   grid = 2*loc - 1.
"""

import torch
import torch.nn.functional as F

# The L0 path branch runs P = 1024 anchors x 500 keypoints = 512,000 points.
# Materialising (B*C, E, P) at fp32 for that P is ~1.6 GB per level, so walk P
# in slices. Nothing about the result depends on the slice size.
DEFAULT_CHUNK = 1 << 16


def deformable_aggregation_forward(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
    chunk=DEFAULT_CHUNK,
):
    B, C, _, E = mc_ms_feat.shape
    P = sampling_location.shape[1]
    G = weights.shape[-1]
    group_dims = E // G

    spatial_shape = spatial_shape.to(torch.int64).tolist()
    scale_start_index = scale_start_index.to(torch.int64).tolist()

    out = mc_ms_feat.new_zeros(B, P, E)

    for lo in range(0, P, chunk):
        hi = min(lo + chunk, P)
        n = hi - lo

        loc = sampling_location[:, lo:hi]                      # (B, n, C, 2)
        in_bounds = (
            (loc[..., 0] > 0) & (loc[..., 0] < 1)
            & (loc[..., 1] > 0) & (loc[..., 1] < 1)
        )                                                       # (B, n, C)
        grid = (loc * 2 - 1).permute(0, 2, 1, 3).reshape(B * C, n, 1, 2)
        mask = in_bounds.permute(0, 2, 1).reshape(B * C, n, 1)  # (B*C, n, 1)

        acc = mc_ms_feat.new_zeros(B * C, n, E)
        for s, ((h, w), start) in enumerate(zip(spatial_shape, scale_start_index)):
            level = mc_ms_feat[:, :, start:start + h * w, :]
            level = level.permute(0, 1, 3, 2).reshape(B * C, E, h, w)

            sampled = F.grid_sample(
                level, grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )                                                   # (B*C, E, n, 1)
            sampled = sampled.squeeze(-1).permute(0, 2, 1)      # (B*C, n, E)

            w_s = weights[:, lo:hi, :, s, :]                    # (B, n, C, G)
            w_s = w_s.permute(0, 2, 1, 3).reshape(B * C, n, G)
            w_s = (w_s * mask).repeat_interleave(group_dims, dim=-1)

            acc += sampled * w_s

        out[:, lo:hi] = acc.reshape(B, C, n, E).sum(1)

    return out


def deformable_aggregation_backward(*args, **kwargs):
    raise NotImplementedError(
        "CPU reference is inference-only; training needs the CUDA extension "
        "on a GPU this torch build supports."
    )
