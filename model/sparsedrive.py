"""SparseDriveV2 end to end: images in, one trajectory out.

    imgs [cams,3,256,512]
      -> ResNet-34 + FPN          four levels, 256 channels each
      -> decoder                  two layers, path/velocity/trajectory
      -> trajectory [poses, 3]

The vocabularies are buffers in the checkpoint, so nothing is read from the
kmeans files at inference; path_pos_embed and vel_pos_embed turn them into the
queries the decoder starts from.
"""

import pathlib
import sys

import torch
import ttnn

from .attention import TtFFN, _t, HIFI
from .decoder import TtDecoder
from .fpn import TtFPN, preprocess as pre_fpn
from .resnet34 import TtResNet34, preprocess as pre_res

M = "agent._sparsedrive_model."


class TtSparseDrive:
    def __init__(self, sd, device, batch_cams=3, in_h=256, in_w=512):
        self.dev, self.C, self.H, self.W = device, batch_cams, in_h, in_w
        self.sizes = [(in_h // 4, in_w // 4), (in_h // 8, in_w // 8),
                      (in_h // 16, in_w // 16), (in_h // 32, in_w // 32)]
        self.backbone = TtResNet34(pre_res(sd, M + "_backbone.img_backbone.", device),
                                   device, batch_size=batch_cams, in_h=in_h, in_w=in_w)
        self.fpn = TtFPN(pre_fpn(sd, M + "_backbone.img_neck."), device,
                         batch_cams, self.sizes)
        T = M + "_trajectory_head."
        self.path_pos = TtFFN(sd, T + "path_pos_embed.", device)
        self.vel_pos = TtFFN(sd, T + "vel_pos_embed.", device)
        self.w_st = _t(sd[M + "_status_encoding.weight"].float().t(), device)
        self.b_st = _t(sd[M + "_status_encoding.bias"].float().reshape(1, -1), device)
        self.cfg = ttnn.WormholeComputeKernelConfig(**HIFI)
        self.path_vocab = sd[T + "path_vocab"].float()
        self.vel_vocab = sd[T + "vel_vocab"].float()
        self.traj_vocab = sd[T + "traj_vocab"].float()
        self.decoder = TtDecoder(sd, T + "decoder.", device)

    def features(self, imgs):
        """imgs [cams, 3, H, W] -> per-level ttnn NHWC + the image tokens."""
        x = torch.nn.functional.pad(imgs.permute(0, 2, 3, 1), (0, 1))
        xt = ttnn.from_torch(x.reshape(1, 1, self.C * self.H * self.W, 4).contiguous(),
                             dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=self.dev)
        outs = self.fpn(self.backbone(xt))
        levels, last = [], None
        for (t, h, w, c) in outs:
            v = ttnn.to_torch(t).float().reshape(self.C, h, w, c)
            levels.append(ttnn.from_torch(v.contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT,
                                          device=self.dev, dtype=ttnn.bfloat16))
            last = v
        # image tokens for the velocity branch's cross-attention: the coarsest
        # level, cams x h x w flattened
        return levels, last.reshape(-1, last.shape[-1])

    def __call__(self, imgs, status_feature, proj, iwh):
        levels, img_value = self.features(imgs)
        status = ttnn.to_torch(ttnn.linear(_t(status_feature.reshape(1, -1), self.dev),
                                           self.w_st, bias=self.b_st,
                                           compute_kernel_config=self.cfg)).float()[0, :256]
        n_path = self.path_vocab.shape[0]
        pe = ttnn.to_torch(self.path_pos(
            _t(self.path_vocab.reshape(n_path, -1), self.dev))).float()[:n_path]
        n_vel = self.vel_vocab.shape[0]
        ve = ttnn.to_torch(self.vel_pos(
            _t(self.vel_vocab, self.dev))).float()[:n_vel]
        return self.decoder(pe, ve, self.path_vocab, self.traj_vocab, status,
                            levels, img_value, proj, iwh)
