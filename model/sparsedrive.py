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

from .attention import TtFFN, _t, to_host, HIFI
from .decoder import TtDecoder
from .attention import _mapper
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
        n_path, n_vel = self.path_vocab.shape[0], self.vel_vocab.shape[0]
        self.pv_tt = _t(self.path_vocab.reshape(n_path, -1), device)
        self.vv_tt = _t(self.vel_vocab, device)
        self.anchor_tt = _t(self.path_vocab[..., :2].reshape(n_path, -1), device,
                            dtype=ttnn.float32)
        tv = self.traj_vocab
        self.poses = tv.shape[2]
        self.tv_all = _t(tv.reshape(n_path * n_vel, -1), device, dtype=ttnn.float32)
        self.tv_xy = _t(tv[..., :2].reshape(n_path * n_vel, -1), device,
                        dtype=ttnn.float32)
        self.p_abs0 = _t(torch.arange(n_path).float().reshape(-1, 1), device,
                         dtype=ttnn.float32)
        self.v_abs0 = _t(torch.arange(n_vel).float().reshape(-1, 1), device,
                         dtype=ttnn.float32)
        self.n_vel = n_vel

    def features(self, imgs):
        """imgs [cams, 3, H, W] -> per-level ttnn NHWC + the image tokens."""
        x = torch.nn.functional.pad(imgs.permute(0, 2, 3, 1), (0, 1))
        n4 = self.C * self.H * self.W
        xt = ttnn.reshape(
            ttnn.from_torch(x.reshape(1, 1, n4 * 4 // 256, 256).bfloat16().contiguous(),
                            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.dev, mesh_mapper=_mapper(self.dev)[1]),
            (1, 1, n4, 4))
        outs = self.fpn(self.backbone(xt))
        levels, last = [], None
        for (t, h, w, c) in outs:
            v = t
            if v.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                v = ttnn.sharded_to_interleaved(v, ttnn.DRAM_MEMORY_CONFIG)
            levels.append(ttnn.reshape(
                ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT), (self.C, h, w, c)))
            last = (v, h, w, c)
        v, h, w, c = last
        return levels, ttnn.reshape(v, (self.C * h * w, c)), self.C * h * w

    def __call__(self, imgs, status_feature, proj, iwh):
        levels, img_tt, n_img = self.features(imgs)
        status = ttnn.linear(_t(status_feature.reshape(1, -1), self.dev),
                             self.w_st, bias=self.b_st,
                             compute_kernel_config=self.cfg)
        pe = self.path_pos(self.pv_tt)
        ve = self.vel_pos(self.vv_tt)
        proj_tt = _t(proj[:, :3].reshape(self.C, -1), self.dev)
        return self.decoder(pe, ve, self.anchor_tt, status, levels, img_tt, n_img,
                            proj, proj_tt, iwh,
                            self.tv_all, self.tv_xy, self.p_abs0, self.v_abs0,
                            self.n_vel, self.poses)
