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
        self.img_buf = self.st_buf = self.pj_buf = None
        self.trace_id = self.trace_out = None
        self._dfas = [d for L in self.decoder.layers
                      for d in (L.p_dfa, getattr(L, "t_dfa", None)) if d is not None]

    def _buf(self, shape, dtype, layout):
        return ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(ttnn.Shape(shape), dtype, layout, ttnn.BufferType.DRAM),
            self.dev)

    def prepare(self, imgs, status_feature, proj, iwh):
        """Every host write for one frame, into buffers allocated once.

        Trace capture refuses host writes, so a traceable frame has to read its
        inputs from device tensors whose addresses do not move. This fills
        them; forward() below touches nothing but the device.
        """
        n4 = self.C * self.H * self.W
        if self.img_buf is None:
            self.img_buf = self._buf([1, 1, n4 * 4 // 256, 256], ttnn.bfloat16,
                                     ttnn.ROW_MAJOR_LAYOUT)
            self.st_buf = self._buf([1, status_feature.numel()], ttnn.bfloat16,
                                    ttnn.TILE_LAYOUT)
            self.pj_buf = self._buf([self.C, 12], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        x = torch.nn.functional.pad(imgs.permute(0, 2, 3, 1), (0, 1))
        for host, devt in (
                (x.reshape(1, 1, n4 * 4 // 256, 256).bfloat16().contiguous(), self.img_buf),
                (status_feature.reshape(1, -1).bfloat16().contiguous(), self.st_buf),
                (proj[:, :3].reshape(self.C, -1).bfloat16().contiguous(), self.pj_buf)):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=devt.dtype, layout=devt.layout,
                                mesh_mapper=_mapper(self.dev)[1]), devt)
        for dfa in self._dfas:
            dfa.prepare_proj(proj, iwh)

    def features(self):
        """-> per-level ttnn NHWC + the image tokens, from self.img_buf."""
        n4 = self.C * self.H * self.W
        xt = ttnn.reshape(self.img_buf, (1, 1, n4, 4))
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

    def read(self, win):
        """The device tensor forward() returns -> [poses, 3] on the host."""
        return to_host(win, self.dev).float()[0].reshape(self.poses, 3)

    def forward(self):
        """One frame, device only: no host reads, no host writes, so it can be
        captured with ttnn.begin_trace_capture. Returns a device tensor."""
        levels, img_tt, n_img = self.features()
        status = ttnn.linear(self.st_buf, self.w_st, bias=self.b_st,
                             compute_kernel_config=self.cfg)
        pe = self.path_pos(self.pv_tt)
        ve = self.vel_pos(self.vv_tt)
        return self.decoder(pe, ve, self.anchor_tt, status, levels, img_tt, n_img,
                            self.pj_buf, self.tv_all, self.tv_xy,
                            self.p_abs0, self.v_abs0, self.n_vel, self.poses)

    def capture(self, imgs, status_feature, proj, iwh):
        """Record one frame as a trace, so a replay is ONE dispatch.

        Not for speed: the frame is compute-bound and a replay measured the
        same as the eager path, 126.1 ms against 125.9. It is for the hang.
        Eager, a frame pushes several hundred programs through the command
        queue -- and that queue reaches the second chip over the ethernet
        tunnel, where the dispatch kernels wait on each other's credits. Every
        hang this port has seen has been those kernels stuck in TAPW/DAPW, at
        50 to 7700 tokens apart. A trace turns hundreds of chances per frame
        into one.

        Warm up before calling this: the program cache has to be populated,
        and nothing may allocate a device buffer between capture and replay or
        the recorded addresses stop meaning what they meant.
        """
        self.prepare(imgs, status_feature, proj, iwh)
        self.forward()                              # warm the cache
        self.prepare(imgs, status_feature, proj, iwh)
        self.trace_id = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self.trace_out = self.forward()
        ttnn.end_trace_capture(self.dev, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.dev)
        return self

    def replay(self, imgs, status_feature, proj, iwh):
        self.prepare(imgs, status_feature, proj, iwh)
        ttnn.execute_trace(self.dev, self.trace_id, cq_id=0, blocking=True)
        return self.read(self.trace_out)

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.dev, self.trace_id)
            self.trace_id = None

    def __call__(self, imgs, status_feature, proj, iwh):
        if self.trace_id is not None:
            return self.replay(imgs, status_feature, proj, iwh)
        self.prepare(imgs, status_feature, proj, iwh)
        return self.read(self.forward())
