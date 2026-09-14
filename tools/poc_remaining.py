"""Phase 3 step 2: everything left unverified before the decoder can be wired.

Six LayerNorms, two vocabulary embeddings, the status encoder, three score MLPs.

The norms were never hooked in dump_golden.py, but they do not need to be:
the dump stores module inputs *and* outputs, and every norm sits between two
modules that were hooked, so its input is a sum of tensors already on disk.

    p_norm1( p_deform_model.out + p_attention.out[0] )  ==  p_ffn.in[0]
    p_norm2( p_ffn.in[0]        + p_ffn.out         )  ==  path_mlp.in[0]

which checks the residual add along with the norm, end to end, rather than
checking ttnn.layer_norm against torch on synthetic input. Dropout is identity
at eval, so it drops out of the identity.

    $TT_PY tools/poc_remaining.py
"""

import os
import pathlib
import sys

import torch
import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.attention import TtFFN, _t, HIFI   # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
P = "agent._sparsedrive_model._trajectory_head."
G = "_trajectory_head."

# norm: (params, residual a, residual b, expected output)
NORMS = [
    ("decoder.layers.0.p_norm1", "decoder.layers.0.p_deform_model.out",
     "decoder.layers.0.p_attention.out[0]", "decoder.layers.0.p_ffn.in[0]"),
    ("decoder.layers.0.p_norm2", "decoder.layers.0.p_ffn.in[0]",
     "decoder.layers.0.p_ffn.out", "decoder.layers.0.path_mlp.in[0]"),
    ("decoder.layers.0.v_norm1", "decoder.layers.0.v_attention.in[0]",
     "decoder.layers.0.v_attention.out[0]", "decoder.layers.0.v_ffn.in[0]"),
    ("decoder.layers.0.v_norm2", "decoder.layers.0.v_ffn.in[0]",
     "decoder.layers.0.v_ffn.out", "decoder.layers.0.vel_mlp.in[0]"),
    ("decoder.layers.1.t_norm1", "decoder.layers.1.t_deform_model.out",
     "decoder.layers.1.t_attention.out[0]", "decoder.layers.1.t_ffn.in[0]"),
    ("decoder.layers.1.t_norm2", "decoder.layers.1.t_ffn.in[0]",
     "decoder.layers.1.t_ffn.out", "decoder.layers.1.traj_mlp.in[0]"),
]
MLPS = [("path_pos_embed", "path_pos_embed.in[0]", "path_pos_embed.out"),
        ("vel_pos_embed", "vel_pos_embed.in[0]", "vel_pos_embed.out"),
        ("decoder.layers.0.path_mlp", "decoder.layers.0.path_mlp.in[0]",
         "decoder.layers.0.path_mlp.out"),
        ("decoder.layers.0.vel_mlp", "decoder.layers.0.vel_mlp.in[0]",
         "decoder.layers.0.vel_mlp.out"),
        ("decoder.layers.1.traj_mlp", "decoder.layers.1.traj_mlp.in[0]",
         "decoder.layers.1.traj_mlp.out")]


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    frame = sorted((EXP / "golden").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    print(f"  frame {frame.name}")
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    ok = True
    try:
        cfg = ttnn.WormholeComputeKernelConfig(**HIFI)
        print("  --- LayerNorm (residual 포함, 모듈 경계로 재구성) ---")
        for name, ra, rb, out in NORMS:
            a, b, ref = d[G + ra][0], d[G + rb][0], d[G + out][0]
            w = _t(sd[P + name + ".weight"].float(), dev)
            bi = _t(sd[P + name + ".bias"].float(), dev)
            got = ttnn.to_torch(ttnn.layer_norm(
                ttnn.add(_t(a, dev), _t(b, dev)), weight=w, bias=bi)).float()[:a.shape[0]]
            p = pcc(got, ref); ok &= p >= 0.999
            print(f"      {name.replace('decoder.layers.',''):16s} T={a.shape[0]:4d}"
                  f"  PCC {p:.6f}  max|d| {(got-ref).abs().max():.2e}")

        print("  --- MLP / 임베딩 ---")
        for name, xi, out in MLPS:
            x, ref = d[G + xi][0], d[G + out][0]
            mlp = TtFFN(sd, P + name + ".", dev)
            got = ttnn.to_torch(mlp(_t(x, dev))).float()[:x.shape[0], :ref.shape[-1]]
            p = pcc(got, ref); ok &= p >= 0.999
            print(f"      {name.replace('decoder.layers.',''):16s} "
                  f"{tuple(x.shape)} -> {tuple(ref.shape)}  PCC {p:.6f}")

        print("  --- status encoding (Linear) ---")
        # _status_encoding hangs off SparseDriveModel, not TrajectoryHead, so
        # it carries no _trajectory_head. prefix in either the dump or the
        # checkpoint.
        x, ref = d["_status_encoding.in[0]"], d["_status_encoding.out"]
        w = _t(sd[P.replace("_trajectory_head.", "") + "_status_encoding.weight"].float().t(), dev)
        bi = _t(sd[P.replace("_trajectory_head.", "") + "_status_encoding.bias"].float().reshape(1, -1), dev)
        got = ttnn.to_torch(ttnn.linear(_t(x, dev), w, bias=bi,
                                        compute_kernel_config=cfg)).float()[:1, :256]
        p = pcc(got, ref); ok &= p >= 0.999
        print(f"      _status_encoding  {tuple(x.shape)} -> {tuple(ref.shape)}  PCC {p:.6f}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
