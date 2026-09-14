"""Phase 2: decoder attention and FFN against golden."""
import os, pathlib, sys, time
import torch, ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.attention import TtMultiheadAttention, TtFFN, _t   # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
PRE = "agent._sparsedrive_model._trajectory_head.decoder.layers.0."


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    frame = sorted((EXP / "golden").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    D = "_trajectory_head.decoder.layers.0."
    print(f"  frame {frame.name}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    ok = True
    try:
        for name, T in (("p_attention", 1024), ("v_attention", 256)):
            x = d[D + name + ".in[0]"][0]
            ref = d[D + name + ".out[0]"][0]
            mha = TtMultiheadAttention(sd, PRE + name + ".", dev)
            xt = _t(x, dev)
            mha(xt, tq=T)
            ttnn.synchronize_device(dev); t0 = time.time()
            got = ttnn.to_torch(mha(xt, tq=T)).float()[:T]
            ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3
            p = pcc(got, ref); ok &= p >= 0.99
            print(f"  {name:16s} T={T:5d}  {dt:6.2f} ms  PCC {p:.6f}"
                  f"  max|d| {(got - ref).abs().max():.3e}")

        # cross-attention: velocity queries against the last FPN level,
        # flattened to 3 cams x 8x16 = 384 image tokens. q comes from a
        # different tensor than k and v, so in_proj has to be split.
        for name, Tq, Tk in (("v_img_attention", 256, 384),):
            q = d[D + name + ".in[0]"][0]
            kv = d[D + name + ".in[1]"][0]
            ref = d[D + name + ".out[0]"][0]
            mha = TtMultiheadAttention(sd, PRE + name + ".", dev)
            qt, kt = _t(q, dev), _t(kv, dev)
            mha(qt, kt, kt, tq=Tq, tk=Tk)
            ttnn.synchronize_device(dev); t0 = time.time()
            got = ttnn.to_torch(mha(qt, kt, kt, tq=Tq, tk=Tk)).float()[:Tq]
            ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3
            pv = pcc(got, ref); ok &= pv >= 0.99
            print(f"  {name:16s} {Tq}x{Tk:<4d} {dt:6.2f} ms  PCC {pv:.6f}"
                  f"  max|d| {(got - ref).abs().max():.3e}")

        for name in ("p_ffn", "v_ffn"):
            x = d[D + name + ".in[0]"][0]
            ref = d[D + name + ".out"][0]
            ffn = TtFFN(sd, PRE + name + ".", dev)
            xt = _t(x, dev)
            ffn(xt)
            ttnn.synchronize_device(dev); t0 = time.time()
            got = ttnn.to_torch(ffn(xt)).float()[:x.shape[0]]
            ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3
            p = pcc(got, ref); ok &= p >= 0.99
            print(f"  {name:16s} T={x.shape[0]:5d}  {dt:6.2f} ms  PCC {p:.6f}"
                  f"  max|d| {(got - ref).abs().max():.3e}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
