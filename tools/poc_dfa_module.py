"""Phase 3 step 1: model/dfa.py against all three DFA calls of a frame.

Only DAF[0] had been checked until now. A frame makes three calls and they do
not have the same shape:

    layer 0 path   1024 x 50 ->  500 keypoints, clp 6000
    layer 1 path    128 x 50 ->  500,           clp 6000
    layer 1 traj    400 x  8 ->   80,           clp  960
"""
import os, pathlib, sys, time
import torch, ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.dfa import TtDFA, legal_splits   # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
SD = "agent._sparsedrive_model._trajectory_head.decoder.layers."
GD = "_trajectory_head.decoder.layers."

CALLS = [("0.p_deform_model.", 50, 1024), ("1.p_deform_model.", 50, 128),
         ("1.t_deform_model.", 8, 400)]


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    frame = sorted((EXP / "golden_dfa").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    mc = d["DAF[0].mc_ms_feat"]
    shapes = [tuple(int(v) for v in r) for r in d["DAF[0].spatial_shape"]]
    starts = [int(v) for v in d["DAF[0].scale_start_index"]]
    proj = d[GD + "0.p_deform_model.in[4].projection_mat"][0]
    iwh = d[GD + "0.p_deform_model.in[4].image_wh"][0]
    print(f"  frame {frame.name}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    ok = True
    try:
        levels = None
        for key, ns, n in CALLS:
            dfa = TtDFA(sd, SD + key, dev, num_sample=ns)
            if levels is None:
                levels = dfa.upload_levels(mc, shapes, starts)
            feat = d[GD + key + "in[0]"][0][:n]
            anchor = d[GD + key + "in[1]"][0][:n]
            ref = d[GD + key + "out"][0][:n]
            CH = 128
            proj_tt = dfa.T(proj[:, :3].reshape(dfa.C, -1))
            f_tt = dfa.T(feat)
            a_tt = dfa.T(anchor, ttnn.float32)
            dfa(f_tt, a_tt, levels, proj, proj_tt, iwh, chunk=CH)
            ttnn.synchronize_device(dev); t0 = time.time()
            got_tt = dfa(f_tt, a_tt, levels, proj, proj_tt, iwh, chunk=CH)
            ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3
            got = dfa.G2T(got_tt).float()[:n]
            p = pcc(got, ref); ok &= p >= 0.999
            print(f"  {key[:-1]:20s} n={n:5d} P={dfa.P:3d} clp={dfa.clp:5d}"
                  f"  {dt:8.1f} ms  PCC {p:.6f}")
            print(f"        legal splits {legal_splits(dfa.clp)[:8]}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
