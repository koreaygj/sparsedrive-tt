"""Phase 2: ResNet-34 backbone on device, stage by stage against golden.

    $TT_PY tools/poc_resnet34.py --stages 1     # stem + layer1 only
    $TT_PY tools/poc_resnet34.py                # all four stages
"""
import argparse, os, pathlib, sys, time
import torch, ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.resnet34 import TtResNet34, preprocess, PLANES        # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
PRE = "agent._sparsedrive_model._backbone.img_backbone."


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=4)
    args = ap.parse_args()

    frame = sorted((EXP / "golden").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    K = "_backbone.img_backbone."
    x_ref = d[K + "in[0]"]                                  # [3, 3, 256, 512]
    refs = [d[K + f"out[{i}]"] for i in range(4)]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]

    B, _, H, W = x_ref.shape
    print(f"  frame {frame.name}   batch(cams)={B}  input {H}x{W}")
    for i, r in enumerate(refs):
        print(f"    stage {i}: {tuple(r.shape)}")
    print()

    # conv2d allocates from the L1_SMALL region, which open_device leaves at 0
    # by default; 24576 is what sparse4D-tt's conv tests use.
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        params = preprocess(sd, PRE, dev)
        net = TtResNet34(params, dev, batch_size=B, in_h=H, in_w=W,
                         out_indices=tuple(range(args.stages)))
        # NHWC, padded to 4 channels, flattened the way conv2d wants it
        xt = torch.nn.functional.pad(x_ref.permute(0, 2, 3, 1), (0, 1))
        x = ttnn.from_torch(xt.reshape(1, 1, B * H * W, 4), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

        outs = net(x)                                        # warm
        ttnn.synchronize_device(dev); t0 = time.time()
        outs = net(ttnn.from_torch(xt.reshape(1, 1, B * H * W, 4), dtype=ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev))
        ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3
        print(f"  [ResNet-34] {dt:7.1f} ms (웜업 후, {args.stages} stage)")

        ok = True
        for i, (t, h, w, c) in enumerate(outs):
            got = ttnn.to_torch(t).float().reshape(B, h, w, c).permute(0, 3, 1, 2)
            ref = refs[i]
            p = pcc(got, ref)
            ok &= p >= 0.99
            print(f"      stage {i} {str(tuple(got.shape)):22s} PCC {p:.6f}"
                  f"   max|d| {(got - ref).abs().max():.3e}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
