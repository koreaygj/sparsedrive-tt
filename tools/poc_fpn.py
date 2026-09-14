"""Phase 2: ResNet-34 + FPN, the whole backbone, against golden."""
import argparse, os, pathlib, sys, time
import torch, ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.resnet34 import TtResNet34, preprocess as pre_r      # noqa: E402
from model.fpn import TtFPN, preprocess as pre_f                # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
B_PRE = "agent._sparsedrive_model._backbone.img_backbone."
N_PRE = "agent._sparsedrive_model._backbone.img_neck."


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    frame = sorted((EXP / "golden").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    x_ref = d["_backbone.img_backbone.in[0]"]
    refs = [d[f"_backbone.img_neck.out.feat_{i}"] for i in range(4)]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    B, _, H, W = x_ref.shape
    sizes = [(H // 4, W // 4), (H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
    print(f"  frame {frame.name}  batch={B}  levels {sizes}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        net = TtResNet34(pre_r(sd, B_PRE, dev), dev, batch_size=B, in_h=H, in_w=W)
        fpn = TtFPN(pre_f(sd, N_PRE), dev, B, sizes)
        xt = torch.nn.functional.pad(x_ref.permute(0, 2, 3, 1), (0, 1))

        def run():
            x = ttnn.from_torch(xt.reshape(1, 1, B * H * W, 4), dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            return fpn(net(x))

        run()
        ttnn.synchronize_device(dev); t0 = time.time()
        outs = run()
        ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3
        print(f"  [ResNet-34 + FPN] {dt:7.1f} ms (웜업 후)")
        ok = True
        for i, (t, h, w, c) in enumerate(outs):
            got = ttnn.to_torch(t).float().reshape(B, h, w, c).permute(0, 3, 1, 2)
            p = pcc(got, refs[i]); ok &= p >= 0.99
            print(f"      feat_{i} {str(tuple(got.shape)):22s} PCC {p:.6f}"
                  f"   max|d| {(got - refs[i]).abs().max():.3e}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
