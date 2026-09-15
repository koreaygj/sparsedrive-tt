"""Phase 3 step 4: images to trajectory, one frame, against golden."""
import os, pathlib, sys, time
import torch, ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.sparsedrive import TtSparseDrive   # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"


def main():
    frame = sorted((EXP / "golden").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    imgs = d["INPUT.imgs"][0]                      # [3, 3, 256, 512]
    status = d["INPUT.status_feature"][0]          # [8]
    proj = d["INPUT.projection_mat"][0]            # [3, 4, 4]
    iwh = d["INPUT.image_wh"][0]                   # [3, 2]
    ref = d["OUTPUT.trajectory"][0]                # [8, 3]
    print(f"  frame {frame.name}   imgs {tuple(imgs.shape)}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        net = TtSparseDrive(sd, dev)
        t0 = time.time()
        traj = net(imgs, status, proj, iwh)
        dt = (time.time() - t0) * 1e3
        err = float((traj - ref).abs().max())
        print(f"  [end to end] {dt:8.1f} ms")
        print(f"      trajectory {tuple(traj.shape)}  max|d| {err:.3e}"
              f"  {'same candidate' if err < 1e-6 else 'DIFFERENT candidate'}")
        print(f"      got  {[round(v,4) for v in traj[-1].tolist()]}  (last pose)")
        print(f"      want {[round(v,4) for v in ref[-1].tolist()]}")
        ok = err < 1e-6
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
