"""Phase 3 step 3: the whole decoder, against golden OUTPUT.trajectory."""
import os, pathlib, sys, time
import torch, ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.decoder import TtDecoder   # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
M = "agent._sparsedrive_model."


def main():
    frame = sorted((EXP / "golden_dfa").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    G = "_trajectory_head."

    path_embed = d[G + "path_pos_embed.out"][0]            # [1024, 256]
    vel_embed = d[G + "vel_pos_embed.out"][0]              # [256, 256]
    status = d["_status_encoding.out"][0]                  # [256]
    path_vocab = sd[M + "_trajectory_head.path_vocab"].float()      # [1024,50,3]
    traj_vocab = sd[M + "_trajectory_head.traj_vocab"].float()      # [1024,256,8,3]
    mc = d["DAF[0].mc_ms_feat"]
    shapes = [tuple(int(v) for v in r) for r in d["DAF[0].spatial_shape"]]
    starts = [int(v) for v in d["DAF[0].scale_start_index"]]
    K = G + "decoder.layers.0.p_deform_model."
    proj = d[K + "in[4].projection_mat"][0]
    iwh = d[K + "in[4].image_wh"][0]
    fm3 = d[K + "in[4].feature_maps[3]"][0]                # [3, 256, 8, 16]
    img_value = fm3.permute(0, 2, 3, 1).reshape(-1, 256)   # [384, 256]
    ref = d["OUTPUT.trajectory"][0]                        # [8, 3]
    print(f"  frame {frame.name}")
    print(f"  path {tuple(path_embed.shape)}  vel {tuple(vel_embed.shape)}"
          f"  img tokens {img_value.shape[0]}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        dec = TtDecoder(sd, M + "_trajectory_head.decoder.", dev)
        levels = dec.layers[0].p_dfa.upload_levels(mc, shapes, starts)
        t0 = time.time()
        traj = dec(path_embed, vel_embed, path_vocab, traj_vocab, status,
                   levels, img_value, proj, iwh)
        dt = (time.time() - t0) * 1e3
        err = (traj - ref).abs().max()
        exact = torch.equal(traj, ref)
        print(f"  [decoder] {dt:8.1f} ms")
        print(f"      trajectory {tuple(traj.shape)}  max|d| {err:.3e}"
              f"  {'같은 후보 선택' if err < 1e-6 else '다른 후보 선택'}")
        print(f"      got  {traj[:3].tolist()}")
        print(f"      want {ref[:3].tolist()}")
        ok = err < 1e-6
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
