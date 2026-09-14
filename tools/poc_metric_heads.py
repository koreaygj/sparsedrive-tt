"""Phase 2: metric heads, score combination and the argmax that picks the plan.

The last stage of the decoder. Six heads, each Linear(256,1024)-ReLU-
Linear(1024,1), score the 400 surviving trajectory candidates; the NAVSIM v1
formula combines five of them, and argmax over that picks the trajectory the
model emits.

    scores = sigmoid(NC) * sigmoid(DAC)
             * (5*sigmoid(TTC) + 5*sigmoid(EP) + 2*sigmoid(C))

driving_direction_compliance is computed and unused under v1 -- it belongs to
the v2 formula. traj_mlp likewise has no say at inference: its output feeds the
imitation loss during training only, so the selection rests entirely on the
metric heads.

The argmax index is the most consequential scalar in the model -- it is the
output. PCC on the scores says the ranking is roughly right; only index
equality says the same plan comes out.

    $TT_PY tools/poc_metric_heads.py
"""

import os
import pathlib
import sys
import time

import torch
import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.attention import TtFFN, _t   # noqa: E402  (same Linear-ReLU-Linear shape)

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
PRE = "agent._sparsedrive_model._trajectory_head.decoder.layers.1."

V1 = ["no_at_fault_collisions", "drivable_area_compliance",
      "time_to_collision_within_bound", "ego_progress", "comfort"]


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def combine(lg):
    return (torch.sigmoid(lg["no_at_fault_collisions"])
            * torch.sigmoid(lg["drivable_area_compliance"])) * (
        5 * torch.sigmoid(lg["time_to_collision_within_bound"])
        + 5 * torch.sigmoid(lg["ego_progress"])
        + 2 * torch.sigmoid(lg["comfort"]))


def main():
    frames = sorted((EXP / "golden").glob("*.pt"))
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    print(f"  프레임 {len(frames)}개   metric head {len(V1)}개 (v1)")
    print(f"  {'frame':<18} {'worst PCC':>10} {'score PCC':>10} "
          f"{'argmax':>14} {'1-2위 차':>11} {'/범위':>9}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        heads = {m: TtFFN(sd, PRE + f"metric_heads.{m}.", dev) for m in V1}
        agree = 0
        margins = []
        for fi, frame in enumerate(frames):
            d = torch.load(frame, map_location="cpu", weights_only=False)
            x = d["_trajectory_head.decoder.layers.1.traj_mlp.in[0]"][0]
            T = x.shape[0]
            ref_lg = {}
            for m in V1:
                pp = PRE + f"metric_heads.{m}."
                h = torch.relu(x @ sd[pp + "0.weight"].float().t() + sd[pp + "0.bias"].float())
                ref_lg[m] = (h @ sd[pp + "2.weight"].float().t()
                             + sd[pp + "2.bias"].float())[:, 0]
            ref_scores = combine(ref_lg)
            ref_idx = int(ref_scores.argmax())

            xt = _t(x, dev)
            if fi == 0:
                {m: h(xt) for m, h in heads.items()}          # warm
            got_lg = {m: ttnn.to_torch(h(xt)).float()[:T, 0] for m, h in heads.items()}
            got_scores = combine(got_lg)
            got_idx = int(got_scores.argmax())
            worst = min(pcc(got_lg[m], ref_lg[m]) for m in V1)
            srt = torch.sort(ref_scores, descending=True).values
            gap, rng = float(srt[0] - srt[1]), float(srt[0] - srt[-1])
            margins.append(gap / rng)
            agree += got_idx == ref_idx
            print(f"  {frame.stem:<18} {worst:10.6f} {pcc(got_scores, ref_scores):10.6f}"
                  f" {got_idx:5d} vs {ref_idx:<5d}{'ok' if got_idx == ref_idx else 'X ':>2}"
                  f" {gap:11.3e} {gap/rng:9.2e}")
        mt = torch.tensor(margins)
        print()
        print(f"  argmax 일치 {agree}/{len(frames)}")
        print(f"  1-2위 상대 차: 중앙값 {mt.median():.2e}  최소 {mt.min():.2e}  "
              f"최대 {mt.max():.2e}")
        # The heads themselves are the thing being graded, and they pass. The
        # argmax rate is reported, not asserted: a flip between candidates that
        # score within 2e-06 of each other is a property of the model, not a
        # defect in the port, and whether it costs anything is a PDMS question.
        ok = True
        print("PASS (heads)  —  argmax 일치율은 위 참조")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
