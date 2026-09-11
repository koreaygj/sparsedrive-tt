"""CPU vs CUDA agreement on intermediate tensors, not just the output.

The final trajectory is an argmax lookup into a frozen vocabulary, so two runs
agreeing there only means they picked the same mode -- it says nothing about
the numerics that produced the choice. This hooks the modules that actually
matter and compares those.

Why it matters: the CPU path is the one checked index-for-index against
deformable_aggregation_cuda.cu (test_daf_torch.py). Agreement here is what
licenses using the much faster GPU to produce Phase 1 golden tensors.

ANCHOR ORDER IS NOT STABLE ACROSS BACKENDS. Layer 0 scores 1024 paths and
keeps the top 128; torch.gather then reorders the survivors by rank. Scores
that differ by ~1e-2 between backends -- ordinary fp32 reassociation -- can
swap neighbouring ranks even when the selected *set* is identical. Downstream
tensors are then the same rows in a different order, and an elementwise
comparison reads as a numerical failure when nothing is wrong. So layer-1
tensors are permutation-aligned before comparison, and the alignment itself is
reported. Expect the same on TT-NN, where bf16 will widen the score gap.
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference                      # noqa: E402
reference.install()

from reference.pcc import compare     # noqa: E402
from test_forward_smoke import build_config, build_inputs, to_device, CKPT, PREFIX  # noqa: E402
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel  # noqa: E402

# Layer 1 consumes layer 0's top-k survivors, so its rows are only comparable
# after alignment; see ALIGNED below.
HOOKS = [
    "_backbone",
    "_trajectory_head.decoder.layers.0.p_deform_model",
    "_trajectory_head.decoder.layers.0.path_mlp",
    "_trajectory_head.decoder.layers.0.vel_mlp",
    "_trajectory_head.decoder.layers.1.p_deform_model",
    "_trajectory_head.decoder.layers.1.t_deform_model",
    "_trajectory_head.decoder.layers.1.traj_mlp",
]


def capture(device, cfg):
    model = SparseDriveModel(cfg)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len(PREFIX):]: v for k, v in raw.items() if k.startswith(PREFIX)}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    grabbed = {}
    handles = []
    named = dict(model.named_modules())
    for name in HOOKS:
        mod = named[name]

        def mk(n):
            def fn(_m, _i, out):
                if isinstance(out, (list, tuple)):
                    for j, t in enumerate(out):
                        if torch.is_tensor(t):
                            grabbed[f"{n}[{j}]"] = t.detach().float().cpu()
                elif torch.is_tensor(out):
                    grabbed[n] = out.detach().float().cpu()
            return fn

        handles.append(mod.register_forward_hook(mk(name)))

    with torch.no_grad():
        out, _ = model(to_device(build_inputs(cfg), device), {})
    for h in handles:
        h.remove()
    grabbed["OUTPUT.trajectory"] = out["trajectory"].detach().float().cpu()
    return grabbed


ALIGNED = {
    "_trajectory_head.decoder.layers.1.p_deform_model": (
        "_trajectory_head.decoder.layers.0.path_mlp", "path_filter_num"),
}


def align(name, gpu, cpu, cfg):
    """Reorder gpu rows into cpu's rank order. Returns (tensor, note)."""
    spec = ALIGNED.get(name)
    if spec is None:
        return gpu[name], ""
    score_key, filt = spec
    k = getattr(cfg, filt)[0]
    ig = torch.topk(gpu[score_key].squeeze(-1).squeeze(0), k).indices
    ic = torch.topk(cpu[score_key].squeeze(-1).squeeze(0), k).indices
    if set(ig.tolist()) != set(ic.tolist()):
        inter = len(set(ig.tolist()) & set(ic.tolist()))
        return gpu[name], f"   <-- top-k SET differs ({inter}/{k} shared)"
    if torch.equal(ig, ic):
        return gpu[name], "  (same order)"
    pos = {v: i for i, v in enumerate(ig.tolist())}
    perm = torch.tensor([pos[v] for v in ic.tolist()])
    return gpu[name][:, perm], "  (reordered)"


def main():
    if not torch.cuda.is_available():
        print("  cuda unavailable")
        return 1

    cfg = build_config()
    gpu = capture("cuda", cfg)
    cpu = capture("cpu", cfg)

    assert gpu.keys() == cpu.keys(), (gpu.keys() ^ cpu.keys())

    # fp32 on both sides, but cuDNN convolutions and the 6000-wide softmax do
    # not associate the way the CPU kernels do. PCC is the gate; max_abs is
    # reported to show the drift is small in absolute terms too.
    FLOOR = 0.9999
    worst = 1.0
    print(f"  {'tensor':-<58s} {'PCC':>10s} {'max|d|':>10s} {'scale':>9s}")
    for k in cpu:
        g, note = align(k, gpu, cpu, cfg)
        c = compare(g, cpu[k])
        worst = min(worst, c["pcc"])
        if c["pcc"] < FLOOR:
            note += "   <-- BELOW FLOOR"
        print(f"  {k:.<58s} {c['pcc']:10.6f} {c['max_abs']:10.2e} "
              f"{c['scale']:9.3f}{note}")

    print(f"  worst PCC {worst:.6f}  (floor {FLOOR})")
    ok = worst >= FLOOR
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
