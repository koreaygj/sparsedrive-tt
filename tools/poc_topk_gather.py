"""Phase 2: the top-k filter and its gather, on device.

Layer 0 scores 1024 paths and keeps 128, and 256 velocity profiles and keeps
64; torch.gather then reorders the survivors by rank. Graded against golden
layers.1.p_deform_model.in[0] and in[1] -- what layer 1 actually received.

Two custom kernels from sparse4D-tt fit this exactly:

    topk_select(scores, values, indices, k)   scores BF16 TILE, out RM
    row_gather(src_a, idx, out_a, src_b, out_b, k)   two sources per call

row_gather taking a pair is what this needs: the embedding [1024,256] and the
path vocabulary [1024,100] are gathered by the same indices.

Phase 1 found that top-k *order* is not stable across backends -- CPU and GPU
selected the same 128 paths and ranked them differently, because scores
differing by 6e-2 sat against a boundary gap of 2.8e-2. bf16 widens that gap,
so this reports set agreement and rank agreement separately, and grades the
gathered tensors after aligning order.

    $TT_PY tools/poc_topk_gather.py
"""

import os
import pathlib
import sys
import time

import torch
import ttnn

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    frame = sorted((EXP / "golden").glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    D = "_trajectory_head.decoder.layers."
    scores = d[D + "0.path_mlp.out"][0, :, 0]              # [1024]
    # p_norm2's output, not p_ffn's. The decoder does
    #     path_embed = path_embed + dropout(p_ffn(path_embed))
    #     path_embed = p_norm2(path_embed)
    # and gathers that; p_ffn.out is the branch before the residual and the
    # norm. Using it scores 0.33 while the vocabulary gathered alongside it
    # scores 0.999998 -- a split like that is a wrong input, not a wrong kernel.
    embed = d[D + "0.path_mlp.in[0]"][0]                  # [1024, 256]
    vocab = d[D + "0.p_deform_model.in[1]"][0]            # [1024, 100]
    # Layer 1 adds the ego-status encoding before it calls its DFA:
    #     path_embed = path_embed + status_encoding.unsqueeze(1)
    #     path_embed = self.p_deform_model(path_embed, ...)
    # so the golden tensor at that boundary is the gather plus that vector, not
    # the gather. Subtract it back out to score the gather alone. (Missing this
    # reads as PCC 0.88 -- close, because it is off by one constant row.)
    status = d["_status_encoding.out"][0]                 # [256]
    ref_embed = d[D + "1.p_deform_model.in[0]"][0] - status   # [128, 256]
    ref_vocab = d[D + "1.p_deform_model.in[1]"][0]        # [128, 100]

    N, K, E, V = scores.shape[0], ref_embed.shape[0], embed.shape[1], vocab.shape[1]
    print(f"  frame {frame.name}   N={N} -> K={K}   embed {E}  vocab {V}")

    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        s_tt = ttnn.from_torch(scores.reshape(1, N), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
        val = ttnn.from_torch(torch.zeros(1, K), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev, dtype=ttnn.bfloat16)
        idx = ttnn.from_torch(torch.zeros(1, K, dtype=torch.int32),
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                              dtype=ttnn.uint32)
        e_tt = ttnn.from_torch(embed, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        v_tt = ttnn.from_torch(vocab, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        oe = ttnn.from_torch(torch.zeros(K, E), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ov = ttnn.from_torch(torch.zeros(K, V), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        def run():
            ttnn.topk_select(s_tt, val, idx, K)
            ttnn.row_gather(e_tt, idx, oe, v_tt, ov, K)

        run()
        ttnn.synchronize_device(dev); t0 = time.time()
        run()
        ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3

        got_idx = ttnn.to_torch(idx).flatten()[:K].to(torch.int64)
        ref_idx = torch.topk(scores, K).indices
        same_set = set(got_idx.tolist()) == set(ref_idx.tolist())
        same_ord = torch.equal(got_idx, ref_idx)
        overlap = len(set(got_idx.tolist()) & set(ref_idx.tolist()))
        srt = torch.sort(scores, descending=True).values
        print(f"  [topk+gather] {dt:6.2f} ms (웜업 후)")
        print(f"      집합 일치 {same_set}   순서 일치 {same_ord}   겹침 {overlap}/{K}")
        print(f"      경계 간격 s[{K-1}]-s[{K}] = {(srt[K-1]-srt[K]):.3e}")

        got_e = ttnn.to_torch(oe).float()[:K]
        got_v = ttnn.to_torch(ov).float()[:K]
        # Align to the reference ranking before comparing: a permutation is not
        # an error, and an elementwise score would call it one.
        if same_set and not same_ord:
            pos = {v: i for i, v in enumerate(got_idx.tolist())}
            perm = torch.tensor([pos[v] for v in ref_idx.tolist()])
            got_e, got_v = got_e[perm], got_v[perm]
            note = "  (순서 정렬 후)"
        else:
            note = ""
        pe, pv = pcc(got_e, ref_embed), pcc(got_v, ref_vocab)
        print(f"      embed  PCC {pe:.6f}{note}")
        print(f"      vocab  PCC {pv:.6f}{note}")
        ok = same_set and pe >= 0.999 and pv >= 0.999
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
