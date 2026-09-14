"""nn.MultiheadAttention and the decoder's FFN on TT-NN.

The decoder uses stock torch modules, so the contract is torch's: a single
in_proj holding Wq|Wk|Wv stacked, then an out_proj.

    qkv  = x @ in_proj_weight.T + in_proj_bias        [B, T, 768]
    heads: [B, T, 8, 32] -> [B, 8, T, 32]
    attn = softmax(q @ k.T / sqrt(32))
    out  = (attn @ v) -> [B, T, 256] -> out_proj

head_dim is 256/8 = 32, exactly the tile width, so the head split is a reshape
with no padding -- the one place in this model where the natural layout and the
tile grid agree for free.

Self-attention gets all three projections from one matmul. Cross-attention
(v_img_attention: velocity queries against image tokens) has to split in_proj,
since q comes from a different tensor than k and v.
"""

import math

import ttnn

HIFI = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
            packer_l1_acc=True)


_MESH = {}


def _mapper(device):
    """Cache the replicate mapper per device; building one per call is wasteful."""
    if id(device) not in _MESH:
        from .mesh import mesh_of
        _MESH[id(device)] = mesh_of(device)
    return _MESH[id(device)]


def _t(x, device, dtype=ttnn.bfloat16):
    from .dfa import _cast
    _, rep, _ = _mapper(device)
    return ttnn.from_torch(_cast(x, dtype).contiguous(), layout=ttnn.TILE_LAYOUT,
                           device=device, dtype=dtype, mesh_mapper=rep)


def to_host(t, device):
    """Compose a replicated tensor back. Callers slice [:T], which also
    discards the second copy."""
    _, _, cat = _mapper(device)
    return ttnn.to_torch(t, mesh_composer=cat) if cat else ttnn.to_torch(t)


class TtMultiheadAttention:
    def __init__(self, sd, prefix, device, embed_dim=256, num_heads=8):
        self.device, self.E, self.H = device, embed_dim, num_heads
        self.hd = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.hd)
        W = sd[prefix + "in_proj_weight"].float()          # [3E, E]
        B = sd[prefix + "in_proj_bias"].float()            # [3E]
        self.w_all = _t(W.t(), device)
        self.b_all = _t(B.reshape(1, -1), device)
        E = embed_dim
        self.w_q, self.w_k, self.w_v = (_t(W[i*E:(i+1)*E].t(), device) for i in range(3))
        self.b_q, self.b_k, self.b_v = (_t(B[i*E:(i+1)*E].reshape(1, -1), device)
                                        for i in range(3))
        self.w_o = _t(sd[prefix + "out_proj.weight"].float().t(), device)
        self.b_o = _t(sd[prefix + "out_proj.bias"].float().reshape(1, -1), device)
        self.cfg = ttnn.WormholeComputeKernelConfig(**HIFI)

    def _heads(self, x, T):
        # [T, E] -> [1, H, T, hd]
        x = ttnn.reshape(x, (1, T, self.H, self.hd))
        return ttnn.permute(x, (0, 2, 1, 3))

    def __call__(self, query, key=None, value=None, tq=None, tk=None):
        """query [Tq, E]; key/value [Tk, E] or None for self-attention."""
        if key is None:
            qkv = ttnn.linear(query, self.w_all, bias=self.b_all,
                              compute_kernel_config=self.cfg)     # [Tq, 3E]
            E = self.E
            q = ttnn.slice(qkv, [0, 0], [tq, E])
            k = ttnn.slice(qkv, [0, E], [tq, 2 * E])
            v = ttnn.slice(qkv, [0, 2 * E], [tq, 3 * E])
            ttnn.deallocate(qkv)
            tk = tq
        else:
            q = ttnn.linear(query, self.w_q, bias=self.b_q, compute_kernel_config=self.cfg)
            k = ttnn.linear(key, self.w_k, bias=self.b_k, compute_kernel_config=self.cfg)
            v = ttnn.linear(value, self.w_v, bias=self.b_v, compute_kernel_config=self.cfg)

        qh, kh, vh = self._heads(q, tq), self._heads(k, tk), self._heads(v, tk)
        scores = ttnn.matmul(qh, ttnn.permute(kh, (0, 1, 3, 2)),
                             compute_kernel_config=self.cfg)      # [1,H,Tq,Tk]
        scores = ttnn.multiply(scores, self.scale)
        attn = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        out = ttnn.matmul(attn, vh, compute_kernel_config=self.cfg)   # [1,H,Tq,hd]
        ttnn.deallocate(attn)
        out = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (tq, self.E))
        return ttnn.linear(out, self.w_o, bias=self.b_o, compute_kernel_config=self.cfg)


class TtFFN:
    """Linear(E, ffn) -> ReLU -> Linear(ffn, E)."""

    def __init__(self, sd, prefix, device):
        self.w1 = _t(sd[prefix + "0.weight"].float().t(), device)
        self.b1 = _t(sd[prefix + "0.bias"].float().reshape(1, -1), device)
        self.w2 = _t(sd[prefix + "2.weight"].float().t(), device)
        self.b2 = _t(sd[prefix + "2.bias"].float().reshape(1, -1), device)
        self.cfg = ttnn.WormholeComputeKernelConfig(**HIFI)

    def __call__(self, x):
        h = ttnn.relu(ttnn.linear(x, self.w1, bias=self.b1,
                                  compute_kernel_config=self.cfg))
        out = ttnn.linear(h, self.w2, bias=self.b2, compute_kernel_config=self.cfg)
        ttnn.deallocate(h)
        return out
