"""Pearson correlation between two tensors, the accuracy gate used throughout.

TT-NN work is graded on PCC rather than allclose because bf16 accumulation
shifts magnitudes without changing the shape of the answer. Same convention as
sparse4D-tt: 0.999 is the working floor for a module, 0.9999 for anything on
the critical path.
"""

import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().to(torch.float64).cpu()
    b = b.detach().flatten().to(torch.float64).cpu()
    if a.numel() != b.numel():
        raise ValueError(f"size mismatch: {a.numel()} vs {b.numel()}")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        # Both constant: correlated iff they are the same constant.
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a_ = a.detach().to(torch.float64).cpu()
    b_ = b.detach().to(torch.float64).cpu()
    diff = (a_ - b_).abs()
    return {
        "pcc": pcc(a, b),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "scale": float(b_.abs().max()) if b_.numel() else 0.0,
    }
