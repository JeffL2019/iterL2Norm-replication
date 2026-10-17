"""
IterL2Norm: Fast Iterative L2-Normalization
Ye, Sim, Kim, Jin, and Jeong (DATE 2025)

Run this file directly as:
    python iterl2norm.py
"""

import math
import torch


def iterl2norm(x, n_iters=5, gamma=None, beta=None, lam_coef=0.4):
    """
    Layer normalization using IterL2Norm instead of sqrt and division.

    x        : tensor of shape (..., d). Each row of length d is normalized.
               The dtype of x (float32, float16, bfloat16) is the format
               all the math runs in, to mimic the hardware.
    n_iters  : number of update steps (the paper uses 5)
    gamma    : scale parameters, shape (d,). Default: all ones.
    beta     : shift parameters, shape (d,). Default: all zeros.
    lam_coef : constant in the update-rate formula (0.4, from Fig. 2a)
    """
    d = x.shape[-1]

    # Shift the mean to zero 
    y = x - x.mean(dim=-1, keepdim=True)

    # Squared length of y
    m = (y * y).sum(dim=-1, keepdim=True)

    # Starting point a0 and step size lambda
    _, e = torch.frexp(m)
    e = e.to(torch.float32)

    # Eq. (6): a0 = 2^(-(E - bias + 1)/2) = 2^(-e/2)
    a = torch.exp2(-e / 2).to(x.dtype)

    # Eq. (10): lambda = 0.4 * 2^(-(E - bias)) = 0.4 * 2^(-(e - 1))
    lam = (lam_coef * torch.exp2(-(e - 1))).to(x.dtype)

    # Iterate toward a = 1/sqrt(m)
    for _ in range(n_iters):
        a = a + lam * m * a * (1 - m * a * a)

    # Normalize, then scale and shift
    y_hat = math.sqrt(d) * a * y

    if gamma is not None:
        y_hat = gamma * y_hat
    if beta is not None:
        y_hat = y_hat + beta
    return y_hat


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 768
    n_iters = 5

    x = torch.rand(1000, d) * 2 - 1

    # Ground truth
    ref = torch.nn.functional.layer_norm(x, (d,))

    print(f"Average / max absolute error vs. PyTorch LayerNorm (d = {d}, {n_iters} iterations)")
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        out = iterl2norm(x.to(dtype), n_iters=n_iters).float()
        err = (out - ref).abs()
        print(f"  {str(dtype):16s} avg {err.mean():.2e}   max {err.max():.2e}")

    # Print single vector convergence
    v = x[:1]
    y = v - v.mean()
    target = 1 / torch.sqrt((y * y).sum())
    print(f"\nOne vector: true a = 1/sqrt(m) = {target:.8f}")
    for n in range(11):
        a_n = iterl2norm(v, n_iters=n)[0, 0] / (math.sqrt(d) * y[0, 0])
        print(f"  step {n}:  a = {a_n:.8f}   relative error {abs(a_n - target) / target:.1e}")