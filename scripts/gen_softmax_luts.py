"""
Generate bipartite exp(-d) LUT hex files for the softmax module

Based on https://www.mdpi.com/2072-666X/17/1/84 (Section 4.2.3)

LUT entries are jointly optimized using Adam over 10,000 uniformly
sampled points in [0, 16) with a hybrid MSE + MAX loss:
  L = 0.5 * MSE + 1.0 * max|error|
Adam lr=1e-4, 1000 epochs, initialized from analytical bipartite formulas
Fake quantization + Straight-Through Estimator (STE) in forward pass

Input d is Q4.7 (11-bit unsigned), split into (x0, x1, x2) = (5, 3, 3) bits:
  x0 = d[10:6],  x1 = d[5:3],  x2 = d[2:0]

Eq 22: $\hat{f}(x) = \tilde{a}_0(x_0, x_1) + \tilde{a}_1(x_0, x_2)$

LUT_0: 256 x 16-bit unsigned (primary table)
LUT_1: 256 x 16-bit signed   (correction table, stored as 2's complement)
"""

import math
import os
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM = os.path.join(PROJ, "mem")

N_Q47 = 2048  # all valid Q4.7 values: 0..2047 (0.0 to ~16.0)

"""
Compensation terms (Eq 21)
$\delta_1 = 2^{-(n_0+1)} - 2^{-(n_0+n_1+1)}$, $\delta_2 = 2^{-(n_0+n_1+1)} - 2^{-(n_0+n_1+n_2+1)}$
For $(n_0, n_1, n_2) = (5, 3, 3)$: $\delta_1 = 7/32$, $\delta_2 = 7/256$
"""
DELTA1 = 7.0 / 32.0    # 0.21875, midpoint of x1 range
DELTA2 = 7.0 / 256.0   # 0.02734375, midpoint of x2 range

"""
Compensated analytical initialization for LUT_0
Eq 19: $\tilde{a}_0(x_0, x_1) = \lfloor 2^{15} \cdot \exp(-Q_{n_0}(x_0) - Q_{n_1}(x_1) - \delta_2) \rceil$
"""
def init_lut0():
    entries = torch.zeros(256, dtype=torch.float64)
    for idx in range(256):
        x0 = (idx >> 3) & 0x1F
        x1 = idx & 0x07
        x0_val = x0 / 2.0
        x1_val = x1 / 16.0
        entries[idx] = math.exp(-(x0_val + x1_val + DELTA2)) * 32768.0
    return entries

"""
Compensated analytical initialization for LUT_1
Eq 20: $\tilde{a}_1(x_0, x_2) = \lfloor 2^{15} \cdot (-\exp'(-Q_{n_0}(x_0) - \delta_1 - \delta_2)) \cdot (Q_{n_2}(x_2) - \delta_2) \rceil$
"""
def init_lut1():
    entries = torch.zeros(256, dtype=torch.float64)
    for idx in range(256):
        x0 = (idx >> 3) & 0x1F
        x2 = idx & 0x07
        x0_val = x0 / 2.0
        x2_val = x2 / 128.0
        entries[idx] = -math.exp(-(x0_val + DELTA1 + DELTA2)) * (x2_val - DELTA2) * 32768.0
    return entries

"""
Section 4.2.3: 10,000 uniformly sampled points in [0, 16]
Adam optimizer, lr=1e-4, 1000 epochs
Hybrid loss: L = alpha * MSE + beta * max|error|
"""
def optimize_luts(n_points=10000, n_epochs=1000, lr=1e-4, alpha=0.5, beta=1.0):

    # Sample training points uniformly in [0, 16), quantized to Q4.7
    torch.manual_seed(42)
    d_float = torch.rand(n_points, dtype=torch.float64) * 16.0
    d_q47 = (d_float * 128.0).long().clamp(0, N_Q47 - 1)

    # Compute address indices and targets for training set
    x0_t = (d_q47 >> 6) & 0x1F
    x1_t = (d_q47 >> 3) & 0x07
    x2_t = d_q47 & 0x07
    a0_t = (x0_t << 3) | x1_t
    a1_t = (x0_t << 3) | x2_t
    tgt_t = torch.exp(-d_q47.double() / 128.0) * 32768.0

    # Initialize LUT parameters from analytical formulas
    lut0 = init_lut0().requires_grad_(True)
    lut1 = init_lut1().requires_grad_(True)

    optimizer = torch.optim.Adam([lut0, lut1], lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Fake quantization with STE (Section 4.2.3):
        # Forward pass rounds to integers, backward pass uses identity gradient
        l0_q = lut0 + (torch.round(lut0).clamp(0, 65535) - lut0).detach()
        l1_q = lut1 + (torch.round(lut1).clamp(-32768, 32767) - lut1).detach()

        approx = l0_q[a0_t] + l1_q[a1_t]
        errors = approx - tgt_t

        mse = (errors ** 2).mean()
        max_abs = errors.abs().max()
        loss = alpha * mse + beta * max_abs

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                # Evaluate with quantized LUTs on all 2048 integer points
                l0_q = torch.round(lut0).clamp(0, 65535)
                l1_q = torch.round(lut1).clamp(-32768, 32767)
                d_all = torch.arange(N_Q47, dtype=torch.long)
                x0_a = (d_all >> 6) & 0x1F
                x1_a = (d_all >> 3) & 0x07
                x2_a = d_all & 0x07
                a0_a = (x0_a << 3) | x1_a
                a1_a = (x0_a << 3) | x2_a
                tgt_a = torch.exp(-d_all.double() / 128.0) * 32768.0
                approx_a = l0_q[a0_a] + l1_q[a1_a]
                errs = (approx_a - tgt_a).abs()
            print(f"  epoch {epoch:4d}: loss={loss.item():.4f}  "
                  f"mse={mse.item():.4f}  max={max_abs.item():.2f}  "
                  f"quant_max={errs.max().item():.2f}  quant_mean={errs.mean().item():.4f}")

    # Quantize final LUTs
    lut0_q = [int(max(0, min(65535, round(v)))) for v in lut0.detach().tolist()]
    lut1_q = [int(max(-32768, min(32767, round(v)))) for v in lut1.detach().tolist()]
    return lut0_q, lut1_q


def evaluate_luts(lut0, lut1):
    max_abs = 0
    max_rel = 0
    sum_abs = 0
    for d in range(N_Q47):
        x0 = (d >> 6) & 0x1F
        x1 = (d >> 3) & 0x07
        x2 = d & 0x07
        approx = lut0[(x0 << 3) | x1] + lut1[(x0 << 3) | x2]
        ideal = math.exp(-d / 128.0) * 32768.0
        abs_err = abs(approx - ideal)
        sum_abs += abs_err
        if abs_err > max_abs:
            max_abs = abs_err
        if ideal > 1.0:
            rel = abs_err / ideal
            if rel > max_rel:
                max_rel = rel
    print(f"\nFinal error over {N_Q47} Q4.7 inputs:")
    print(f"  Max absolute: {max_abs:.2f} (Q1.15 units)")
    print(f"  Mean absolute: {sum_abs / N_Q47:.4f}")
    print(f"  Max relative (where ideal>1): {max_rel*100:.4f}%")


def main():
    os.makedirs(MEM, exist_ok=True)

    print("Optimizing bipartite LUTs (Adam, lr=1e-4, 1000 epochs, 10k points)...")
    lut0, lut1 = optimize_luts()

    evaluate_luts(lut0, lut1)

    path0 = os.path.join(MEM, "exp_lut0.hex")
    with open(path0, "w") as f:
        for val in lut0:
            f.write(f"{val:04x}\n")
    print(f"\nWrote {path0} ({len(lut0)} entries, unsigned)")

    path1 = os.path.join(MEM, "exp_lut1.hex")
    with open(path1, "w") as f:
        for val in lut1:
            if val < 0:
                val = val + 65536
            f.write(f"{val:04x}\n")
    print(f"Wrote {path1} ({len(lut1)} entries, signed)")


if __name__ == "__main__":
    main()
