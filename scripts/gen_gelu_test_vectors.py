"""
Generate fp16 test input vectors for tb_gelu

Sweep interesting fp16 values in [-8, 8] plus edge cases.
Output: mem/gelu_test_inputs.hex (fp16, 4-digit hex per line)
"""

import os
import struct
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "mem", "gelu_test_inputs.hex")

# Read breakpoints from gelu_pwl.hex so we can test near them
PWL_HEX = os.path.join(PROJ, "mem", "gelu_pwl.hex")


def fp16_bits(f):
    return int(np.float16(f).view(np.uint16))


def fp16_float(bits):
    return float(np.uint16(bits).view(np.float16))


def load_breakpoints():
    breaks = []
    with open(PWL_HEX) as f:
        count = 0
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if count < 15:
                breaks.append(fp16_float(int(s, 16)))
            count += 1
            if count >= 15:
                break
    return breaks


if __name__ == "__main__":
    breaks = load_breakpoints()
    print(f"Loaded {len(breaks)} breakpoints")

    test_vals = set()

    # Uniform sweep in [-6, 6] at fp16 granularity (step ~0.05)
    for x in np.arange(-6.0, 6.01, 0.05):
        test_vals.add(fp16_bits(x))

    # Dense sweep near zero [-0.5, 0.5] at finer step
    for x in np.arange(-0.5, 0.51, 0.005):
        test_vals.add(fp16_bits(x))

    # Near each breakpoint (both positive and negative)
    for b in breaks:
        for delta in [-0.01, -0.001, 0, 0.001, 0.01]:
            test_vals.add(fp16_bits(b + delta))
            test_vals.add(fp16_bits(-b + delta))

    # Edge cases
    test_vals.add(0x0000)  # +0
    test_vals.add(0x8000)  # -0
    test_vals.add(0x0001)  # smallest positive denormal
    test_vals.add(0x8001)  # smallest negative denormal
    test_vals.add(0x0400)  # smallest positive normal
    test_vals.add(0x8400)  # smallest negative normal
    test_vals.add(0x3C00)  # 1.0
    test_vals.add(0xBC00)  # -1.0
    test_vals.add(0x7C00)  # +inf
    test_vals.add(0xFC00)  # -inf
    test_vals.add(0x7E00)  # NaN

    # Sort for reproducibility
    test_vals = sorted(test_vals)

    with open(OUT, "w") as f:
        for bits in test_vals:
            f.write(f"{bits:04x}\n")

    print(f"Wrote {OUT} ({len(test_vals)} fp16 values)")
    # Show range
    floats = [fp16_float(b) for b in test_vals if (b & 0x7FFF) < 0x7C00]
    print(f"  Range: [{min(floats):.4f}, {max(floats):.4f}]")