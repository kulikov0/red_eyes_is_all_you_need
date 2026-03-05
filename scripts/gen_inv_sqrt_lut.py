"""
Generate inv_sqrt_lut.hex for the inv_sqrt module.

Algorithm (LOD-LUT-Shift from https://www.mdpi.com/2072-666X/17/1/84):

    1/sqrt(d) = LUT[k_lsb, mantissa] >> floor(k/2)

where k = leading one position, mantissa = 8 bits below leading one

LUT has 512 entries (9-bit address = {k[0], mantissa[7:0]}):
  k[0]=0 (even k): stores round(1/sqrt(1 + m/256) * 32768)
  k[0]=1 (odd  k): stores round(1/sqrt(2 * (1 + m/256)) * 32768)

Output is Q1.15 unsigned fixed-point (1 integer bit, 15 fractional bits)
"""

import math
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM = os.path.join(PROJ, "mem")

# Generate 512-entry LUT for inv_sqrt
def generate_lut():
    entries = []
    for i in range(512):
        k_lsb = (i >> 8) & 1
        m = i & 0xFF

        normalized = 1.0 + m / 256.0

        if k_lsb == 0:
            val = 1.0 / math.sqrt(normalized)
        else:
            val = 1.0 / math.sqrt(2.0 * normalized)

        q = round(val * 32768)
        q = min(q, 65535)
        entries.append(q)
    return entries

def main():
    os.makedirs(MEM, exist_ok=True)

    lut = generate_lut()

    hex_path = os.path.join(MEM, "inv_sqrt_lut.hex")
    with open(hex_path, "w") as f:
        for val in lut:
            f.write(f"{val:04x}\n")
    print(f"Wrote {hex_path} ({len(lut)} entries)")


if __name__ == "__main__":
    main()