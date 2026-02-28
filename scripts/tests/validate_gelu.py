"""
Validate tb_gelu xsim output against the GELU LUT hex file

Parses logs/tb_gelu.log, compares each xsim output byte against
the corresponding entry in gelu_lut.hex
"""

import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LUT_HEX = os.path.join(PROJ, "mem", "gelu_lut.hex")
LOG = os.path.join(PROJ, "logs", "tb_gelu.log")

N_LAYERS = 4


def load_lut(path):
    values = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            values.append(int(s, 16))
    return values


# Convert unsigned byte to signed int8
def to_signed8(b):
    return b - 256 if b >= 128 else b


"""
Parse tb_gelu.log for L=N IN=xx OUT=xx lines
Returns dict: layer -> list of (in_byte, out_byte)
"""
def parse_log(path):
    pat = re.compile(r"L=(\d+)\s+IN=([0-9a-fA-F]{2})\s+OUT=([0-9a-fA-F]{2})")
    results = {i: [] for i in range(N_LAYERS)}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                layer = int(m.group(1))
                in_byte = int(m.group(2), 16)
                out_byte = int(m.group(3), 16)
                results[layer].append((in_byte, out_byte))
    return results


if __name__ == "__main__":
    if not os.path.exists(LUT_HEX):
        print(f"LUT not found: {LUT_HEX}")
        print("Run: python3 scripts/gen_gelu_lut.py")
        sys.exit(1)

    lut = load_lut(LUT_HEX)
    print(f"Loaded LUT: {len(lut)} entries from {LUT_HEX}")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_gelu simulation first (Tcl: 'run all')")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if all(len(v) == 0 for v in xsim.values()):
        print("No L=N IN=xx OUT=xx lines found in log")
        print("Set tb_gelu as sim top, then 'run all'")
        sys.exit(1)

    total_errors = 0

    for layer in range(N_LAYERS):
        entries = xsim[layer]
        if len(entries) == 0:
            print(f"Layer {layer}: no data")
            total_errors += 1
            continue

        errors = 0
        max_abs_delta = 0
        sum_abs_delta = 0

        print(f"=== Layer {layer} ===")
        print(f"{'in':>5s}  {'xsim':>5s}  {'golden':>6s}  {'delta':>6s}  {'status'}")
        print("-" * 38)

        for in_byte, out_byte in entries:
            in_signed = to_signed8(in_byte)
            out_signed = to_signed8(out_byte)

            gold_byte = lut[layer * 256 + in_byte]
            gold_signed = to_signed8(gold_byte)
            delta = out_signed - gold_signed

            max_abs_delta = max(max_abs_delta, abs(delta))
            sum_abs_delta += abs(delta)

            if out_byte != gold_byte:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            print(f"{in_signed:5d}  {out_signed:5d}  {gold_signed:6d}  {delta:+6d}  {status}")

        n = len(entries)
        print()
        print(f"  Golden match:     {n - errors}/{n}")
        print(f"  Max abs delta:    {max_abs_delta}")
        print(f"  Mean abs delta:   {sum_abs_delta / n:.3f}")
        total_errors += errors
        print()

    if total_errors == 0:
        print(f"PASSED - all 1024 outputs match golden LUT")
    else:
        print(f"FAILED - {total_errors} mismatches vs golden LUT")
    sys.exit(0 if total_errors == 0 else 1)