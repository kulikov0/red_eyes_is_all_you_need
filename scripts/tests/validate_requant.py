"""
Validate tb_requant xsim output against Python golden model

Parses logs/tb_requant.log, computes expected clamp((acc*scale)>>>shift, -128, 127)
in Python, compares byte-exact
"""

import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(PROJ, "logs", "tb_requant.log")

# DUT configurations matching tb_requant.v
DUTS = {
    "A": {"acc_w": 24, "scale_w": 16, "shift": 22},
    "B": {"acc_w": 19, "scale_w": 16, "shift": 19},
}


# Convert unsigned byte to signed int8
def to_signed8(b):
    return b - 256 if b >= 128 else b


# Golden model: (acc * scale) >>> shift, clamp to int8
def golden_requant(acc, scale, shift, acc_w, scale_w):
    prod_w = acc_w + scale_w
    # Signed * unsigned multiply (sign-extend scale to signed)
    product = acc * scale
    # Mask to prod_w+1 bits to handle sign correctly
    mask = (1 << prod_w) - 1
    product = product & mask
    if product >= (1 << (prod_w - 1)):
        product -= (1 << prod_w)
    # Arithmetic right shift
    shifted = product >> shift
    # Clamp to int8
    clamped = max(-128, min(127, shifted))
    return clamped & 0xFF


# Parse log: T=N DUT=X ACC=value SCALE=value OUT=xx
def parse_log(path):
    pat = re.compile(
        r"T=(\d+)\s+DUT=([AB])\s+ACC=(-?\d+)\s+SCALE=(\d+)\s+OUT=([0-9a-fA-F]{2})")
    results = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                results.append({
                    "t": int(m.group(1)),
                    "dut": m.group(2),
                    "acc": int(m.group(3)),
                    "scale": int(m.group(4)),
                    "out": int(m.group(5), 16),
                })
    return results


if __name__ == "__main__":
    if not os.path.exists(LOG):
        print(f"Log not found: {LOG}")
        print("Run tb_requant simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    entries = parse_log(LOG)

    if not entries:
        print("No T=N DUT=X ACC=n SCALE=n OUT=xx lines found in log")
        sys.exit(1)

    # Group by DUT
    by_dut = {}
    for e in entries:
        dut = e["dut"]
        if dut not in by_dut:
            by_dut[dut] = []
        by_dut[dut].append(e)

    total_errors = 0
    total_count = 0

    for dut_name in sorted(by_dut.keys()):
        cfg = DUTS[dut_name]
        tests = by_dut[dut_name]

        print(f"=== DUT {dut_name}: ACC_W={cfg['acc_w']} SHIFT={cfg['shift']} ===")
        print(f"{'idx':>5s}  {'xsim':>5s}  {'golden':>6s}  {'delta':>6s}  {'status'}")

        errors = 0
        max_abs_delta = 0
        sum_abs_delta = 0

        for i, e in enumerate(tests):
            xsim_byte = e["out"]
            gold_byte = golden_requant(
                e["acc"], e["scale"], cfg["shift"],
                cfg["acc_w"], cfg["scale_w"])

            xsim_s = to_signed8(xsim_byte)
            gold_s = to_signed8(gold_byte)
            delta = xsim_s - gold_s

            max_abs_delta = max(max_abs_delta, abs(delta))
            sum_abs_delta += abs(delta)

            if xsim_byte != gold_byte:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            print(f"{i:5d}  {xsim_s:5d}  {gold_s:6d}  {delta:+6d}  {status}")

        n = len(tests)
        print()
        print(f"  Golden match:     {n - errors}/{n}")
        print(f"  Max abs delta:    {max_abs_delta}")
        print(f"  Mean abs delta:   {sum_abs_delta / n:.3f}")
        total_errors += errors
        total_count += n
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match golden model")
    else:
        print(f"FAILED - {total_errors} mismatches vs golden model")
    sys.exit(0 if total_errors == 0 else 1)