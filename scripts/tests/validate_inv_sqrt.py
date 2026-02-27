"""
Validate tb_inv_sqrt xsim output against Python golden model.

Parses logs/tb_inv_sqrt.log, recomputes 1/sqrt(d) through the same
LOD-LUT-Shift algorithm used by the hardware, and prints deltas
from the ideal floating-point result.
"""

import math
import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEX = os.path.join(PROJ, "mem", "inv_sqrt_lut.hex")
LOG = os.path.join(PROJ, "logs", "tb_inv_sqrt.log")

D_W = 14


def load_lut(path):
    """Load 512-entry LUT from hex file."""
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def golden_inv_sqrt(d, lut):
    """Simulate the hardware inv_sqrt for a given input d."""
    if d == 0:
        return 0xFFFF

    # LOD
    k = 0
    for i in range(D_W):
        if d & (1 << i):
            k = i

    # Normalize and extract mantissa
    norm_shift = (D_W - 1) - k
    d_norm = (d << norm_shift) & ((1 << D_W) - 1)
    mantissa = (d_norm >> (D_W - 9)) & 0xFF

    # LUT lookup
    lut_addr = ((k & 1) << 8) | mantissa
    lut_out = lut[lut_addr]

    # Barrel shift
    result = lut_out >> (k >> 1)
    return result


# Parse tb_inv_sqrt.log for OK/FAIL lines.
def parse_log(path):
    ok_pat = re.compile(r"OK\s+d=(\d+)\s+result=(\d+)\s+\(0x([0-9a-fA-F]{4})\)")
    fail_pat = re.compile(r"FAIL\s+d=(\d+):\s+got\s+(\d+)\s+\(0x([0-9a-fA-F]{4})\),\s+expected\s+(\d+)")

    results = []
    with open(path) as f:
        for line in f:
            m = ok_pat.search(line)
            if m:
                results.append({
                    "d": int(m.group(1)),
                    "result": int(m.group(2)),
                    "ok": True,
                })
                continue
            m = fail_pat.search(line)
            if m:
                results.append({
                    "d": int(m.group(1)),
                    "result": int(m.group(2)),
                    "expected": int(m.group(4)),
                    "ok": False,
                })
    return results


if __name__ == "__main__":
    if not os.path.exists(HEX):
        print(f"LUT not found: {HEX}")
        print("Run: python3 scripts/gen_inv_sqrt_lut.py")
        sys.exit(1)

    lut = load_lut(HEX)
    print(f"Loaded LUT: {len(lut)} entries from {HEX}")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_inv_sqrt simulation first (Tcl: 'run all')")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if not xsim:
        print("No OK/FAIL lines found in log.")
        print("Set tb_inv_sqrt as sim top, then 'run all'")
        sys.exit(1)

    errors = 0
    print(f"{'d':>6s}  {'xsim':>6s}  {'golden':>6s}  {'ideal':>8s}  {'xsim_f':>8s}  {'delta':>8s}  {'err%':>6s}  {'status'}")
    print("-" * 72)

    for entry in xsim:
        d = entry["d"]
        xsim_val = entry["result"]
        gold_val = golden_inv_sqrt(d, lut)

        if d > 0:
            ideal = 1.0 / math.sqrt(d)
        else:
            ideal = float('inf')

        xsim_f = xsim_val / 32768.0
        delta = xsim_val - gold_val

        if d > 0:
            err_pct = abs(xsim_f - ideal) / ideal * 100
        else:
            err_pct = 0.0

        # Check xsim matches golden model
        if xsim_val != gold_val:
            status = "MISMATCH"
            errors += 1
        elif not entry["ok"]:
            status = "XSIM_FAIL"
            errors += 1
        else:
            status = "OK"

        print(f"{d:6d}  {xsim_val:6d}  {gold_val:6d}  {ideal:8.5f}  {xsim_f:8.5f}  {delta:+8d}  {err_pct:6.3f}%  {status}")

    print()
    if errors == 0:
        print(f"PASSED - all {len(xsim)} checks match golden model")
    else:
        print(f"FAILED - {errors} mismatches out of {len(xsim)} checks")
    sys.exit(0 if errors == 0 else 1)
