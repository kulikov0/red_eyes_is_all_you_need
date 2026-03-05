"""
Validate tb_softmax xsim output against Python rtl model

Parses logs/tb_softmax.log, recomputes softmax through the same
bipartite LUT + LOD algorithm used by the hardware, and prints deltas
from the ideal floating-point result
"""

import re
import os
import sys

from rtl_ops import load_lut16, rtl_softmax
from ideal_ops import ideal_softmax

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LUT0_HEX = os.path.join(PROJ, "mem", "exp_lut0.hex")
LUT1_HEX = os.path.join(PROJ, "mem", "exp_lut1.hex")
LOG = os.path.join(PROJ, "logs", "tb_softmax.log")


def parse_log(path):
    test_pat = re.compile(r"Test (\d+): (.+)")
    out_pat = re.compile(r"OUT\[(\d+)\] input=(-?\d+) output=(\d+)")
    sum_pat = re.compile(r"SUM=(\d+)")

    tests = {}
    current_test = None
    with open(path) as f:
        for line in f:
            m = test_pat.search(line)
            if m:
                current_test = int(m.group(1))
                tests[current_test] = {
                    "name": m.group(2),
                    "inputs": [],
                    "outputs": [],
                    "sum": 0,
                }
                continue
            if current_test is None:
                continue
            m = out_pat.search(line)
            if m:
                tests[current_test]["inputs"].append(int(m.group(2)))
                tests[current_test]["outputs"].append(int(m.group(3)))
                continue
            m = sum_pat.search(line)
            if m:
                tests[current_test]["sum"] = int(m.group(1))
    return tests


def analyze_test(test, lut0, lut1):
    inputs = test["inputs"]
    xsim_out = test["outputs"]
    n = len(inputs)
    if n == 0:
        print("  No data")
        return 0

    rtl = rtl_softmax(inputs, lut0, lut1)
    errors = 0
    max_abs_err = 0
    sum_abs_err = 0

    ideal_f = ideal_softmax(inputs)

    print(f"  {'idx':>5s}  {'input':>7s}  {'xsim':>6s}  {'rtl':>6s}  {'ideal':>8s}  {'xsim_f':>8s}  {'delta':>6s}  {'err%':>7s}  {'status'}")

    for j in range(n):
        xv = xsim_out[j]
        gv = rtl[j]
        delta = xv - gv
        xsim_f = xv / 32768.0

        if ideal_f[j] > 1e-9:
            err_pct = abs(xsim_f - ideal_f[j]) / ideal_f[j] * 100
        else:
            err_pct = 0.0 if xv == 0 else float('inf')

        abs_err = abs(xsim_f - ideal_f[j])
        max_abs_err = max(max_abs_err, abs_err)
        sum_abs_err += abs_err

        if xv != gv:
            status = "MISMATCH"
            errors += 1
        else:
            status = "OK"

        print(f"  {j:5d}  {inputs[j]:7d}  {xv:6d}  {gv:6d}  {ideal_f[j]:8.5f}  {xsim_f:8.5f}  {delta:+6d}  {err_pct:6.2f}%  {status}")

    print(f"\n  RTL match:     {n - errors}/{n}")
    print(f"  Max abs delta:    {max_abs_err:.6f} (vs ideal float)")
    print(f"  Mean abs delta:   {sum_abs_err / n:.6f}")
    print(f"  Output sum:       {test['sum']} (ideal 32768, delta {test['sum'] - 32768:+d})")
    return errors


if __name__ == "__main__":
    if not os.path.exists(LUT0_HEX) or not os.path.exists(LUT1_HEX):
        print(f"LUT files not found in mem/")
        print("Run: python3 scripts/gen_softmax_luts.py")
        sys.exit(1)

    lut0 = load_lut16(LUT0_HEX, signed=False)
    lut1 = load_lut16(LUT1_HEX, signed=True)
    print(f"Loaded LUT0: {len(lut0)} entries, LUT1: {len(lut1)} entries")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_softmax simulation first (Tcl: 'run all')")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    tests = parse_log(LOG)

    if not tests:
        print("No test data found in log")
        print("Set tb_softmax as sim top, then 'run all'")
        sys.exit(1)

    total_errors = 0
    for tnum in sorted(tests.keys()):
        test = tests[tnum]
        print(f"Test {tnum}: {test['name']}")
        total_errors += analyze_test(test, lut0, lut1)
        print()

    if total_errors == 0:
        print(f"PASSED - all xsim outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} total mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)