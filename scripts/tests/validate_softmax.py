"""
Validate tb_softmax xsim output against Python golden model

Parses logs/tb_softmax.log, recomputes softmax through the same
bipartite LUT + LOD algorithm used by the hardware, and prints deltas
from the ideal floating-point result
"""

import math
import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LUT0_HEX = os.path.join(PROJ, "mem", "exp_lut0.hex")
LUT1_HEX = os.path.join(PROJ, "mem", "exp_lut1.hex")
LOG = os.path.join(PROJ, "logs", "tb_softmax.log")

FRAC_W = 7
D_CLIP = 2048  # 16.0 in Q4.7
LN2_Q7 = 89    # round(ln(2) * 128)

# ln(1 + s/16) * 128 LUT (matches RTL)
LN1PS_LUT = [0, 8, 15, 22, 29, 35, 41, 47, 53, 58, 63, 68, 73, 78, 82, 87]


# Load 256-entry hex LUT
def load_lut(path, signed=False):
    entries = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            val = int(s, 16)
            if signed and val >= 32768:
                val -= 65536
            entries.append(val)
    return entries


# Bipartite exp(-d) approximation (matches RTL)
def bipartite_exp(d_q47, lut0, lut1):
    if d_q47 >= D_CLIP:
        return 0
    x0 = (d_q47 >> 6) & 0x1F
    x1 = (d_q47 >> 3) & 0x07
    x2 = d_q47 & 0x07
    val = lut0[(x0 << 3) | x1] + lut1[(x0 << 3) | x2]
    if val < 0:
        return 0
    if val > 32768:
        return 32768
    return val


# LOD: find leading one position in a 24-bit value
def lod24(val):
    if val == 0:
        return 0
    k = 0
    for i in range(24):
        if val & (1 << i):
            k = i
    return k


# Extract 4-bit mantissa below leading one
def lod_mantissa(val, k):
    if k >= 4:
        return (val >> (k - 4)) & 0xF
    else:
        return (val << (4 - k)) & 0xF


# Full hardware softmax simulation (matches RTL pipeline)
def golden_softmax(inputs, lut0, lut1):
    max_val = max(inputs)

    # Phase 2: EXP_ACC
    exp_vals = []
    sum_acc = 0
    for x in inputs:
        d_raw = max_val - x
        d_int = d_raw >> FRAC_W
        d_frac = d_raw & ((1 << FRAC_W) - 1)
        if d_int >= 16:
            d_q47 = D_CLIP
        else:
            d_q47 = (d_int << FRAC_W) | d_frac
        ev = bipartite_exp(d_q47, lut0, lut1)
        exp_vals.append(ev)
        sum_acc += ev
    sum_acc = min(sum_acc, 0xFFFFFF)

    # Phase 3: LN_SUM
    if sum_acc == 0:
        ln_offset = D_CLIP
    else:
        k = lod24(sum_acc)
        s = lod_mantissa(sum_acc, k)
        k_minus_15 = k - 15
        kln2 = k_minus_15 * LN2_Q7
        ln1ps = LN1PS_LUT[s]
        ln_raw = kln2 + ln1ps
        if ln_raw < 0:
            ln_offset = 0
        elif ln_raw >= D_CLIP:
            ln_offset = D_CLIP
        else:
            ln_offset = ln_raw

    # Phase 4: NORM
    outputs = []
    for x in inputs:
        d_raw = max_val - x
        d_int = d_raw >> FRAC_W
        d_frac = d_raw & ((1 << FRAC_W) - 1)
        d_overflow = d_int >= 16
        if d_overflow:
            d_q47 = D_CLIP
        else:
            d_q47 = (d_int << FRAC_W) | d_frac
        d_plus_ln = d_q47 + ln_offset
        if d_overflow or d_plus_ln >= D_CLIP:
            d_norm = D_CLIP
        else:
            d_norm = d_plus_ln
        out = bipartite_exp(d_norm, lut0, lut1)
        outputs.append(out)

    return outputs


def parse_log(path):
    test_pat = re.compile(r"=== Test (\d+): (.+?) ===")
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

    golden = golden_softmax(inputs, lut0, lut1)
    errors = 0
    max_abs_err = 0
    sum_abs_err = 0

    # Ideal float softmax
    floats = [x / (1 << FRAC_W) for x in inputs]
    max_f = max(floats)
    exps = [math.exp(f - max_f) for f in floats]
    s = sum(exps)
    ideal_f = [e / s for e in exps]

    print(f"  {'idx':>5s}  {'input':>7s}  {'xsim':>6s}  {'golden':>6s}  {'ideal':>8s}  {'xsim_f':>8s}  {'delta':>6s}  {'err%':>7s}  {'status'}")
    print(f"  {'-'*72}")

    for j in range(n):
        xv = xsim_out[j]
        gv = golden[j]
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

        # Print first 4, last 2, any non-zero, and any mismatches
        show = (j < 4 or j >= n - 2 or xv != 0 or status == "MISMATCH")
        if show:
            print(f"  {j:5d}  {inputs[j]:7d}  {xv:6d}  {gv:6d}  {ideal_f[j]:8.5f}  {xsim_f:8.5f}  {delta:+6d}  {err_pct:6.2f}%  {status}")
        elif j == 4:
            print(f"  {'...':>5s}")

    print(f"\n  Golden match:     {n - errors}/{n}")
    print(f"  Max abs delta:    {max_abs_err:.6f} (vs ideal float)")
    print(f"  Mean abs delta:   {sum_abs_err / n:.6f}")
    print(f"  Output sum:       {test['sum']} (ideal 32768, delta {test['sum'] - 32768:+d})")
    return errors


if __name__ == "__main__":
    if not os.path.exists(LUT0_HEX) or not os.path.exists(LUT1_HEX):
        print(f"LUT files not found in mem/")
        print("Run: python3 scripts/gen_softmax_luts.py")
        sys.exit(1)

    lut0 = load_lut(LUT0_HEX, signed=False)
    lut1 = load_lut(LUT1_HEX, signed=True)
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
        print(f"PASSED - all xsim outputs match golden model")
    else:
        print(f"FAILED - {total_errors} total mismatches vs golden model")
    sys.exit(0 if total_errors == 0 else 1)