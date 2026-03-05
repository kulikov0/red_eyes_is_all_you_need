"""
Validate tb_gelu xsim output against rtl PWL model

Parses logs/tb_gelu.log, computes expected fp16 GELU using
bit-exact Python fp16 PWL that matches RTL gelu.v
"""

import re
import os
import sys

from rtl_ops import (
    fp16_to_float,
    load_gelu_pwl, rtl_gelu_fp16,
)
from ideal_ops import ideal_gelu_fp16

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(PROJ, "logs", "tb_gelu.log")


def parse_log(path):
    pat = re.compile(r"IN=([0-9a-fA-F]{4})\s+OUT=([0-9a-fA-F]{4})")
    results = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                results.append((int(m.group(1), 16), int(m.group(2), 16)))
    return results


if __name__ == "__main__":
    if not os.path.exists(LOG):
        print(f"Log not found: {LOG}")
        print("Run tb_gelu simulation first")
        sys.exit(1)

    breaks, slopes, icepts = load_gelu_pwl()
    print(f"Loaded PWL coefficients: {len(breaks)} breaks, {len(slopes)} segments")

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)
    if not xsim:
        print("No output lines found in log")
        sys.exit(1)

    print(f"{'idx':>5s}  {'input':>6s}  {'xsim':>6s}  {'rtl':>6s}"
          f"  {'ideal':>10s}  {'g-i delta':>10s}  {'x-i delta':>10s}  {'status'}")

    errors = 0
    max_gi_delta = 0.0
    sum_gi_delta = 0.0
    max_xi_delta = 0.0
    sum_xi_delta = 0.0
    n_delta = 0

    for idx, (in_bits, xsim_bits) in enumerate(xsim):
        rtl_bits = rtl_gelu_fp16(in_bits, breaks, slopes, icepts)
        ideal_val = ideal_gelu_fp16(in_bits)

        match = (xsim_bits == rtl_bits)
        if not match:
            errors += 1
            status = "MISMATCH"
        else:
            status = "OK"

        xsim_special = (xsim_bits & 0x7FFF) >= 0x7C00
        rtl_special = (rtl_bits & 0x7FFF) >= 0x7C00
        in_special = (in_bits & 0x7FFF) >= 0x7C00

        if not xsim_special and not rtl_special and not in_special:
            xsim_f = fp16_to_float(xsim_bits)
            rtl_f = fp16_to_float(rtl_bits)
            gi_delta = rtl_f - ideal_val
            xi_delta = xsim_f - ideal_val
            max_gi_delta = max(max_gi_delta, abs(gi_delta))
            sum_gi_delta += abs(gi_delta)
            max_xi_delta = max(max_xi_delta, abs(xi_delta))
            sum_xi_delta += abs(xi_delta)
            n_delta += 1
            print(f"{idx:5d}  {in_bits:04x}    {xsim_bits:04x}    {rtl_bits:04x}"
                  f"    {ideal_val:10.4f}  {gi_delta:+10.4f}  {xi_delta:+10.4f}  {status}")
        else:
            print(f"{idx:5d}  {in_bits:04x}    {xsim_bits:04x}    {rtl_bits:04x}"
                  f"    {'---':>10s}  {'---':>10s}  {'---':>10s}  {status}")

    n = len(xsim)
    print()
    print(f"  RTL match:       {n - errors}/{n}")
    if n_delta > 0:
        print(f"  RTL vs ideal:    max={max_gi_delta:.6f}  mean={sum_gi_delta / n_delta:.6f}")
        print(f"  Xsim vs ideal:      max={max_xi_delta:.6f}  mean={sum_xi_delta / n_delta:.6f}")
    print()

    if errors == 0:
        print(f"PASSED - all {n} outputs match rtl model")
    else:
        print(f"FAILED - {errors} mismatches vs rtl model")
    sys.exit(0 if errors == 0 else 1)