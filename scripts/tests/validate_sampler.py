"""Validate tb_sampler xsim output against rtl_sample_token golden"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_ops import rtl_sample_token, load_lut16
from gen_sampler_test_vectors import TESTS

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(PROJ, "logs", "tb_sampler.log")
LUT0_HEX = os.path.join(PROJ, "mem", "exp_lut0.hex")
LUT1_HEX = os.path.join(PROJ, "mem", "exp_lut1.hex")


def parse_log(path):
    head = re.compile(r"Test (\d+):")
    line = re.compile(
        r"TOKEN=(\d+) INV_TEMP=([0-9a-fA-F]+) TOPK=(\d+) SEED=([0-9a-fA-F]+)"
    )
    tests = {}
    cur = None
    with open(path) as f:
        for ln in f:
            m = head.search(ln)
            if m:
                cur = int(m.group(1))
                continue
            if cur is None:
                continue
            m = line.search(ln)
            if m:
                tests[cur] = {
                    "token": int(m.group(1)),
                    "inv_temp": int(m.group(2), 16),
                    "top_k": int(m.group(3)),
                    "seed": int(m.group(4), 16),
                }
                cur = None
    return tests


if __name__ == "__main__":
    if not os.path.exists(LOG):
        print(f"Log not found: {LOG}")
        sys.exit(1)

    lut0 = load_lut16(LUT0_HEX, signed=False)
    lut1 = load_lut16(LUT1_HEX, signed=True)

    parsed = parse_log(LOG)
    n_tests = len(TESTS)
    total_mismatches = 0

    for ti, t in enumerate(TESTS):
        params = (
            f"name={t['name']} inv_temp={t['inv_temp_bits']:04x} "
            f"top_k={t['top_k']} seed={t['seed']:04x}"
        )
        print(f"Test {ti}: {params}")
        print(f"  {'idx':>4s}  {'xsim':>5s}  {'golden':>6s}  {'delta':>5s}  status")

        if ti not in parsed:
            print("  no log entry for this test")
            total_mismatches += 1
            print()
            continue

        xsim_tok = parsed[ti]["token"]
        golden_tok, _ = rtl_sample_token(
            t["logits"], t["inv_temp_bits"], t["top_k"], t["seed"], lut0, lut1
        )

        delta = xsim_tok - golden_tok
        status = "OK" if delta == 0 else "MISMATCH"
        sign = "+" if delta >= 0 else "-"
        print(f"  {0:>4d}  {xsim_tok:>5d}  {golden_tok:>6d}  {sign}{abs(delta):>4d}  {status}")
        match = 0 if delta else 1
        print(f"  Golden match: {match}/1")
        print(f"  Max abs delta: {abs(delta)}")
        print(f"  Mean abs delta: {float(abs(delta)):.3f}")
        if delta:
            total_mismatches += 1
        print()

    if total_mismatches == 0:
        print(f"PASSED - all {n_tests} outputs match golden model")
        sys.exit(0)
    print(f"FAILED - {total_mismatches} mismatches vs golden model")
    sys.exit(1)