"""
Validate tb_embedding xsim output against rtl model

Parses logs/tb_embedding.log, computes expected
fp16(tok_int8) * tok_scale + fp16(pos_int8) * pos_scale
using bit-exact rtl model, compares bit-exact. Also shows ideal float64 reference
"""

import re
import os
import sys

from rtl_ops import (
    DIM, MEM,
    to_signed8, load_hex, parse_scale_bits,
    fp16_to_float, fp16_from_int, fp16_mul, fp16_add,
    rtl_embedding_fp16,
)

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK_HEX = os.path.join(MEM, "tok_emb_weight.hex")
POS_HEX = os.path.join(MEM, "pos_emb_weight.hex")
LOG = os.path.join(PROJ, "logs", "tb_embedding.log")


# Ideal: dequant each to fp16 (same as RTL), then add in float64
def ideal_embedding_element(tok_byte, pos_byte, tok_scale_bits, pos_scale_bits):
    tok_dq = fp16_to_float(fp16_mul(fp16_from_int(to_signed8(tok_byte)), tok_scale_bits))
    pos_dq = fp16_to_float(fp16_mul(fp16_from_int(to_signed8(pos_byte)), pos_scale_bits))
    return tok_dq + pos_dq


# Parse log: T=N TOK=n POS=n I=n OUT=xxxx (4-digit fp16 hex)
def parse_log(path):
    pat = re.compile(
        r"T=(\d+)\s+TOK=(\d+)\s+POS=(\d+)\s+I=(\d+)\s+OUT=([0-9a-fA-F]{4})")
    results = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                t = int(m.group(1))
                tok = int(m.group(2))
                pos = int(m.group(3))
                idx = int(m.group(4))
                out = int(m.group(5), 16)
                if t not in results:
                    results[t] = (tok, pos, {})
                results[t][2][idx] = out
    return results


if __name__ == "__main__":
    for path, name in [(TOK_HEX, "tok_emb"), (POS_HEX, "pos_emb")]:
        if not os.path.exists(path):
            print(f"{name} not found: {path}")
            sys.exit(1)

    tok_mem = load_hex(TOK_HEX)
    pos_mem = load_hex(POS_HEX)
    print(f"Loaded tok_emb: {len(tok_mem)} bytes")
    print(f"Loaded pos_emb: {len(pos_mem)} bytes")

    tok_scale = parse_scale_bits("SCALE_TOK_EMB_WEIGHT")
    pos_scale = parse_scale_bits("SCALE_POS_EMB_WEIGHT")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_embedding simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if len(xsim) == 0:
        print("No output lines found in log")
        sys.exit(1)

    total_errors = 0
    total_count = 0

    for test_num in sorted(xsim.keys()):
        tok, pos, data = xsim[test_num]
        rtl_vec = rtl_embedding_fp16(tok, pos, tok_mem, pos_mem,
                                            tok_scale, pos_scale)
        errors = 0
        max_gi_delta = 0.0
        sum_gi_delta = 0.0
        max_xi_delta = 0.0
        sum_xi_delta = 0.0
        tok_base = tok * DIM
        pos_base = pos * DIM

        print(f"Test {test_num}: tok={tok} pos={pos}")
        print(f"{'idx':>5s}  {'xsim':>6s}  {'rtl':>6s}  {'ideal':>10s}"
              f"  {'g-i delta':>10s}  {'x-i delta':>10s}  {'status'}")

        for i in range(DIM):
            if i not in data:
                print(f"{i:5d}  {'?':>6s}  {'?':>6s}  {'?':>10s}"
                      f"  {'?':>10s}  {'?':>10s}  MISSING")
                errors += 1
                continue

            xsim_bits = data[i]
            rtl_bits = rtl_vec[i]
            ideal = ideal_embedding_element(
                tok_mem[tok_base + i], pos_mem[pos_base + i],
                tok_scale, pos_scale)

            match = (xsim_bits == rtl_bits)
            if not match:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            rtl_f = fp16_to_float(rtl_bits)
            xsim_f = fp16_to_float(xsim_bits)
            gi_delta = rtl_f - ideal
            xi_delta = xsim_f - ideal

            max_gi_delta = max(max_gi_delta, abs(gi_delta))
            sum_gi_delta += abs(gi_delta)
            max_xi_delta = max(max_xi_delta, abs(xi_delta))
            sum_xi_delta += abs(xi_delta)

            print(f"{i:5d}  {xsim_bits:04x}    {rtl_bits:04x}    {ideal:10.4f}"
                  f"  {gi_delta:+10.4f}  {xi_delta:+10.4f}  {status}")

        n = DIM
        print()
        print(f"  RTL match:       {n - errors}/{n}")
        print(f"  RTL vs ideal:    max={max_gi_delta:.6f}  mean={sum_gi_delta / n:.6f}")
        print(f"  Xsim vs ideal:      max={max_xi_delta:.6f}  mean={sum_xi_delta / n:.6f}")
        total_errors += errors
        total_count += n
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)