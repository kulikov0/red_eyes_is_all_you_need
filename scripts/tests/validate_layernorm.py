"""
Validate tb_layernorm xsim output against rtl model

Parses logs/tb_layernorm.log, computes expected fp16 LayerNorm using
bit-exact Python fp16 primitives that match RTL combinational modules
"""

import re
import os
import sys

from rtl_ops import (
    DIM, LN_OFFSETS,
    to_signed8, load_hex, load_lut16, parse_scale_bits,
    fp16_from_int, fp16_mul, fp16_to_float,
    rtl_layernorm_fp16,
)
from ideal_ops import ideal_layernorm_fp16

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LN_HEX = os.path.join(PROJ, "mem", "ln_params.hex")
INPUT_HEX = os.path.join(PROJ, "mem", "ln_test_inputs.hex")
SCALES_VH = os.path.join(PROJ, "rtl", "weight_scales.vh")
LOG = os.path.join(PROJ, "logs", "tb_layernorm.log")

# Test definitions matching tb_layernorm.v
TESTS = [
    {"name": "embed(0,0) block0_ln1", "gamma_sel": 2,
     "gamma_scale": "SCALE_BLOCK0_LN1_WEIGHT", "beta_scale": "SCALE_BLOCK0_LN1_BIAS"},
    {"name": "embed(42,10) block0_ln2", "gamma_sel": 6,
     "gamma_scale": "SCALE_BLOCK0_LN2_WEIGHT", "beta_scale": "SCALE_BLOCK0_LN2_BIAS"},
    {"name": "embed(200,100) block1_ln1", "gamma_sel": 10,
     "gamma_scale": "SCALE_BLOCK1_LN1_WEIGHT", "beta_scale": "SCALE_BLOCK1_LN1_BIAS"},
    {"name": "large values block2_ln1", "gamma_sel": 18,
     "gamma_scale": "SCALE_BLOCK2_LN1_WEIGHT", "beta_scale": "SCALE_BLOCK2_LN1_BIAS"},
    {"name": "near-constant block2_ln2", "gamma_sel": 22,
     "gamma_scale": "SCALE_BLOCK2_LN2_WEIGHT", "beta_scale": "SCALE_BLOCK2_LN2_BIAS"},
    {"name": "mixed outliers block3_ln1", "gamma_sel": 26,
     "gamma_scale": "SCALE_BLOCK3_LN1_WEIGHT", "beta_scale": "SCALE_BLOCK3_LN1_BIAS"},
    {"name": "all negative block3_ln2", "gamma_sel": 30,
     "gamma_scale": "SCALE_BLOCK3_LN2_WEIGHT", "beta_scale": "SCALE_BLOCK3_LN2_BIAS"},
    {"name": "sparse ln_f", "gamma_sel": 34,
     "gamma_scale": "SCALE_LN_F_WEIGHT", "beta_scale": "SCALE_LN_F_BIAS"},
]


# Dequant int8 byte to fp16 bit pattern: fp16_from_int(signed(byte)) * scale
def dequant_byte(byte, scale_bits):
    return fp16_mul(fp16_from_int(to_signed8(byte)), scale_bits)


# Parse log: T=N I=N OUT=xxxx
def parse_log(path):
    pat = re.compile(r"T=(\d+)\s+I=(\d+)\s+OUT=([0-9a-fA-F]{4})")
    results = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                t = int(m.group(1))
                idx = int(m.group(2))
                out = int(m.group(3), 16)
                if t not in results:
                    results[t] = {}
                results[t][idx] = out
    return results


if __name__ == "__main__":
    for path, name in [(LN_HEX, "ln_params"), (INPUT_HEX, "ln_test_inputs")]:
        if not os.path.exists(path):
            print(f"{name} not found: {path}")
            sys.exit(1)

    isqrt_hex = os.path.join(PROJ, "mem", "inv_sqrt_lut.hex")
    if not os.path.exists(isqrt_hex):
        print(f"inv_sqrt LUT not found: {isqrt_hex}")
        sys.exit(1)

    ln_mem = load_hex(LN_HEX)
    input_mem = load_hex(INPUT_HEX)
    isqrt_lut = load_lut16(isqrt_hex)
    print(f"Loaded LN params: {len(ln_mem)} bytes")
    print(f"Loaded test inputs: {len(input_mem)} fp16 values")
    print(f"Loaded inv_sqrt LUT: {len(isqrt_lut)} entries")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_layernorm simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if not xsim:
        print("No output lines found in log")
        sys.exit(1)

    total_errors = 0
    total_count = 0

    for t_idx, test in enumerate(TESTS):
        data = xsim.get(t_idx, {})
        if not data:
            print(f"Test {t_idx} ({test['name']}): no data")
            total_errors += DIM
            total_count += DIM
            continue

        # Load test input fp16 values (as bit patterns)
        base = t_idx * DIM
        x_fp16 = [input_mem[base + i] for i in range(DIM)]

        # Load and dequant gamma/beta using bit-exact fp16 arithmetic
        gamma_sel = test["gamma_sel"]
        gamma_off = LN_OFFSETS[gamma_sel]
        beta_off = LN_OFFSETS[gamma_sel + 1]
        gamma_scale_bits = parse_scale_bits(test["gamma_scale"], SCALES_VH)
        beta_scale_bits = parse_scale_bits(test["beta_scale"], SCALES_VH)

        gamma_fp16 = [dequant_byte(ln_mem[gamma_off + i], gamma_scale_bits) for i in range(DIM)]
        beta_fp16 = [dequant_byte(ln_mem[beta_off + i], beta_scale_bits) for i in range(DIM)]

        # Run models
        rtl, g_mean, g_var, g_inv_std = rtl_layernorm_fp16(x_fp16, gamma_fp16, beta_fp16, isqrt_lut)
        ideal = ideal_layernorm_fp16(x_fp16, gamma_fp16, beta_fp16)

        errors = 0
        max_gi_delta = 0.0
        sum_gi_delta = 0.0
        max_xi_delta = 0.0
        sum_xi_delta = 0.0
        n_delta = 0

        print(f"Test {t_idx}: {test['name']} (gamma_sel={gamma_sel})")
        print(f"  rtl: mean={g_mean:04x} var={g_var:04x} inv_std={g_inv_std:04x}")
        print(f"{'idx':>5s}  {'xsim':>6s}  {'rtl':>6s}  {'ideal':>10s}  {'g-i delta':>10s}  {'x-i delta':>10s}  {'status'}")

        for i in range(DIM):
            if i not in data:
                print(f"{i:5d}  {'?':>6s}  {'?':>6s}  {'?':>10s}  {'?':>10s}  {'?':>10s}  MISSING")
                errors += 1
                continue

            xsim_bits = data[i]
            rtl_bits = rtl[i]
            ideal_val = ideal[i]

            match = (xsim_bits == rtl_bits)
            if not match:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            rtl_f = fp16_to_float(rtl_bits)
            xsim_f = fp16_to_float(xsim_bits)

            # Skip special values for delta stats
            xsim_special = (xsim_bits & 0x7FFF) >= 0x7C00
            rtl_special = (rtl_bits & 0x7FFF) >= 0x7C00
            if not xsim_special and not rtl_special:
                gi_delta = rtl_f - ideal_val
                xi_delta = xsim_f - ideal_val
                max_gi_delta = max(max_gi_delta, abs(gi_delta))
                sum_gi_delta += abs(gi_delta)
                max_xi_delta = max(max_xi_delta, abs(xi_delta))
                sum_xi_delta += abs(xi_delta)
                n_delta += 1
                print(f"{i:5d}  {xsim_bits:04x}    {rtl_bits:04x}    {ideal_val:10.4f}  {gi_delta:+10.4f}  {xi_delta:+10.4f}  {status}")
            else:
                print(f"{i:5d}  {xsim_bits:04x}    {rtl_bits:04x}    {'---':>10s}  {'---':>10s}  {'---':>10s}  {status}")

        n = DIM
        print()
        print(f"  RTL match:       {n - errors}/{n}")
        if n_delta > 0:
            print(f"  RTL vs ideal:    max={max_gi_delta:.6f}  mean={sum_gi_delta / n_delta:.6f}")
            print(f"  Xsim vs ideal:      max={max_xi_delta:.6f}  mean={sum_xi_delta / n_delta:.6f}")
        total_errors += errors
        total_count += n
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)