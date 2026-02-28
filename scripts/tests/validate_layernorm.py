"""
Validate tb_layernorm xsim output against Python golden model

Parses logs/tb_layernorm.log, recomputes LayerNorm through the same
int8 arithmetic used by the hardware (LOD-LUT-Shift inv_sqrt, fixed-point
multiply-shift chain), and also computes ideal float LayerNorm for
quantization error analysis
"""

import math
import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LN_HEX = os.path.join(PROJ, "mem", "ln_params.hex")
ISQRT_HEX = os.path.join(PROJ, "mem", "inv_sqrt_lut.hex")
LOG = os.path.join(PROJ, "logs", "tb_layernorm.log")

DIM = 128
ISQRT_DW = 17

# Offset into ln_params.hex for each tensor_sel (matches weight_store.v)
LN_OFFSETS = {
    2: 0, 3: 128, 6: 256, 7: 384,
    10: 512, 11: 640, 14: 768, 15: 896,
    18: 1024, 19: 1152, 22: 1280, 23: 1408,
    26: 1536, 27: 1664, 30: 1792, 31: 1920,
    34: 2048, 35: 2176,
}

# Test definitions matching tb_layernorm.v
TESTS = [
    {"name": "ramp", "gamma_sel": 2,
     "inputs": list(range(128))},
    {"name": "constant", "gamma_sel": 2,
     "inputs": [42] * 128},
    {"name": "signed ramp", "gamma_sel": 6,
     "inputs": list(range(-64, 64))},
    {"name": "alternating", "gamma_sel": 10,
     "inputs": [50 if i % 2 == 0 else -50 for i in range(128)]},
    {"name": "ramp with ln_f", "gamma_sel": 34,
     "inputs": list(range(128))},
]


def load_hex(path):
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


# Simulate hardware inv_sqrt (LOD-LUT-Shift) with D_W=17
def golden_inv_sqrt(d, lut):
    if d == 0:
        return 0xFFFF

    # LOD: find position of leading one
    k = 0
    for i in range(ISQRT_DW):
        if d & (1 << i):
            k = i

    # Normalize and extract 8-bit mantissa
    norm_shift = (ISQRT_DW - 1) - k
    d_norm = (d << norm_shift) & ((1 << ISQRT_DW) - 1)
    mantissa = (d_norm >> (ISQRT_DW - 9)) & 0xFF

    # LUT lookup: {k[0], mantissa}
    lut_addr = ((k & 1) << 8) | mantissa
    lut_out = lut[lut_addr]

    # Barrel shift: result = LUT >> floor(k/2)
    result = lut_out >> (k >> 1)
    return result


"""
Golden layernorm matching RTL arithmetic exactly

FSM: MEAN_ACC -> VAR_ACC -> INV_SQRT -> NORM
Math: y = clamp(round((x-mean)*inv_std*gamma/32768) + beta, -128, 127)
"""
def golden_layernorm(inputs, gamma_bytes, beta_bytes, isqrt_lut):
    # Pass 1: mean = sum(x) >>> 7
    total = sum(inputs)
    mean = total >> 7

    # Pass 2: center and accumulate variance
    centered = [x - mean for x in inputs]
    var_acc = sum(d * d for d in centered)

    # inv_sqrt(var_acc >> 7) via LOD-LUT-Shift
    isqrt_input = var_acc >> 7
    inv_std = golden_inv_sqrt(isqrt_input, isqrt_lut)

    # Pass 3: combined multiply-shift with rounding
    # Match RTL bit widths: full_prod is 33-bit signed, biased is 18-bit signed
    outputs = []
    for i in range(DIM):
        diff = centered[i]
        g = to_signed8(gamma_bytes[i])
        b = to_signed8(beta_bytes[i])

        # full_prod = diff * inv_std * gamma (33-bit signed)
        full_prod = diff * inv_std * g
        full_prod = full_prod & 0x1FFFFFFFF
        if full_prod >= (1 << 32):
            full_prod -= (1 << 33)

        # (full_prod + 16384) >>> 15, then + beta (18-bit signed)
        biased = ((full_prod + 16384) >> 15) + b
        biased = biased & 0x3FFFF
        if biased >= (1 << 17):
            biased -= (1 << 18)

        y = max(-128, min(127, biased))
        outputs.append(y)

    return outputs


"""
Ideal float LayerNorm: y = (x - mean) / sqrt(var) * gamma + beta
Uses real math, quantized to int8 at output only
"""
def ideal_layernorm(inputs, gamma_bytes, beta_bytes):
    n = len(inputs)
    mean = sum(inputs) / n
    var = sum((x - mean) ** 2 for x in inputs) / n

    if var < 1e-10:
        std_inv = 0.0
    else:
        std_inv = 1.0 / math.sqrt(var)

    outputs = []
    for i in range(n):
        g = to_signed8(gamma_bytes[i])
        b = to_signed8(beta_bytes[i])
        norm = (inputs[i] - mean) * std_inv
        y_f = norm * g + b
        y = max(-128, min(127, round(y_f)))
        outputs.append(y)

    return outputs


"""
Parse tb_layernorm.log for T=N OUT[idx]=value lines
Returns dict: test_num -> list of (idx, signed_value)
"""
def parse_log(path):
    pat = re.compile(r"T=(\d+)\s+OUT\[(\d+)\]=(-?\d+)")
    results = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                t = int(m.group(1))
                idx = int(m.group(2))
                val = int(m.group(3))
                if t not in results:
                    results[t] = []
                results[t].append((idx, val))
    return results


if __name__ == "__main__":
    if not os.path.exists(LN_HEX):
        print(f"LN params not found: {LN_HEX}")
        print("Run: python3 scripts/extract_weights.py")
        sys.exit(1)

    if not os.path.exists(ISQRT_HEX):
        print(f"inv_sqrt LUT not found: {ISQRT_HEX}")
        print("Run: python3 scripts/gen_inv_sqrt_lut.py")
        sys.exit(1)

    ln_mem = load_hex(LN_HEX)
    print(f"Loaded LN params: {len(ln_mem)} bytes from {LN_HEX}")

    isqrt_lut = load_hex(ISQRT_HEX)
    print(f"Loaded inv_sqrt LUT: {len(isqrt_lut)} entries from {ISQRT_HEX}")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_layernorm simulation first (Tcl: 'run all')")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if not xsim:
        print("No T=N OUT[idx]=value lines found in log")
        print("Set tb_layernorm as sim top, then 'run all'")
        sys.exit(1)

    total_errors = 0

    for t_idx, test in enumerate(TESTS):
        entries = xsim.get(t_idx, [])
        if not entries:
            print(f"=== Test {t_idx} ({test['name']}): no data ===")
            total_errors += 1
            continue

        # Load gamma and beta from ln_params.hex
        gamma_sel = test["gamma_sel"]
        gamma_off = LN_OFFSETS[gamma_sel]
        beta_off = LN_OFFSETS[gamma_sel + 1]
        gamma_bytes = ln_mem[gamma_off:gamma_off + DIM]
        beta_bytes = ln_mem[beta_off:beta_off + DIM]

        # Run both models
        golden = golden_layernorm(test["inputs"], gamma_bytes, beta_bytes,
                                  isqrt_lut)
        ideal = ideal_layernorm(test["inputs"], gamma_bytes, beta_bytes)

        errors = 0
        max_abs_delta = 0
        sum_abs_delta = 0
        max_abs_ideal = 0
        sum_abs_ideal = 0

        print(f"=== Test {t_idx}: {test['name']} (gamma_sel={gamma_sel})=== ")
        print(f"{'idx':>5s}  {'xsim':>5s}  {'golden':>6s}  {'ideal':>6s}  {'d_gold':>6s}  {'d_ideal':>7s}  {'status'}")
        print("-" * 56)

        for idx, xsim_val in entries:
            gold_val = golden[idx]
            ideal_val = ideal[idx]
            d_gold = xsim_val - gold_val
            d_ideal = xsim_val - ideal_val

            max_abs_delta = max(max_abs_delta, abs(d_gold))
            sum_abs_delta += abs(d_gold)
            max_abs_ideal = max(max_abs_ideal, abs(d_ideal))
            sum_abs_ideal += abs(d_ideal)

            if xsim_val != gold_val:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            print(f"{idx:5d}  {xsim_val:5d}  {gold_val:6d}  {ideal_val:6d}  {d_gold:+6d}  {d_ideal:+7d}  {status}")

        n = len(entries)
        print()
        print(f"Golden match:       {n - errors}/{n}")
        print(f"  Max abs delta:    {max_abs_delta}")
        print(f"  Mean abs delta:   {sum_abs_delta / n:.3f}")
        print(f"vs ideal float:")
        print(f"  Max abs delta:    {max_abs_ideal}")
        print(f"  Mean abs delta:   {sum_abs_ideal / n:.3f}")
        total_errors += errors
        print()

    n_tests = len(TESTS)
    n_outputs = n_tests * DIM
    if total_errors == 0:
        print(f"PASSED - all {n_outputs} outputs match golden model")
    else:
        print(f"FAILED - {total_errors} mismatches vs golden model")
    sys.exit(0 if total_errors == 0 else 1)