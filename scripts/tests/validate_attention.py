"""
Validate tb_attention xsim output against Python rtl model (W8A16)

Two reference models:
  rtl: exact W8A16 hardware fp16 arithmetic (matvec_fp16, bipartite LUT softmax)
          xsim must match rtl exactly (bit-for-bit fp16)
  ideal:  float64 QKV/attention, real softmax, fp16 output
          shows quantization/rounding error
"""

import re
import os
import sys

from rtl_ops import (
    DIM, MEM,
    to_signed8, fp16_to_float, fp16_from_int,
    load_hex, load_lut16, parse_scale_bits,
    rtl_attention_fp16,
)
from ideal_ops import ideal_attention_fp16

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOG = os.path.join(PROJ, "logs", "tb_attention.log")


def build_input_fp16(seed):
    return [fp16_from_int(to_signed8((seed + i) & 0xFF)) for i in range(DIM)]


def parse_log(path):
    test_pat = re.compile(r"TEST (\d+) LAYER=(\d+) POS=(\d+) SEED=(\d+)")
    out_pat = re.compile(r"OUT\[(\d+)\]=([0-9a-fA-F]{4})")

    tests = {}
    current_test = None
    with open(path) as f:
        for line in f:
            m = test_pat.search(line)
            if m:
                current_test = int(m.group(1))
                tests[current_test] = {
                    "layer": int(m.group(2)),
                    "pos": int(m.group(3)),
                    "seed": int(m.group(4)),
                    "outputs": {},
                }
                continue
            if current_test is None:
                continue
            m = out_pat.search(line)
            if m:
                idx = int(m.group(1))
                val = int(m.group(2), 16)
                tests[current_test]["outputs"][idx] = val
    return tests


if __name__ == "__main__":
    # Load weights and LUTs
    qkv_hex = os.path.join(MEM, "block0_attn_qkv_weight.hex")
    proj_hex = os.path.join(MEM, "block0_attn_proj_weight.hex")
    lut0_hex = os.path.join(MEM, "exp_lut0.hex")
    lut1_hex = os.path.join(MEM, "exp_lut1.hex")

    for path, name in [(qkv_hex, "qkv"), (proj_hex, "proj"),
                        (lut0_hex, "lut0"), (lut1_hex, "lut1")]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            sys.exit(1)

    qkv_w = load_hex(qkv_hex)
    proj_w = load_hex(proj_hex)
    lut0 = load_lut16(lut0_hex, signed=False)
    lut1 = load_lut16(lut1_hex, signed=True)
    qkv_scale = parse_scale_bits("SCALE_BLOCK0_ATTN_QKV_WEIGHT")
    proj_scale = parse_scale_bits("SCALE_BLOCK0_ATTN_PROJ_WEIGHT")

    print(f"Loaded QKV weights: {len(qkv_w)} bytes (scale=0x{qkv_scale:04x})")
    print(f"Loaded proj weights: {len(proj_w)} bytes (scale=0x{proj_scale:04x})")
    print(f"Loaded softmax LUTs: {len(lut0)}+{len(lut1)} entries")

    LOG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_attention simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    tests = parse_log(LOG)

    if not tests:
        print("No test data found in log")
        sys.exit(1)

    # Shared KV caches across tests (sequential positions)
    kv_cache = {}
    kv_ideal = {}
    total_errors = 0
    total_count = 0

    for tnum in sorted(tests.keys()):
        t = tests[tnum]
        layer_val = t["layer"]
        pos_val = t["pos"]
        seed = t["seed"]
        xsim_out = t["outputs"]

        x_fp16 = build_input_fp16(seed)
        rtl = rtl_attention_fp16(x_fp16, layer_val, pos_val, kv_cache,
                                       qkv_w, proj_w, lut0, lut1,
                                       qkv_scale, proj_scale)
        ideal = ideal_attention_fp16(x_fp16, layer_val, pos_val, kv_ideal,
                                      qkv_w, proj_w, qkv_scale, proj_scale)

        print(f"Test {tnum}: layer={layer_val} pos={pos_val} seed={seed}")

        errors = 0
        max_xi_delta = 0.0
        sum_xi_delta = 0.0
        n_delta = 0

        print(f"  {'idx':>5s}  {'xsim':>6s}  {'rtl':>6s}  {'ideal':>6s}"
              f"  {'x-i delta':>10s}  {'status'}")

        for i in range(DIM):
            xv = xsim_out.get(i, 0)
            gv = rtl[i]
            iv = ideal[i]

            match = (xv == gv)
            if not match:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            # Skip special values for delta
            xv_special = (xv & 0x7FFF) >= 0x7C00
            iv_special = (iv & 0x7FFF) >= 0x7C00
            if not xv_special and not iv_special:
                xf = fp16_to_float(xv)
                ivf = fp16_to_float(iv)
                xi_delta = xf - ivf
                max_xi_delta = max(max_xi_delta, abs(xi_delta))
                sum_xi_delta += abs(xi_delta)
                n_delta += 1
                print(f"  {i:5d}  {xv:04x}    {gv:04x}    {iv:04x}"
                      f"    {xi_delta:+10.6f}  {status}")
            else:
                print(f"  {i:5d}  {xv:04x}    {gv:04x}    {iv:04x}"
                      f"    {'---':>10s}  {status}")

        n = DIM
        print()
        print(f"  RTL match:       {n - errors}/{n}")
        if n_delta > 0:
            print(f"  Xsim vs ideal:      max={max_xi_delta:.6f}"
                  f"  mean={sum_xi_delta / n_delta:.6f}")
        total_errors += errors
        total_count += n
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)