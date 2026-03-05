"""
Validate tb_fp16 xsim output against bit-exact rtl models

Parses logs/tb_fp16.log (unified testbench) for all fp16 primitives:
add, mul, mac, from_int8, to_int8, to_q167, q115_to_fp16, rsqrt, matvec_fp16
"""

import re
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_ops import (
    fp16_add, fp16_mul, fp16_mac,
    fp16_from_int, fp16_from_float, fp16_to_float, fp16_to_q167,
    q115_to_fp16 as rtl_q115_to_fp16, fp16_rsqrt_lut,
    load_lut16, to_signed8,
)

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(PROJ, "logs", "tb_fp16.log")
MEM = os.path.join(PROJ, "mem")


def is_nan(bits):
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


# Print results for one section, return updated error count
def print_section(name, results, total_errors):
    if not results:
        return total_errors

    errors = 0
    max_abs_delta = 0
    sum_abs_delta = 0
    n = len(results)

    has_ideal = any(r.get("ideal") is not None for r in results)
    is_hex = results[0].get("is_hex", True) if results else True

    if has_ideal:
        print(f"{'idx':>5s}  {'xsim':>8s}  {'rtl':>8s}  {'ideal':>10s}  {'g-i delta':>10s}  {'status'}")
    else:
        print(f"{'idx':>5s}  {'xsim':>8s}  {'rtl':>8s}  {'delta':>6s}  {'status'}")

    n_delta = 0
    for r in results:
        idx = r["idx"]
        got = r["got"]
        rtl = r["rtl"]
        ideal = r.get("ideal")
        got_nan = is_nan(got) if is_hex else False
        rtl_nan = is_nan(rtl) if is_hex else False

        if got_nan and rtl_nan:
            match = True
        else:
            match = (got == rtl)

        if not match:
            errors += 1
            status = "MISMATCH"
        else:
            status = "OK"

        if is_hex:
            got_s = f"{got:04x}"
            rtl_s = f"{rtl:04x}"
        else:
            got_s = f"{to_signed8(got):4d}"
            rtl_s = f"{to_signed8(rtl):4d}"

        if has_ideal and ideal is not None:
            # Detect special values where delta is meaningless
            rtl_inf = is_hex and (rtl & 0x7FFF) == 0x7C00
            special = False
            if isinstance(ideal, float) and (math.isnan(ideal) or math.isinf(ideal)):
                special = True
            elif isinstance(ideal, int) and abs(ideal) > 9000:
                special = True
            elif rtl_nan or rtl_inf:
                special = True

            if isinstance(ideal, float):
                ideal_s = f"{ideal:10.4f}"
            else:
                ideal_s = f"{ideal:10d}"

            if special:
                gi_delta_s = f"{'---':>10s}"
            elif isinstance(ideal, float):
                if is_hex:
                    rtl_f = fp16_to_float(rtl)
                    gi_delta_s = f"{rtl_f - ideal:+10.4f}"
                else:
                    gi_delta_s = f"{to_signed8(rtl) - ideal:+10.4f}"
            else:
                if is_hex:
                    gi_delta_s = f"{rtl - ideal:+10d}"
                else:
                    gi_delta_s = f"{to_signed8(rtl) - ideal:+10d}"

            if not special:
                if is_hex:
                    abs_d = abs(fp16_to_float(rtl) - ideal) if isinstance(ideal, float) else abs(rtl - ideal)
                else:
                    abs_d = abs(to_signed8(rtl) - ideal)
                max_abs_delta = max(max_abs_delta, abs_d)
                sum_abs_delta += abs_d
                n_delta += 1

            print(f"{idx:5d}  {got_s:>8s}  {rtl_s:>8s}  {ideal_s}  {gi_delta_s}  {status}")
        else:
            delta = got - rtl
            abs_d = abs(delta)
            max_abs_delta = max(max_abs_delta, abs_d)
            sum_abs_delta += abs_d
            n_delta += 1
            print(f"{idx:5d}  {got_s:>8s}  {rtl_s:>8s}  {delta:+6d}  {status}")

    print()
    print(f"  RTL match:    {n - errors}/{n}")
    if n_delta > 0:
        print(f"  Max abs delta:   {max_abs_delta}")
        print(f"  Mean abs delta:  {sum_abs_delta / n_delta:.3f}")
    else:
        print(f"  Max abs delta:   N/A (all special)")
        print(f"  Mean abs delta:  N/A (all special)")
    print()
    return total_errors + errors


def parse_and_validate():
    with open(LOG) as f:
        lines = f.readlines()

    total_errors = 0
    total_count = 0

    # fp16_add
    add_pat = re.compile(r"ADD \[(\d+)\] a=([0-9a-f]{4}) b=([0-9a-f]{4}) got=([0-9a-f]{4})")
    add_results = []
    for line in lines:
        m = add_pat.search(line)
        if m:
            a_bits = int(m.group(2), 16)
            b_bits = int(m.group(3), 16)
            got = int(m.group(4), 16)
            rtl = fp16_add(a_bits, b_bits)
            ideal = fp16_to_float(a_bits) + fp16_to_float(b_bits)
            add_results.append({"idx": int(m.group(1)), "got": got,
                                "rtl": rtl, "ideal": ideal})

    if add_results:
        print(f"Test: fp16_add ({len(add_results)} vectors)")
        total_errors = print_section("fp16_add", add_results, total_errors)
        total_count += len(add_results)

    # fp16_mul
    mul_pat = re.compile(r"MUL \[(\d+)\] a=([0-9a-f]{4}) b=([0-9a-f]{4}) got=([0-9a-f]{4})")
    mul_results = []
    for line in lines:
        m = mul_pat.search(line)
        if m:
            a_bits = int(m.group(2), 16)
            b_bits = int(m.group(3), 16)
            got = int(m.group(4), 16)
            rtl = fp16_mul(a_bits, b_bits)
            ideal = fp16_to_float(a_bits) * fp16_to_float(b_bits)
            mul_results.append({"idx": int(m.group(1)), "got": got,
                                "rtl": rtl, "ideal": ideal})

    if mul_results:
        print(f"Test: fp16_mul ({len(mul_results)} vectors)")
        total_errors = print_section("fp16_mul", mul_results, total_errors)
        total_count += len(mul_results)

    # fp16_mac
    mac_pat = re.compile(r"MAC \[(\d+)\] got=([0-9a-f]{4})")
    mac_results = []
    pairs_path = os.path.join(MEM, "fp16_mac_pairs.hex")
    if os.path.exists(pairs_path):
        pairs = []
        with open(pairs_path) as pf:
            for pline in pf:
                s = pline.strip()
                if s:
                    val = int(s, 16)
                    pairs.append(((val >> 16) & 0xFFFF, val & 0xFFFF))

        for line in lines:
            m = mac_pat.search(line)
            if m:
                idx = int(m.group(1))
                got = int(m.group(2), 16)
                acc = 0x0000
                acc_f64 = 0.0
                for j in range(16):
                    a_bits = pairs[idx*16+j][0]
                    b_bits = pairs[idx*16+j][1]
                    acc = fp16_mac(acc, a_bits, b_bits)
                    acc_f64 += fp16_to_float(a_bits) * fp16_to_float(b_bits)
                mac_results.append({"idx": idx, "got": got,
                                    "rtl": acc, "ideal": acc_f64})

    if mac_results:
        print(f"Test: fp16_mac ({len(mac_results)} dot-products)")
        total_errors = print_section("fp16_mac", mac_results, total_errors)
        total_count += len(mac_results)

    # fp16_from_int8
    from_pat = re.compile(r"FROM \[(\d+)\] in=([0-9a-f]{2}) got=([0-9a-f]{4})")
    from_results = []
    for line in lines:
        m = from_pat.search(line)
        if m:
            byte_val = int(m.group(2), 16)
            got = int(m.group(3), 16)
            rtl = fp16_from_int(to_signed8(byte_val))
            from_results.append({"idx": int(m.group(1)), "got": got,
                                 "rtl": rtl, "ideal": None})

    if from_results:
        print(f"Test: fp16_from_int8 ({len(from_results)} vectors, exact)")
        total_errors = print_section("fp16_from_int8", from_results, total_errors)
        total_count += len(from_results)

    # fp16_to_int8
    to_pat = re.compile(r"TO \[(\d+)\] in=([0-9a-f]{4}) got=([0-9a-f]{2})")
    to_results = []
    for line in lines:
        m = to_pat.search(line)
        if m:
            in_bits = int(m.group(2), 16)
            got = int(m.group(3), 16)
            f = fp16_to_float(in_bits)
            if math.isnan(f):
                rtl = 127 & 0xFF
                ideal_int = 127
            elif math.isinf(f):
                rtl = (-128 & 0xFF) if f < 0 else (127 & 0xFF)
                ideal_int = -9999 if f < 0 else 9999
            else:
                ideal_int = int(round(f))
                rtl = max(-128, min(127, ideal_int)) & 0xFF
            to_results.append({"idx": int(m.group(1)), "got": got,
                               "rtl": rtl, "ideal": ideal_int,
                               "is_hex": False})

    if to_results:
        print(f"Test: fp16_to_int8 ({len(to_results)} vectors)")
        total_errors = print_section("fp16_to_int8", to_results, total_errors)
        total_count += len(to_results)

    # fp16_to_q167
    q167_pat = re.compile(r"Q167 \[(\d+)\] in=([0-9a-f]{4}) got=([0-9a-f]{6})")
    q167_results = []
    for line in lines:
        m = q167_pat.search(line)
        if m:
            fp16_bits = int(m.group(2), 16)
            got = int(m.group(3), 16)
            rtl = fp16_to_q167(fp16_bits)
            # Ideal: round(float_val * 128)
            f = fp16_to_float(fp16_bits)
            if math.isnan(f) or math.isinf(f):
                ideal_int = 9999999
            else:
                ideal_int = int(round(f * 128))
            q167_results.append({"idx": int(m.group(1)), "got": got,
                                 "rtl": rtl, "ideal": ideal_int,
                                 "is_hex": False})

    if q167_results:
        print(f"Test: fp16_to_q167 ({len(q167_results)} vectors)")
        total_errors = print_section("fp16_to_q167", q167_results, total_errors)
        total_count += len(q167_results)

    # q115_to_fp16
    q115_pat = re.compile(r"Q115 \[(\d+)\] in=([0-9a-f]{4}) got=([0-9a-f]{4})")
    q115_results = []
    for line in lines:
        m = q115_pat.search(line)
        if m:
            q115_val = int(m.group(2), 16)
            got = int(m.group(3), 16)
            rtl = rtl_q115_to_fp16(q115_val)
            ideal = q115_val / 32768.0
            q115_results.append({"idx": int(m.group(1)), "got": got,
                                 "rtl": rtl, "ideal": ideal})

    if q115_results:
        print(f"Test: q115_to_fp16 ({len(q115_results)} vectors)")
        total_errors = print_section("q115_to_fp16", q115_results, total_errors)
        total_count += len(q115_results)

    # fp16_rsqrt
    rsqrt_pat = re.compile(r"RSQRT \[(\d+)\] in=([0-9a-f]{4}) got=([0-9a-f]{4})")
    rsqrt_results = []
    isqrt_lut_path = os.path.join(MEM, "inv_sqrt_lut.hex")
    if os.path.exists(isqrt_lut_path):
        isqrt_lut = load_lut16(isqrt_lut_path, signed=False)
        for line in lines:
            m = rsqrt_pat.search(line)
            if m:
                fp16_bits = int(m.group(2), 16)
                got = int(m.group(3), 16)
                rtl = fp16_rsqrt_lut(fp16_bits, isqrt_lut)
                f = fp16_to_float(fp16_bits)
                if f > 0 and not math.isinf(f):
                    ideal = 1.0 / math.sqrt(f)
                else:
                    ideal = float('inf') if f == 0 else float('nan')
                rsqrt_results.append({"idx": int(m.group(1)), "got": got,
                                      "rtl": rtl, "ideal": ideal})

    if rsqrt_results:
        print(f"Test: fp16_rsqrt ({len(rsqrt_results)} vectors)")
        total_errors = print_section("fp16_rsqrt", rsqrt_results, total_errors)
        total_count += len(rsqrt_results)

    # matvec_fp16
    mv_pat = re.compile(r"MV([12]) \[(\d+)\] got=([0-9a-f]{4})")
    configs = {
        "1": {"name": "4x4", "in_dim": 4, "out_dim": 4},
        "2": {"name": "8x4", "in_dim": 4, "out_dim": 8},
    }

    for mv_id in ["1", "2"]:
        cfg = configs[mv_id]
        name = cfg["name"]
        in_dim = cfg["in_dim"]
        out_dim = cfg["out_dim"]

        w_path = os.path.join(MEM, f"matvec_fp16_{name}_weights.hex")
        iv_path = os.path.join(MEM, f"matvec_fp16_{name}_input.hex")
        if not os.path.exists(w_path):
            continue

        from rtl_ops import rtl_matvec_fp16, load_hex as load_hex_vals

        weights_u8 = load_hex_vals(w_path)[:out_dim * in_dim]

        in_fp16 = []
        with open(iv_path) as ivf:
            for ivline in ivf:
                s = ivline.strip()
                if s:
                    in_fp16.append(int(s, 16))
        in_fp16 = in_fp16[:in_dim]

        scale_bits = 0x2C00  # fp16 0.0625
        rtl_vec = rtl_matvec_fp16(in_fp16, weights_u8, out_dim, in_dim, scale_bits)

        # Ideal: float64 math
        scale_f = fp16_to_float(scale_bits)
        ideal_vec = []
        for r in range(out_dim):
            acc_f64 = 0.0
            for c in range(in_dim):
                w_f = to_signed8(weights_u8[r * in_dim + c]) * scale_f
                acc_f64 += w_f * fp16_to_float(in_fp16[c])
            ideal_vec.append(acc_f64)

        mv_results = []
        for line in lines:
            m = mv_pat.search(line)
            if m and m.group(1) == mv_id:
                row = int(m.group(2))
                got = int(m.group(3), 16)
                mv_results.append({"idx": row, "got": got,
                                   "rtl": rtl_vec[row],
                                   "ideal": ideal_vec[row]})

        if mv_results:
            print(f"Test: matvec_fp16 {name} ({len(mv_results)} rows)")
            total_errors = print_section(f"matvec_{name}", mv_results, total_errors)
            total_count += len(mv_results)

    return total_errors, total_count


if __name__ == "__main__":
    if not os.path.exists(LOG):
        print(f"Log not found: {LOG}")
        print("Run tb_fp16 simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    total_errors, total_count = parse_and_validate()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)