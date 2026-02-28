"""
Validate tb_embedding xsim output against golden model

Parses logs/tb_embedding.log, computes expected (tok+pos)>>>1 from
hex weight files, compares byte-exact. Also shows ideal float32 reference
"""

import re
import os
import sys
import struct

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK_HEX = os.path.join(PROJ, "mem", "tok_emb_weight.hex")
POS_HEX = os.path.join(PROJ, "mem", "pos_emb_weight.hex")
SCALES_VH = os.path.join(PROJ, "rtl", "weight_scales.vh")
LOG = os.path.join(PROJ, "logs", "tb_embedding.log")

DIM = 128


# Parse IEEE754 scale from weight_scales.vh by localparam name
def parse_scale(path, name):
    pat = re.compile(r"localparam\s+\[31:0\]\s+" + name + r"\s*=\s*32'h([0-9a-fA-F]{8})")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                return struct.unpack('>f', bytes.fromhex(m.group(1)))[0]
    print(f"Scale {name} not found in {path}")
    sys.exit(1)


SCALE_TOK = parse_scale(SCALES_VH, "SCALE_TOK_EMB_WEIGHT")
SCALE_POS = parse_scale(SCALES_VH, "SCALE_POS_EMB_WEIGHT")
S_OUT = SCALE_TOK + SCALE_POS


def load_hex(path):
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            vals.append(int(s, 16))
    return vals


# Convert unsigned byte to signed int8
def to_signed8(b):
    return b - 256 if b >= 128 else b


# Compute golden embedding: 9-bit add then extract [8:1] (matches Verilog)
def golden_byte(t_unsigned, p_unsigned):
    t = to_signed8(t_unsigned)
    p = to_signed8(p_unsigned)
    s9 = (t + p) & 0x1FF
    return (s9 >> 1) & 0xFF


# Compute ideal float32 reference, requantized to int8 with S_OUT
def ideal_byte(t_unsigned, p_unsigned):
    t = to_signed8(t_unsigned)
    p = to_signed8(p_unsigned)
    f = t * SCALE_TOK + p * SCALE_POS
    r = int(round(f / S_OUT))
    r = max(-128, min(127, r))
    return r & 0xFF


"""
Parse tb_embedding.log for T=N TOK=n POS=n I=n OUT=xx lines
Returns dict: test_num -> (tok, pos, {idx: out_byte})
"""
def parse_log(path):
    pat = re.compile(
        r"T=(\d+)\s+TOK=(\d+)\s+POS=(\d+)\s+I=(\d+)\s+OUT=([0-9a-fA-F]{2})")
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

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_embedding simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if len(xsim) == 0:
        print("No T=N TOK=n POS=n I=n OUT=xx lines found in log")
        sys.exit(1)

    total_errors = 0
    total_count = 0

    for test_num in sorted(xsim.keys()):
        tok, pos, data = xsim[test_num]
        errors = 0
        max_abs_delta = 0
        sum_abs_delta = 0
        tok_base = tok * DIM
        pos_base = pos * DIM

        print(f"=== Test {test_num}: tok={tok} pos={pos} ===")
        print(f"{'idx':>5s}  {'xsim':>5s}  {'golden':>6s}  {'ideal':>6s}  {'delta':>6s}  {'status'}")

        for i in range(DIM):
            if i not in data:
                print(f"{i:5d}  {'?':>5s}  {'?':>6s}  {'?':>6s}  {'?':>6s}  MISSING")
                errors += 1
                continue

            xsim_byte = data[i]
            gold_byte = golden_byte(tok_mem[tok_base + i], pos_mem[pos_base + i])
            idl_byte = ideal_byte(tok_mem[tok_base + i], pos_mem[pos_base + i])

            xsim_s = to_signed8(xsim_byte)
            gold_s = to_signed8(gold_byte)
            ideal_s = to_signed8(idl_byte)
            delta = xsim_s - gold_s

            max_abs_delta = max(max_abs_delta, abs(delta))
            sum_abs_delta += abs(delta)

            if xsim_byte != gold_byte:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            print(f"{i:5d}  {xsim_s:5d}  {gold_s:6d}  {ideal_s:6d}  {delta:+6d}  {status}")

        n = DIM
        print()
        print(f"  Golden match:     {n - errors}/{n}")
        print(f"  Max abs delta:    {max_abs_delta}")
        print(f"  Mean abs delta:   {sum_abs_delta / n:.3f}")
        total_errors += errors
        total_count += n
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match golden model")
    else:
        print(f"FAILED - {total_errors} mismatches vs golden model")
    sys.exit(0 if total_errors == 0 else 1)