"""
Validate tb_matvec_int8 xsim output against weights_int8.bin.

Reads simulate.log, finds the "out[N] = V" lines, and compares each
one against a golden reference computed with the same arithmetic as
the Verilog matvec_int8 module.

"""

import struct
import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(PROJ, "scripts", "train", "weights_int8.bin")
LOG = os.path.join(PROJ, "logs", "tb_matvec_int8.log")

def parse_bin(path):
    tensors = []
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"TFPGA001", f"Bad magic: {magic!r}"
        num = struct.unpack("<I", f.read(4))[0]
        for i in range(num):
            nl = struct.unpack("<I", f.read(4))[0]
            name = f.read(nl).decode("ascii")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
            scale_bytes = f.read(4)
            scale = struct.unpack("<f", scale_bytes)[0]
            scale_hex = struct.unpack("<I", scale_bytes)[0]
            size = 1
            for s in shape:
                size *= s
            data = f.read(size)
            assert len(data) == size
            tensors.append({
                "index": i, "name": name, "shape": shape,
                "size": size, "scale": scale, "scale_hex": scale_hex,
                "data": data,
            })
    return tensors


"""
Testbench uses block0_attn_proj_weight (128x128) with input = all ones

golden: mirrors hardware (24-bit masked accumulator, >>7 requant, clamp)
ref:    standard int8 matmul (full-precision accumulator, >>7 requant, clamp)
"""

def as_signed8(b):
    return b - 256 if b >= 128 else b

def clamp8(v):
    if v > 127: return 127
    if v < -128: return -128
    return v

# Wrap to 24-bit signed, same as Verilog reg signed [23:0]
def mask24(v):
    v &= 0xFFFFFF
    return v - 0x1000000 if v >= 0x800000 else v

def golden_matvec(tensors):
    proj = next(t for t in tensors if t["name"] == "blocks.0.attn.proj.weight")
    raw = proj["data"]
    out = []
    for row in range(128):
        acc = 0
        for col in range(128):
            acc = mask24(acc + as_signed8(raw[row * 128 + col]))
        out.append(clamp8(acc >> 7))
    return out

# Reference int8 matmul with full-precision accumulator (no 24-bit wrap)
def ref_int8_matvec(tensors):
    proj = next(t for t in tensors if t["name"] == "blocks.0.attn.proj.weight")
    raw = proj["data"]
    out = []
    for row in range(128):
        acc = 0
        for col in range(128):
            acc += as_signed8(raw[row * 128 + col])
        out.append(clamp8(acc >> 7))
    return out


# Parse xsim log for "out[N] = V" lines
def parse_log(path):
    pat = re.compile(r"out\[(\d+)\]\s*=\s*(-?\d+)")
    results = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                results[int(m.group(1))] = int(m.group(2))
    return results


if __name__ == "__main__":
    tensors = parse_bin(BIN)
    print(f"Loaded {len(tensors)} tensors from weights_int8.bin")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_matvec_int8 simulation first (Tcl: 'run all')")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if not xsim:
        print("No 'out[N] = ...' lines found in log")
        print("Set tb_matvec_int8 as sim top, then 'run all'")
        sys.exit(1)

    golden = golden_matvec(tensors)
    ref = ref_int8_matvec(tensors)
    gold_errors = 0
    ref_errors = 0
    max_g_delta = 0
    max_r_delta = 0
    sum_g_delta = 0
    sum_r_delta = 0

    print(f"{'row':>5s}  {'xsim':>5s}  {'golden':>6s}  {'ref':>5s}  {'g_delta':>7s}  {'r_delta':>7s}  {'status'}")
    print("-" * 56)

    for i in range(128):
        if i not in xsim:
            print(f"{i:5d}  {'MISSING':>5s}")
            gold_errors += 1
            ref_errors += 1
            continue

        got = xsim[i]
        gv = golden[i]
        rv = ref[i]
        g_delta = got - gv
        r_delta = got - rv

        max_g_delta = max(max_g_delta, abs(g_delta))
        max_r_delta = max(max_r_delta, abs(r_delta))
        sum_g_delta += abs(g_delta)
        sum_r_delta += abs(r_delta)

        if got != gv:
            status = "GOLD_MISMATCH"
            gold_errors += 1
        elif got != rv:
            status = "REF_MISMATCH"
        else:
            status = "OK"

        if got != rv:
            ref_errors += 1

        print(f"{i:5d}  {got:5d}  {gv:6d}  {rv:5d}  {g_delta:+7d}  {r_delta:+7d}  {status}")

    n = len(xsim)
    print()
    print(f"Golden match:       {128 - gold_errors}/128  (hardware model)")
    print(f"  Max abs delta:    {max_g_delta}")
    print(f"  Mean abs delta:   {sum_g_delta / n:.3f}")
    print(f"Ref match:          {128 - ref_errors}/128  (int8 matmul)")
    print(f"  Max abs delta:    {max_r_delta}")
    print(f"  Mean abs delta:   {sum_r_delta / n:.3f}")
    print()
    if gold_errors == 0:
        print(f"PASSED - all {n} outputs match golden model")
    else:
        print(f"FAILED - {gold_errors} mismatches vs golden model")
    sys.exit(0 if gold_errors == 0 else 1)
