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
Same math as matvec_int8.v

For each output row:
  acc (24-bit signed) = sum of in_vec[col] * weight[row*128 + col]
  out[row] = clamp( acc >> 7, -128, 127 )

Testbench uses block0_attn_proj_weight (128x128) with input = all ones
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
            # in_vec[col] = 1 for all cols (testbench fills with ones)
            acc = mask24(acc + as_signed8(raw[row * 128 + col]))
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
        print("Run tb_matvec_int8 simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if not xsim:
        print("No 'out[N] = ...' lines found in log")
        print("Set tb_matvec_int8 as sim top, then 'run all'")
        sys.exit(1)

    golden = golden_matvec(tensors)
    errors = 0

    for i in range(128):
        if i not in xsim:
            print(f"MISSING out[{i}] in log")
            errors += 1
            continue

        got = xsim[i]
        exp = golden[i]
        if got != exp:
            print(f"MISMATCH out[{i}]: xsim={got}  expected={exp}")
            errors += 1
        else:
            print(f"OK out[{i:3d}] = {got}")

    print()
    if errors == 0:
        print(f"PASSED - all {len(xsim)} outputs match")
    else:
        print(f"FAILED - {errors} mismatches out of {len(xsim)} checks")
    sys.exit(0 if errors == 0 else 1)
