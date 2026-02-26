"""
Validate tb_weight_store xsim output against weights_int8.bin.

Reads simulate.log, finds the "OK tensor ..." and "FAIL tensor ..."
lines, and compares each one against the first/last bytes read
directly

"""

import struct
import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(PROJ, "scripts", "train", "weights_int8.bin")
LOG = os.path.join(PROJ, "logs", "tb_weight_store.log")

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

def golden_weight_store(tensors):
    return {t["index"]: (t["data"][0], t["data"][-1]) for t in tensors}


# Parse xsim log for tb_weight_store output
def parse_log(path):
    # "OK   tensor  5  addr=0      data=0xe6  scale=0x3a987282"
    ok_pat   = re.compile(r"OK\s+tensor\s+(\d+)\s+addr=(\d+)\s+data=0x([0-9a-fA-F]{2})")
    # "FAIL tensor 5 first: got 0xxx, expected 0xe6"
    fail_pat = re.compile(r"FAIL\s+tensor\s+(\d+)\s+(first|last):\s+got\s+0x([0-9a-fA-FxX]+)")

    results = []
    with open(path) as f:
        for line in f:
            m = ok_pat.search(line)
            if m:
                results.append({
                    "tensor": int(m.group(1)),
                    "addr": int(m.group(2)),
                    "data": int(m.group(3), 16),
                    "ok": True,
                })
                continue

            m = fail_pat.search(line)
            if m:
                got_str = m.group(3)
                results.append({
                    "tensor": int(m.group(1)),
                    "which": m.group(2),
                    "data": None if "x" in got_str.lower() else int(got_str, 16),
                    "ok": False,
                })
    return results

if __name__ == "__main__":
    tensors = parse_bin(BIN)
    print(f"Loaded {len(tensors)} tensors from weights_int8.bin")

    if not os.path.exists(LOG):
        print(f"\nLog not found: {LOG}")
        print("Run tb_weight_store simulation first (Tcl: 'run all')")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    xsim = parse_log(LOG)

    if not xsim:
        print("No 'OK tensor ...' or 'FAIL tensor ...' lines found in log")
        print("Set tb_weight_store as sim top, then 'run all'")
        sys.exit(1)

    golden = golden_weight_store(tensors)
    errors = 0

    for entry in xsim:
        idx = entry["tensor"]
        g_first, g_last = golden[idx]

        # xsim already flagged this as a failure
        if not entry["ok"]:
            errors += 1
            if entry["data"] is None:
                print(f"FAIL tensor {idx:2d} {entry['which']}: xsim got X (uninitialized)")
            else:
                print(f"FAIL tensor {idx:2d} {entry['which']}: xsim=0x{entry['data']:02x}")
            continue

        # xsim said OK - double-check against the binary
        expected = g_first if entry["addr"] == 0 else g_last
        if entry["data"] != expected:
            print(f"MISMATCH tensor {idx:2d} addr={entry['addr']}: "
                  f"xsim=0x{entry['data']:02x} expected=0x{expected:02x}")
            errors += 1
        else:
            print(f"OK tensor {idx:2d}  addr={entry['addr']:<5d}  data=0x{entry['data']:02x}")

    print()
    if errors == 0:
        print(f"PASSED - all {len(xsim)} checks match.")
    else:
        print(f"FAILED - {errors} mismatches out of {len(xsim)} checks.")
    sys.exit(0 if errors == 0 else 1)
