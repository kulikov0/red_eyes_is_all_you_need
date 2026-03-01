"""
Validate tb_kv_cache xsim output against expected written values

Parses logs/tb_kv_cache.log, compares read-back values against what was
written by the testbench
"""

import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(PROJ, "logs", "tb_kv_cache.log")

# Expected values matching tb_kv_cache.v test cases
# Each entry: (test_num, layer, kv, head, pos, dim, expected_byte)
EXPECTED = [
    # Test 0: single byte
    (0, 0, 0, 0, 0, 0, 0xA5),
    # Test 1: full position (16 dims)
    *[(1, 1, 1, 3, 42, d, d + 10) for d in range(16)],
    # Test 2: cross-head isolation
    (2, 0, 0, 0, 5, 0, 0x11),
    (2, 0, 0, 1, 5, 0, 0x22),
    # Test 3: K/V isolation
    (3, 2, 0, 5, 100, 7, 0xAA),
    (3, 2, 1, 5, 100, 7, 0x55),
    # Test 4: cross-layer isolation
    (4, 0, 0, 7, 200, 15, 0xFF),
    (4, 3, 0, 7, 200, 15, 0x01),
]


# Convert unsigned byte to signed int8
def to_signed8(b):
    return b - 256 if b >= 128 else b


# Parse log: T=N L=l KV=k H=h P=p D=d OUT=xx
def parse_log(path):
    pat = re.compile(
        r"T=(\d+)\s+L=(\d+)\s+KV=(\d+)\s+H=(\d+)\s+P=(\d+)\s+D=(\d+)\s+OUT=([0-9a-fA-F]{2})")
    results = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                results.append({
                    "t": int(m.group(1)),
                    "l": int(m.group(2)),
                    "kv": int(m.group(3)),
                    "h": int(m.group(4)),
                    "p": int(m.group(5)),
                    "d": int(m.group(6)),
                    "out": int(m.group(7), 16),
                })
    return results


if __name__ == "__main__":
    if not os.path.exists(LOG):
        print(f"Log not found: {LOG}")
        print("Run tb_kv_cache simulation first")
        sys.exit(1)

    print(f"Reading: {LOG}\n")
    entries = parse_log(LOG)

    if not entries:
        print("No T=N L=l KV=k H=h P=p D=d OUT=xx lines found in log")
        sys.exit(1)

    if len(entries) != len(EXPECTED):
        print(f"Expected {len(EXPECTED)} entries, got {len(entries)}")
        sys.exit(1)

    # Group by test number
    by_test = {}
    for i, (e, exp) in enumerate(zip(entries, EXPECTED)):
        t = exp[0]
        if t not in by_test:
            by_test[t] = []
        by_test[t].append((i, e, exp))

    test_names = {
        0: "single byte",
        1: "full position (16 dims)",
        2: "cross-head isolation",
        3: "K/V isolation",
        4: "cross-layer isolation",
    }

    total_errors = 0
    total_count = 0

    for t in sorted(by_test.keys()):
        items = by_test[t]
        print(f"Test {t}: {test_names.get(t, '')}")
        print(f"{'idx':>5s}  {'xsim':>5s}  {'golden':>6s}  {'delta':>6s}  {'status'}")

        errors = 0
        max_abs_delta = 0
        sum_abs_delta = 0

        for idx_in_test, (i, e, exp) in enumerate(items):
            xsim_byte = e["out"]
            gold_byte = exp[6] & 0xFF

            xsim_s = to_signed8(xsim_byte)
            gold_s = to_signed8(gold_byte)
            delta = xsim_s - gold_s

            max_abs_delta = max(max_abs_delta, abs(delta))
            sum_abs_delta += abs(delta)

            if xsim_byte != gold_byte:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            print(f"{idx_in_test:5d}  {xsim_s:5d}  {gold_s:6d}  {delta:+6d}  {status}")

        n = len(items)
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