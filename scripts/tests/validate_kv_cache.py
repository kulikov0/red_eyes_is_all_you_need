"""
Validate tb_kv_cache xsim output against expected written values

Parses logs/tb_kv_cache.log, compares read-back values against what was
written by the testbench. Both K and V caches are fp16 (16-bit)
"""

import re
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(PROJ, "logs", "tb_kv_cache.log")

# Expected values matching tb_kv_cache.v test cases (all 16-bit fp16)
# Each entry: (test_num, cache_type, layer, head, pos, dim, expected_value)
EXPECTED = [
    # Test 0: single V fp16 (1.0 = 0x3C00)
    (0, "V", 0, 0, 0, 0, 0x3C00),
    # Test 1: full position V (16 dims)
    *[(1, "V", 1, 3, 42, d, 0x3C00 + d) for d in range(16)],
    # Test 2: cross-head K isolation
    (2, "K", 0, 0, 5, 0, 0x4011),
    (2, "K", 0, 1, 5, 0, 0x4022),
    # Test 3: K/V cache isolation
    (3, "K", 2, 5, 100, 7, 0xCAFE),
    (3, "V", 2, 5, 100, 7, 0x5500),
    # Test 4: cross-layer K isolation
    (4, "K", 0, 7, 200, 15, 0xFFFF),
    (4, "K", 3, 7, 200, 15, 0x0001),
]


# Convert unsigned to signed with given bit width
def to_signed(val, bits):
    if val >= (1 << (bits - 1)):
        return val - (1 << bits)
    return val


# Parse log: T=N C=K/V L=l H=h P=p D=d OUT=hex
def parse_log(path):
    pat = re.compile(
        r"T=(\d+)\s+C=([KV])\s+L=(\d+)\s+H=(\d+)\s+P=(\d+)\s+D=(\d+)"
        r"\s+OUT=([0-9a-fA-F]+)")
    results = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                results.append({
                    "t": int(m.group(1)),
                    "c": m.group(2),
                    "l": int(m.group(3)),
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
        print("No T=N C=K/V L=l H=h P=p D=d OUT=xx lines found in log")
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
        0: "single V fp16",
        1: "full position V (16 dims)",
        2: "cross-head K isolation",
        3: "K/V cache isolation",
        4: "cross-layer K isolation",
    }

    total_errors = 0
    total_count = 0

    for t in sorted(by_test.keys()):
        items = by_test[t]
        print(f"Test {t}: {test_names.get(t, '')}")
        print(f"{'idx':>5s}  {'xsim':>7s}  {'rtl':>7s}  {'delta':>6s}  {'status'}")

        errors = 0
        max_abs_delta = 0
        sum_abs_delta = 0

        for idx_in_test, (i, e, exp) in enumerate(items):
            xsim_val = e["out"]
            rtl_val = exp[6] & 0xFFFF

            xsim_s = to_signed(xsim_val, 16)
            rtl_s = to_signed(rtl_val, 16)
            delta = xsim_s - rtl_s

            max_abs_delta = max(max_abs_delta, abs(delta))
            sum_abs_delta += abs(delta)

            if xsim_val != rtl_val:
                errors += 1
                status = "MISMATCH"
            else:
                status = "OK"

            print(f"{idx_in_test:5d}  {xsim_s:7d}  {rtl_s:7d}  {delta:+6d}  {status}")

        n = len(items)
        print()
        print(f"  RTL match:     {n - errors}/{n}")
        print(f"  Max abs delta:    {max_abs_delta}")
        print(f"  Mean abs delta:   {sum_abs_delta / n:.3f}")
        total_errors += errors
        total_count += n
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)