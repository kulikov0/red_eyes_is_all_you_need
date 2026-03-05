"""
Generate fp16 test input vectors for tb_layernorm

3 embedding outputs + 5 synthetic stress vectors = 8 tests x 128 fp16
-> mem/ln_test_inputs.hex (1024 lines, 4-digit hex)
"""

import re
import os
import sys
import struct
import random
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_HEX = os.path.join(PROJ, "mem", "tok_emb_weight.hex")
POS_HEX = os.path.join(PROJ, "mem", "pos_emb_weight.hex")
SCALES_VH = os.path.join(PROJ, "rtl", "weight_scales.vh")
OUT = os.path.join(PROJ, "mem", "ln_test_inputs.hex")

DIM = 128


def load_hex(path):
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            vals.append(int(s, 16))
    return vals


def parse_fp16_scale(path, name):
    pat = re.compile(r"localparam\s+\[15:0\]\s+" + name + r"\s*=\s*16'h([0-9a-fA-F]{4})")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                bits = int(m.group(1), 16)
                return np.uint16(bits).view(np.float16)
    print(f"Scale {name} not found in {path}")
    sys.exit(1)


def to_signed8(b):
    return b - 256 if b >= 128 else b


def fp16_bits(f):
    return int(np.float16(f).view(np.uint16))


# Compute embedding output: fp16(tok_int8) * tok_scale + fp16(pos_int8) * pos_scale
def embed_fp16(tok_mem, pos_mem, tok_id, pos_id, tok_scale, pos_scale):
    vec = []
    base_t = tok_id * DIM
    base_p = pos_id * DIM
    for i in range(DIM):
        t = np.float16(float(to_signed8(tok_mem[base_t + i])))
        p = np.float16(float(to_signed8(pos_mem[base_p + i])))
        val = np.float16(np.float16(t * tok_scale) + np.float16(p * pos_scale))
        vec.append(val)
    return [fp16_bits(v) for v in vec]


# Synthetic: fp16 bit patterns from float list
def synth_vec(floats):
    return [fp16_bits(f) for f in floats]


if __name__ == "__main__":
    tok_mem = load_hex(TOK_HEX)
    pos_mem = load_hex(POS_HEX)
    tok_scale = parse_fp16_scale(SCALES_VH, "SCALE_TOK_EMB_WEIGHT")
    pos_scale = parse_fp16_scale(SCALES_VH, "SCALE_POS_EMB_WEIGHT")

    random.seed(42)
    tests = []

    # Test 0: embed(tok=0, pos=0) - typical small values
    tests.append(embed_fp16(tok_mem, pos_mem, 0, 0, tok_scale, pos_scale))

    # Test 1: embed(tok=42, pos=10) - different embedding
    tests.append(embed_fp16(tok_mem, pos_mem, 42, 10, tok_scale, pos_scale))

    # Test 2: embed(tok=200, pos=100) - high token/pos
    tests.append(embed_fp16(tok_mem, pos_mem, 200, 100, tok_scale, pos_scale))

    # Test 3: large magnitude values (simulates residual accumulation at deep layers)
    # Values in [-50, +50] range, std ~20
    tests.append(synth_vec([random.uniform(-50, 50) for _ in range(DIM)]))

    # Test 4: near-constant vector (tiny variance, stress inv_sqrt)
    # All values ~5.0 with tiny perturbation
    tests.append(synth_vec([5.0 + random.uniform(-0.01, 0.01) for _ in range(DIM)]))

    # Test 5: mixed outliers (a few large values among small ones)
    mixed = [random.uniform(-0.5, 0.5) for _ in range(DIM)]
    for i in [0, 32, 64, 96]:
        mixed[i] = random.choice([-30.0, 30.0])
    tests.append(synth_vec(mixed))

    # Test 6: all negative (biased mean)
    tests.append(synth_vec([random.uniform(-20, -1) for _ in range(DIM)]))

    # Test 7: wide range with zeros (sparse-like)
    sparse = [0.0] * DIM
    for i in range(0, DIM, 4):
        sparse[i] = random.uniform(-100, 100)
    tests.append(synth_vec(sparse))

    with open(OUT, "w") as f:
        for t in tests:
            for bits in t:
                f.write(f"{bits:04x}\n")

    print(f"Wrote {OUT} ({len(tests) * DIM} fp16 values)")
    for i, t in enumerate(tests):
        vals = [float(np.uint16(b).view(np.float16)) for b in t]
        mn, mx = min(vals), max(vals)
        mean = sum(vals) / len(vals)
        std = np.std(vals)
        print(f"  Test {i}: min={mn:+9.4f} max={mx:+9.4f} mean={mean:+9.4f} std={std:7.4f}")