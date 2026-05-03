"""Generate sampler test vectors for tb_sampler

Record layout per test:
  index 0      inv_temp
  index 1      top_k in low byte
  index 2      seed
  index 3..258 N logits
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_ops import fp16_from_float

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJ, "mem", "sampler_test_vectors.hex")

N = 256
REC_W = 3 + N


def make_logits(rng, kind):
    if kind == "peaked":
        x = [rng.gauss(0, 1.0) for _ in range(N)]
        x[42] = 8.0
    elif kind == "realistic":
        x = [rng.gauss(0, 1.5) for _ in range(N)]
        for _ in range(5):
            x[rng.randrange(N)] += 4.0
    elif kind == "two_peak":
        x = [rng.gauss(0, 0.5) for _ in range(N)]
        x[100] = 6.0
        x[200] = 5.5
    else:
        raise ValueError(kind)
    return [fp16_from_float(v) for v in x]


def build_tests():
    rng = random.Random(42)
    return [
        {
            "name": "greedy_peaked",
            "inv_temp_bits": fp16_from_float(1.0),
            "top_k": 1,
            "seed": 0xACE1,
            "logits": make_logits(rng, "peaked"),
        },
        {
            "name": "topk4_realistic",
            "inv_temp_bits": fp16_from_float(1.0),
            "top_k": 4,
            "seed": 0xBEEF,
            "logits": make_logits(rng, "realistic"),
        },
        {
            "name": "topk16_twopeak_lowT",
            "inv_temp_bits": fp16_from_float(2.0),
            "top_k": 16,
            "seed": 0x1234,
            "logits": make_logits(rng, "two_peak"),
        },
        {
            "name": "fullvocab_realistic",
            "inv_temp_bits": fp16_from_float(1.0),
            "top_k": 0,
            "seed": 0x5678,
            "logits": make_logits(rng, "realistic"),
        },
        {
            "name": "greedy_peaked_highT",
            "inv_temp_bits": fp16_from_float(0.5),
            "top_k": 1,
            "seed": 0xACE1,
            "logits": make_logits(rng, "peaked"),
        },
    ]


TESTS = build_tests()


def write_hex(path, tests):
    with open(path, "w") as f:
        for t in tests:
            f.write(f"{t['inv_temp_bits']:04x}\n")
            f.write(f"{t['top_k']:04x}\n")
            f.write(f"{t['seed']:04x}\n")
            for v in t["logits"]:
                f.write(f"{v:04x}\n")


if __name__ == "__main__":
    write_hex(OUT, TESTS)
    print(f"Wrote {len(TESTS)} test vectors to {OUT}")