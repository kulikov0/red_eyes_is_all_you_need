"""
Generate rtl test vectors for FP16 arithmetic primitives

Uses bit-exact fp16 models from rtl_ops.py (not numpy fp16) so rtl
values match RTL rounding exactly. Each hex file has one concatenated hex
value per line so $readmemh loads each line as a single wide entry
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from rtl_ops import (
    fp16_add, fp16_mul, fp16_from_int, fp16_from_float, fp16_to_float,
    fp16_to_q167, q115_to_fp16, fp16_rsqrt_lut, load_lut16,
)

MEM_DIR = os.path.join(os.path.dirname(__file__), '..', 'mem')

# Convert Python float to fp16 bit pattern via rtl_ops
def float_to_fp16(f):
  return fp16_from_float(f)

# RTL model: fp16 bits -> signed int8 (RNE, clamped)
# Matches fp16_to_int8.v behavior
def fp16_to_int8_rtl(bits):
  sign = (bits >> 15) & 1
  exp = (bits >> 10) & 0x1F
  mant = bits & 0x3FF

  if exp == 0:
    return 0
  if exp == 31:
    return -128 if sign else 127

  full_mant = (1 << 10) | mant
  # val = full_mant * 2^(exp-25)
  shift = 25 - exp
  if shift > 11:
    return 0
  if shift > 0:
    mag = full_mant >> shift
    guard = (full_mant >> (shift - 1)) & 1
    sticky = (full_mant & ((1 << (shift - 1)) - 1)) != 0 if shift > 1 else 0
    # RNE
    if guard and (sticky or (mag & 1)):
      mag += 1
  elif shift == 0:
    mag = full_mant
  else:
    mag = full_mant << (-shift)

  if mag > 128 if sign else mag > 127:
    return -128 if sign else 127
  return (-mag if sign else mag) & 0xFF

# Generate test vectors for fp16_add: {a[15:0], b[15:0], expected[15:0]} = 48 bits
def gen_fp16_add_vectors():
  pairs = [
    (1.0, 1.0),
    (1.0, -1.0),
    (3.5, 2.25),
    (100.0, 0.5),
    (-50.0, 25.0),
    (0.0, 5.0),
    (5.0, 0.0),
    (0.0, 0.0),
    (1024.0, 0.25),
    (256.0, -256.0),
    (1.001953125, -1.0),
    (32768.0, 32768.0),
    (65504.0, 0.0),
    (-65504.0, -1.0),
    (float('inf'), 1.0),
    (float('inf'), float('-inf')),
    (float('nan'), 1.0),
    (5.96e-8, 1.0),
    (1.0, 0.000244140625),
    (1.0, 0.0009765625),
  ]

  vectors = []
  for a, b in pairs:
    a_bits = float_to_fp16(a)
    b_bits = float_to_fp16(b)
    expected = fp16_add(a_bits, b_bits)
    vectors.append((a_bits, b_bits, expected))

  rng = random.Random(42)
  for _ in range(30):
    a_bits = float_to_fp16(rng.uniform(-100, 100))
    b_bits = float_to_fp16(rng.uniform(-100, 100))
    expected = fp16_add(a_bits, b_bits)
    vectors.append((a_bits, b_bits, expected))

  path = os.path.join(MEM_DIR, 'fp16_add_vectors.hex')
  with open(path, 'w') as f:
    for a_bits, b_bits, expected in vectors:
      f.write(f'{a_bits:04x}{b_bits:04x}{expected:04x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

# Generate test vectors for fp16_mul: {a, b, expected} = 48 bits
def gen_fp16_mul_vectors():
  pairs = [
    (2.0, 3.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (-2.0, -3.0),
    (0.5, 0.5),
    (0.0, 5.0),
    (5.0, 0.0),
    (0.0, 0.0),
    (256.0, 256.0),
    (128.0, 0.5),
    (0.25, 4.0),
    (256.0, 255.0),
    (255.0, 0.00390625),
    (float('inf'), 2.0),
    (float('inf'), 0.0),
    (float('nan'), 1.0),
    (float('-inf'), float('-inf')),
    (0.01, 0.01),
  ]

  vectors = []
  for a, b in pairs:
    a_bits = float_to_fp16(a)
    b_bits = float_to_fp16(b)
    expected = fp16_mul(a_bits, b_bits)
    vectors.append((a_bits, b_bits, expected))

  rng = random.Random(43)
  for _ in range(30):
    a_bits = float_to_fp16(rng.uniform(-50, 50))
    b_bits = float_to_fp16(rng.uniform(-50, 50))
    expected = fp16_mul(a_bits, b_bits)
    vectors.append((a_bits, b_bits, expected))

  path = os.path.join(MEM_DIR, 'fp16_mul_vectors.hex')
  with open(path, 'w') as f:
    for a_bits, b_bits, expected in vectors:
      f.write(f'{a_bits:04x}{b_bits:04x}{expected:04x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

# Generate exhaustive int8->fp16 vectors: {int8[7:0], expected[15:0]} = 24 bits
def gen_fp16_from_int8_vectors():
  vectors = []
  for i in range(256):
    signed_val = i if i < 128 else i - 256
    fp16_bits = fp16_from_int(signed_val)
    vectors.append((i, fp16_bits))

  path = os.path.join(MEM_DIR, 'fp16_from_int8_vectors.hex')
  with open(path, 'w') as f:
    for int_val, fp16_bits in vectors:
      f.write(f'{int_val:02x}{fp16_bits:04x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

# Generate fp16->int8 vectors: {fp16[15:0], expected[7:0]} = 24 bits
def gen_fp16_to_int8_vectors():
  test_floats = [
    0.0, 1.0, -1.0, 0.5, -0.5,
    127.0, -128.0,
    127.5, -128.5,
    200.0, -200.0,
    0.25, -0.25,
    0.75, -0.75,
    1.5, 2.5, 3.5,
    -1.5, -2.5,
    65504.0,
    -65504.0,
    float('inf'),
    float('-inf'),
    float('nan'),
    0.1, -0.1,
    42.0, -42.0, 100.0, -100.0,
  ]

  vectors = []
  for fv in test_floats:
    bits = float_to_fp16(fv)
    expected = fp16_to_int8_rtl(bits)
    vectors.append((bits, expected & 0xFF))

  rng = random.Random(44)
  for _ in range(20):
    bits = float_to_fp16(rng.uniform(-150, 150))
    expected = fp16_to_int8_rtl(bits)
    vectors.append((bits, expected & 0xFF))

  path = os.path.join(MEM_DIR, 'fp16_to_int8_vectors.hex')
  with open(path, 'w') as f:
    for fp16_bits, int8_val in vectors:
      f.write(f'{fp16_bits:04x}{int8_val:02x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

"""
Generate MAC test vectors: pairs file + expected file
Pairs: {a[15:0], b[15:0]} = 32 bits, 16 per test
Expected: {result[15:0]} = 16 bits, 1 per test
"""
def gen_fp16_mac_vectors():
  rng = random.Random(45)

  # Helper: build fp16 bits list from floats
  def to_fp16_list(floats):
    return [float_to_fp16(f) for f in floats]

  tests = []
  tests.append((to_fp16_list([1.0] * 16), to_fp16_list([1.0] * 16)))
  tests.append((to_fp16_list([1.0, -1.0] * 8), to_fp16_list([2.0, 2.0] * 8)))

  for _ in range(3):
    a = to_fp16_list([rng.uniform(-10, 10) for _ in range(16)])
    b = to_fp16_list([rng.uniform(-10, 10) for _ in range(16)])
    tests.append((a, b))

  pairs_path = os.path.join(MEM_DIR, 'fp16_mac_pairs.hex')
  exp_path = os.path.join(MEM_DIR, 'fp16_mac_expected.hex')
  with open(pairs_path, 'w') as fp, open(exp_path, 'w') as fe:
    for a_bits, b_bits in tests:
      for j in range(16):
        fp.write(f'{a_bits[j]:04x}{b_bits[j]:04x}\n')
      # Accumulate using bit-exact fp16_mul + fp16_add
      acc = 0x0000
      for j in range(16):
        prod = fp16_mul(a_bits[j], b_bits[j])
        acc = fp16_add(acc, prod)
      fe.write(f'{acc:04x}\n')
  print(f'Wrote {len(tests)} MAC tests to {pairs_path} + {exp_path}')

# Generate fp16_to_q167 vectors: {fp16[15:0], expected[23:0]} = 40 bits
def gen_fp16_to_q167_vectors():
  test_values = [
    0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25,
    0.0078125, -0.0078125,
    127.0, -128.0,
    255.0, -255.0,
    256.0, -256.0,
    1000.0, -1000.0,
    65504.0, -65504.0,
    0.00390625,
    100.5, -100.5,
    0.1, -0.1,
    float('inf'), float('-inf'), float('nan'),
  ]

  vectors = []
  for fv in test_values:
    bits = float_to_fp16(fv)
    expected = fp16_to_q167(bits)
    vectors.append((bits, expected))

  rng = random.Random(46)
  for _ in range(30):
    bits = float_to_fp16(rng.uniform(-65504, 65504))
    expected = fp16_to_q167(bits)
    vectors.append((bits, expected))

  path = os.path.join(MEM_DIR, 'fp16_to_q167_vectors.hex')
  with open(path, 'w') as f:
    for fp16_bits, q167_val in vectors:
      f.write(f'{fp16_bits:04x}{q167_val:06x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

"""
Generate q115_to_fp16 vectors: {q115[15:0], expected[15:0]} = 32 bits
Q1.15 unsigned: 0 = 0.0, 32768 = 1.0. Valid range [0, 32768]
"""
def gen_q115_to_fp16_vectors():
  # Powers of 2 (exact)
  test_values = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  # Key boundaries
  test_values += [
    32768,  # 1.0
    16384,  # 0.5
    8192,   # 0.25
    32767,  # just below 1.0
  ]
  # Near rounding thresholds (lod=11..15 triggers RNE in q115_to_fp16)
  # lod=11: discard 1 bit
  test_values += [2047, 2048, 2049, 3071, 3072, 3073]
  # lod=12: discard 2 bits
  test_values += [4095, 4096, 4097, 6143, 6144, 6145]
  # lod=14: discard 4 bits (softmax typical range)
  test_values += [16383, 16385, 24575, 24576, 24577]
  # lod=15: discard 5 bits (near 1.0)
  test_values += [32700, 32750, 32760]

  vectors = []
  for val in test_values:
    expected = q115_to_fp16(val)
    vectors.append((val, expected))

  rng = random.Random(47)
  for _ in range(30):
    val = rng.randint(0, 32768)
    expected = q115_to_fp16(val)
    vectors.append((val, expected))

  path = os.path.join(MEM_DIR, 'q115_to_fp16_vectors.hex')
  with open(path, 'w') as f:
    for q115_val, fp16_bits in vectors:
      f.write(f'{q115_val:04x}{fp16_bits:04x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

# Generate fp16_rsqrt vectors: {input[15:0], expected[15:0]} = 32 bits
# Uses the same inv_sqrt_lut.hex as RTL
def gen_fp16_rsqrt_vectors():
  lut_path = os.path.join(MEM_DIR, 'inv_sqrt_lut.hex')
  lut = load_lut16(lut_path, signed=False)

  # Exact squares
  test_floats = [1.0, 4.0, 9.0, 16.0, 25.0, 64.0, 100.0, 256.0]
  # Powers of 2
  test_floats += [0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 1024.0, 4096.0]
  # Small values near denorm boundary
  test_floats += [0.00006103515625, 0.000244140625, 0.001953125]
  # Large values
  test_floats += [10000.0, 50000.0, 65504.0]
  # Non-trivial mantissa
  test_floats += [1.5, 2.5, 3.14, 7.0, 13.0, 42.0, 0.1, 0.3]
  # Special cases
  test_floats += [0.0, float('inf')]

  vectors = []
  for fv in test_floats:
    bits = float_to_fp16(fv)
    expected = fp16_rsqrt_lut(bits, lut)
    vectors.append((bits, expected))

  rng = random.Random(48)
  for _ in range(30):
    bits = float_to_fp16(rng.uniform(0.001, 65504))
    expected = fp16_rsqrt_lut(bits, lut)
    vectors.append((bits, expected))

  path = os.path.join(MEM_DIR, 'fp16_rsqrt_vectors.hex')
  with open(path, 'w') as f:
    for in_bits, exp_bits in vectors:
      f.write(f'{in_bits:04x}{exp_bits:04x}\n')
  print(f'Wrote {len(vectors)} vectors to {path}')

if __name__ == '__main__':
  gen_fp16_add_vectors()
  gen_fp16_mul_vectors()
  gen_fp16_from_int8_vectors()
  gen_fp16_to_int8_vectors()
  gen_fp16_mac_vectors()
  gen_fp16_to_q167_vectors()
  gen_q115_to_fp16_vectors()
  gen_fp16_rsqrt_vectors()