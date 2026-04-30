"""
Generate rtl test vectors for matvec_fp16 and matvec_fp16_w8

Uses bit-exact fp16 models from rtl_ops.py (not numpy fp16)

Test 1: 4x4 matrix
Test 2: 8x4 matrix
Test 3: 32x4 matrix packed for matvec_fp16_w8 (64-bit packed weights)
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from rtl_ops import fp16_add, fp16_mul, fp16_from_int, fp16_from_float

MEM_DIR = os.path.join(os.path.dirname(__file__), '..', 'mem')

# RTL model matching RTL matvec_fp16.v accumulation order:
# for each row: acc = 0; for col: acc += fp16_from_int(w) * scale * in[col]
def fp16_matvec_rtl(weights_i8, in_vec_fp16, scale_fp16, out_dim, in_dim):
  result = []
  for r in range(out_dim):
    acc = 0x0000
    for c in range(in_dim):
      w_fp16 = fp16_from_int(weights_i8[r * in_dim + c])
      dequant = fp16_mul(w_fp16, scale_fp16)
      prod = fp16_mul(dequant, in_vec_fp16[c])
      acc = fp16_add(acc, prod)
    result.append(acc)
  return result

# Generate one matvec test case
def gen_test(name, out_dim, in_dim, seed):
  rng = random.Random(seed)

  # Random int8 weights [-50, 50]
  weights = [rng.randint(-50, 50) for _ in range(out_dim * in_dim)]
  # Random fp16 input vector
  in_vec = [fp16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(in_dim)]
  # Scale factor: 0.0625 = 1/16
  scale = fp16_from_float(0.0625)

  # RTL output
  out_vec = fp16_matvec_rtl(weights, in_vec, scale, out_dim, in_dim)

  # Write weight hex (row-major, int8 as unsigned bytes)
  w_path = os.path.join(MEM_DIR, f'matvec_fp16_{name}_weights.hex')
  with open(w_path, 'w') as f:
    for w in weights:
      f.write(f'{w & 0xFF:02x}\n')
  print(f'  Weights: {w_path} ({len(weights)} entries)')

  # Write input vector hex (fp16 bits)
  iv_path = os.path.join(MEM_DIR, f'matvec_fp16_{name}_input.hex')
  with open(iv_path, 'w') as f:
    for bits in in_vec:
      f.write(f'{bits:04x}\n')
  print(f'  Input:   {iv_path} ({in_dim} entries)')

  # Write expected output hex (fp16 bits)
  ov_path = os.path.join(MEM_DIR, f'matvec_fp16_{name}_expected.hex')
  with open(ov_path, 'w') as f:
    for bits in out_vec:
      f.write(f'{bits:04x}\n')
  print(f'  Output:  {ov_path} ({out_dim} entries)')

  print(f'  Scale:   0x{scale:04x}')
  return scale

# Pack weights for matvec_fp16_w8 BRAM layout
# Each 64-bit word at addr (group*K*in_dim + k*in_dim + col) holds 8 bytes:
#   byte L = weights[(group*K + k)*8 + L][col]
# Requires out_dim divisible by 8*K
def pack_weights_w8(weights_i8, out_dim, in_dim, K=4):
  assert out_dim % (8 * K) == 0, 'out_dim must be divisible by 8*K'
  n_packed = (out_dim // 8) * in_dim
  packed = []
  for j in range(n_packed):
    g = j // in_dim
    col = j % in_dim
    word = 0
    for L in range(8):
      byte = weights_i8[(8 * g + L) * in_dim + col] & 0xFF
      word |= byte << (L * 8)
    packed.append(word)
  return packed

# Generate one matvec_fp16_w8 test case with 64-bit packed weights
def gen_test_w8(name, out_dim, in_dim, seed):
  rng = random.Random(seed)

  weights = [rng.randint(-50, 50) for _ in range(out_dim * in_dim)]
  in_vec = [fp16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(in_dim)]
  scale = fp16_from_float(0.0625)

  out_vec = fp16_matvec_rtl(weights, in_vec, scale, out_dim, in_dim)
  packed = pack_weights_w8(weights, out_dim, in_dim)

  w_path = os.path.join(MEM_DIR, f'matvec_fp16_w8_{name}_weights.hex')
  with open(w_path, 'w') as f:
    for word in packed:
      f.write(f'{word:016x}\n')
  print(f'  Weights: {w_path} ({len(packed)} packed words)')

  iv_path = os.path.join(MEM_DIR, f'matvec_fp16_w8_{name}_input.hex')
  with open(iv_path, 'w') as f:
    for bits in in_vec:
      f.write(f'{bits:04x}\n')
  print(f'  Input:   {iv_path} ({in_dim} entries)')

  ov_path = os.path.join(MEM_DIR, f'matvec_fp16_w8_{name}_expected.hex')
  with open(ov_path, 'w') as f:
    for bits in out_vec:
      f.write(f'{bits:04x}\n')
  print(f'  Output:  {ov_path} ({out_dim} entries)')

  print(f'  Scale:   0x{scale:04x}')
  return scale

if __name__ == '__main__':
  print('Test 1: 4x4 matvec')
  s1 = gen_test('4x4', 4, 4, seed=100)
  print(f'\nTest 2: 8x4 matvec')
  s2 = gen_test('8x4', 8, 4, seed=101)
  print(f'\nTest 3: 32x4 matvec_fp16_w8 (packed)')
  s3 = gen_test_w8('32x4', 32, 4, seed=103)
  print(f'\nScale bits: test1=0x{s1:04x}, test2=0x{s2:04x}, test3=0x{s3:04x}')