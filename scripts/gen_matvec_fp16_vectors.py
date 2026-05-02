"""
Generate rtl test vectors for matvec_fp16_w16

Uses bit-exact fp16 models from rtl_ops.py, not numpy fp16

Test 3: 128x4 matrix packed for matvec_fp16_w16, 128-bit packed weights
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from rtl_ops import fp16_add, fp16_mul, fp16_from_int, fp16_from_float

MEM_DIR = os.path.join(os.path.dirname(__file__), '..', 'mem')

# RTL model matching matvec_fp16_w16.v per-row accumulation order:
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

# Each 128-bit word at addr (group*K*in_dim + k*in_dim + col) holds 16 bytes,
# byte L = weights[(group*K + k)*16 + L][col]. Requires out_dim divisible by 16*K
def pack_weights_w16(weights_i8, out_dim, in_dim, K=8):
  assert out_dim % (16 * K) == 0, 'out_dim must be divisible by 16*K'
  n_packed = (out_dim // 16) * in_dim
  packed = []
  for j in range(n_packed):
    g = j // in_dim
    col = j % in_dim
    word = 0
    for L in range(16):
      byte = weights_i8[(16 * g + L) * in_dim + col] & 0xFF
      word |= byte << (L * 8)
    packed.append(word)
  return packed

def gen_test_w16(name, out_dim, in_dim, seed):
  rng = random.Random(seed)

  weights = [rng.randint(-50, 50) for _ in range(out_dim * in_dim)]
  in_vec = [fp16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(in_dim)]
  scale = fp16_from_float(0.0625)

  out_vec = fp16_matvec_rtl(weights, in_vec, scale, out_dim, in_dim)
  packed = pack_weights_w16(weights, out_dim, in_dim)

  w_path = os.path.join(MEM_DIR, f'matvec_fp16_w16_{name}_weights.hex')
  with open(w_path, 'w') as f:
    for word in packed:
      f.write(f'{word:032x}\n')
  print(f'  Weights: {w_path} ({len(packed)} packed words)')

  iv_path = os.path.join(MEM_DIR, f'matvec_fp16_w16_{name}_input.hex')
  with open(iv_path, 'w') as f:
    for bits in in_vec:
      f.write(f'{bits:04x}\n')
  print(f'  Input:   {iv_path} ({in_dim} entries)')

  ov_path = os.path.join(MEM_DIR, f'matvec_fp16_w16_{name}_expected.hex')
  with open(ov_path, 'w') as f:
    for bits in out_vec:
      f.write(f'{bits:04x}\n')
  print(f'  Output:  {ov_path} ({out_dim} entries)')

  print(f'  Scale:   0x{scale:04x}')
  return scale

if __name__ == '__main__':
  print('Test 3: 128x4 matvec_fp16_w16 packed')
  s3 = gen_test_w16('128x4', 128, 4, seed=103)
  print(f'\nScale bits: test3=0x{s3:04x}')
