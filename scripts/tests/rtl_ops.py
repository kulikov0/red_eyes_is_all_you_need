"""
RTL-exact reference models for W8A16 validation

Pure-Python fp16 primitives (fp16_add, fp16_mul, etc) that replicate RTL
rounding exactly, plus composite functions (layernorm, attention, matvec,
transformer layer, full forward) built from those primitives.

Ideal float64 models for error analysis are in ideal_ops.py.
"""

import re
import os
import struct

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEM = os.path.join(PROJ, "mem")
SCALES_VH = os.path.join(PROJ, "rtl", "weight_scales.vh")

# Architecture
DIM = 128
N_HEADS = 8
HEAD_DIM = 16
N_LAYERS = 4
VOCAB = 256
N_SM = 256

# Softmax
FRAC_W = 7
D_CLIP = 2048
LN2_Q7 = 89
LN1PS_LUT = [0, 8, 15, 22, 29, 35, 41, 47, 53, 58, 63, 68, 73, 78, 82, 87]


# LN tensor_sel offsets into ln_params.hex (parsed from weight_store.v)
def _parse_ln_offsets():
    ws_path = os.path.join(PROJ, "rtl", "weight_store.v")
    pat = re.compile(r"6'd(\d+):\s*ln_offset\s*=\s*12'd(\d+);")
    offsets = {}
    with open(ws_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                offsets[int(m.group(1))] = int(m.group(2))
    return offsets

LN_OFFSETS = _parse_ln_offsets()


def to_signed8(b):
    return b - 256 if b >= 128 else b


def load_hex(path):
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            vals.append(int(s, 16))
    return vals


def load_lut16(path, signed=False):
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            v = int(s, 16)
            if signed and v >= 32768:
                v -= 65536
            vals.append(v)
    return vals


# Parse fp16 scale from weight_scales.vh, returns float
def parse_scale(name, scales_vh=None):
    if scales_vh is None:
        scales_vh = SCALES_VH
    pat = re.compile(
        r"localparam\s+\[15:0\]\s+" + name + r"\s*=\s*16'h([0-9a-fA-F]{4})")
    with open(scales_vh) as f:
        for line in f:
            m = pat.search(line)
            if m:
                bits = int(m.group(1), 16)
                return fp16_to_float(bits)
    raise ValueError(f"Scale {name} not found in {scales_vh}")


# Parse fp16 scale, returns fp16 bit pattern (uint16)
def parse_scale_bits(name, scales_vh=None):
    if scales_vh is None:
        scales_vh = SCALES_VH
    pat = re.compile(
        r"localparam\s+\[15:0\]\s+" + name + r"\s*=\s*16'h([0-9a-fA-F]{4})")
    with open(scales_vh) as f:
        for line in f:
            m = pat.search(line)
            if m:
                return int(m.group(1), 16)
    raise ValueError(f"Scale {name} not found in {scales_vh}")



# ---- FP16 IEEE 754 half-precision primitives ----
# Flush-to-zero for denormals, round-to-nearest-even
# These match RTL fp16_add_comb / fp16_mul_comb bit-for-bit

FP16_BIAS = 15


def fp16_unpack(bits):
    s = (bits >> 15) & 1
    e = (bits >> 10) & 0x1F
    f = bits & 0x3FF
    m = (1 << 10) | f if e != 0 else 0
    return s, e, m


def fp16_pack(s, e, f):
    return (s << 15) | ((e & 0x1F) << 10) | (f & 0x3FF)


# Matches RTL fp16_add_comb in matvec_fp16.v exactly
def fp16_add(a, b):
    a_s = (a >> 15) & 1
    a_e = (a >> 10) & 0x1F
    a_f = a & 0x3FF
    b_s = (b >> 15) & 1
    b_e = (b >> 10) & 0x1F
    b_f = b & 0x3FF
    a_zero = (a_e == 0)
    b_zero = (b_e == 0)
    a_inf = (a_e == 31) and (a_f == 0)
    b_inf = (b_e == 31) and (b_f == 0)
    a_nan = (a_e == 31) and (a_f != 0)
    b_nan = (b_e == 31) and (b_f != 0)
    a_full = 0 if a_zero else (0x400 | a_f)
    b_full = 0 if b_zero else (0x400 | b_f)
    a_ge_b = (a_e > b_e) or (a_e == b_e and a_full >= b_full)
    if a_ge_b:
        lg_s, lg_e, lg_m = a_s, a_e, a_full
        sm_s, sm_e, sm_m = b_s, b_e, b_full
    else:
        lg_s, lg_e, lg_m = b_s, b_e, b_full
        sm_s, sm_e, sm_m = a_s, a_e, a_full
    exp_diff = lg_e - sm_e
    # lg_ext = {0, lg_mant[10:0], 2'b00} = 14 bits
    lg_ext = (lg_m << 2) & 0x3FFF
    # sm_wide = {0, sm_mant[10:0], 2'b00, 13'b0} = 27 bits, then >> exp_diff
    sm_wide = (sm_m << 2) << 13  # 27-bit
    sm_shifted = sm_wide >> exp_diff
    sm_ext = (sm_shifted >> 13) & 0x3FFF
    sticky = (sm_shifted & 0x1FFF) != 0
    eff_sub = lg_s ^ sm_s
    if eff_sub:
        mant_sum = lg_ext - sm_ext
    else:
        mant_sum = lg_ext + sm_ext
    mant_sum &= 0x7FFF
    sum_is_zero = (mant_sum == 0)
    # LOD: find highest set bit
    lod = 0
    for i in range(15):
        if mant_sum & (1 << i):
            lod = i
    overflow = (lod == 13) or (lod == 14)
    if lod > 12:
        rshift = lod - 12
        lshift = 0
    else:
        rshift = 0
        lshift = 12 - lod
    if sum_is_zero:
        norm_mant = 0
    elif overflow:
        norm_mant = mant_sum >> rshift
    else:
        norm_mant = mant_sum << lshift
    norm_mant &= 0x7FFF
    if sum_is_zero:
        exp_adj = 0
    elif overflow:
        exp_adj = lg_e + rshift
    else:
        exp_adj = lg_e - lshift
    trunc_mant = (norm_mant >> 2) & 0x3FF
    guard_bit = (norm_mant >> 1) & 1
    round_bit = norm_mant & 1
    extra_sticky = (mant_sum & 1) if overflow else 0
    sticky_bit = 1 if (sticky or extra_sticky) else 0
    use_sticky = sticky_bit & (1 if not eff_sub else 0)
    round_up = guard_bit & (round_bit | use_sticky | (trunc_mant & 1))
    rounded = trunc_mant + round_up
    round_ovf = rounded > 0x3FF
    if round_ovf:
        rounded &= 0x3FF
        final_exp = exp_adj + 1
    else:
        final_exp = exp_adj
    normal = (lg_s << 15) | ((final_exp & 0x1F) << 10) | (rounded & 0x3FF)
    exp_of = final_exp >= 31
    exp_uf = (final_exp <= 0) and not sum_is_zero
    inf_r = (lg_s << 15) | (31 << 10)
    zero_r = lg_s << 15
    nan_r = 0x7E00
    if a_nan or b_nan: return nan_r
    if a_inf and b_inf and eff_sub: return nan_r
    if a_inf: return (a_s << 15) | (31 << 10)
    if b_inf: return (b_s << 15) | (31 << 10)
    if a_zero and b_zero: return ((a_s & b_s) << 15)
    if a_zero: return b
    if b_zero: return a
    if sum_is_zero: return 0
    if exp_uf: return zero_r
    if exp_of: return inf_r
    return normal


# Matches RTL fp16_mul_comb in fp16_mac.v exactly
def fp16_mul(a, b):
    a_s = (a >> 15) & 1
    a_e = (a >> 10) & 0x1F
    a_f = a & 0x3FF
    b_s = (b >> 15) & 1
    b_e = (b >> 10) & 0x1F
    b_f = b & 0x3FF
    r_s = a_s ^ b_s
    a_zero = (a_e == 0)
    b_zero = (b_e == 0)
    a_inf = (a_e == 31) and (a_f == 0)
    b_inf = (b_e == 31) and (b_f == 0)
    a_nan = (a_e == 31) and (a_f != 0)
    b_nan = (b_e == 31) and (b_f != 0)
    a_full = 0x400 | a_f
    b_full = 0x400 | b_f
    prod = a_full * b_full
    exp_raw = a_e + b_e - 15
    norm_shift = (prod >> 21) & 1
    if norm_shift:
        trunc = (prod >> 11) & 0x3FF
        g = (prod >> 10) & 1
        r = (prod >> 9) & 1
        s = 1 if (prod & 0x1FF) else 0
    else:
        trunc = (prod >> 10) & 0x3FF
        g = (prod >> 9) & 1
        r = (prod >> 8) & 1
        s = 1 if (prod & 0xFF) else 0
    round_up = g & (r | s | (trunc & 1))
    rounded = trunc + round_up
    round_ovf = rounded > 0x3FF
    if round_ovf:
        rounded &= 0x3FF
    final_exp = exp_raw + norm_shift + (1 if round_ovf else 0)
    normal = (r_s << 15) | ((final_exp & 0x1F) << 10) | (rounded & 0x3FF)
    inf_r = (r_s << 15) | (31 << 10)
    nan_r = 0x7E00
    zero_r = r_s << 15
    if a_nan or b_nan: return nan_r
    if (a_inf and b_zero) or (b_inf and a_zero): return nan_r
    if a_inf or b_inf: return inf_r
    if a_zero or b_zero: return zero_r
    if final_exp <= 0: return zero_r
    if final_exp >= 31: return inf_r
    return normal


def fp16_mac(acc, a, b):
    return fp16_add(acc, fp16_mul(a, b))


def fp16_negate(bits):
    return bits ^ 0x8000


# Signed integer to fp16 (exact for |val| <= 2048)
def fp16_from_int(val):
    if val == 0:
        return 0
    s = 0
    if val < 0:
        s = 1
        val = -val
    k = val.bit_length() - 1
    e = k + FP16_BIAS
    if e >= 31: return fp16_pack(s, 31, 0)
    if k <= 10:
        frac = (val << (10 - k)) & 0x3FF
    else:
        frac = (val >> (k - 10)) & 0x3FF
    return fp16_pack(s, e, frac)


# Python float to fp16 bit pattern with RNE
def fp16_from_float(f):
    if f == 0.0:
        return 0
    bits = struct.unpack('>I', struct.pack('>f', f))[0]
    s = (bits >> 31) & 1
    e32 = (bits >> 23) & 0xFF
    m32 = bits & 0x7FFFFF
    e16 = e32 - 127 + 15
    if e16 >= 31: return fp16_pack(s, 31, 0)
    if e16 <= 0: return s << 15
    frac = m32 >> 13
    g = (m32 >> 12) & 1
    rs = m32 & 0xFFF
    if g and (rs or (frac & 1)):
        frac += 1
        if frac > 0x3FF:
            frac = 0
            e16 += 1
            if e16 >= 31: return fp16_pack(s, 31, 0)
    return fp16_pack(s, e16, frac)


# fp16 bit pattern to Python float
def fp16_to_float(bits):
    s, e, m = fp16_unpack(bits)
    if e == 0: return 0.0
    if e == 31:
        if bits & 0x3FF: return float('nan')
        return float('-inf') if s else float('inf')
    val = m * (2.0 ** (e - 25))
    return -val if s else val



# 1/sqrt(x) via LUT matching RTL fp16_rsqrt.v exactly
def fp16_rsqrt_lut(bits, lut):
    e = (bits >> 10) & 0x1F
    f = bits & 0x3FF
    if e == 0: return 0x7C00
    if e == 31: return 0x0000
    parity = 1 if (e & 1) == 0 else 0
    lut_addr = (parity << 8) | ((f >> 2) & 0xFF)
    lut_out = lut[lut_addr]
    k_is_15 = (lut_out >> 15) & 1
    base = 15 - e + parity
    half_base = base >> 1
    out_e_s = half_base + (15 if k_is_15 else 14)
    raw_frac = (lut_out >> 4) & 0x3FF
    g = (lut_out >> 3) & 1
    r = (lut_out >> 2) & 1
    s = lut_out & 0x3
    round_up = g & (r | (1 if s else 0) | (raw_frac & 1))
    rounded = raw_frac + round_up
    round_ovf = rounded > 0x3FF
    if k_is_15:
        out_frac = 0
        out_e = out_e_s
    elif round_ovf:
        out_frac = 0
        out_e = out_e_s + 1
    else:
        out_frac = rounded & 0x3FF
        out_e = out_e_s
    if out_e <= 0: return 0x0000
    if out_e >= 31: return 0x7C00
    return (out_e << 10) | out_frac



# FP16 LayerNorm matching RTL bit-for-bit. Returns list of fp16 bit patterns
# isqrt_lut: 512x16 LUT from inv_sqrt_lut.hex (for RTL-exact rsqrt)
def rtl_layernorm_fp16(x_fp16, gamma_fp16, beta_fp16, isqrt_lut):
    n = len(x_fp16)
    inv_n = fp16_from_float(1.0 / n)
    total = 0
    for v in x_fp16:
        total = fp16_add(total, v)
    mean = fp16_mul(total, inv_n)
    neg_mean = fp16_negate(mean)
    var_acc = 0
    for v in x_fp16:
        diff = fp16_add(v, neg_mean)
        var_acc = fp16_mac(var_acc, diff, diff)
    var = fp16_mul(var_acc, inv_n)
    inv_std = fp16_rsqrt_lut(var, isqrt_lut)
    out = []
    for i in range(n):
        diff = fp16_add(x_fp16[i], neg_mean)
        normed = fp16_mul(diff, inv_std)
        scaled = fp16_mul(normed, gamma_fp16[i])
        out.append(fp16_add(scaled, beta_fp16[i]))
    return out, mean, var, inv_std


# ---- End FP16 primitives ----


# Bipartite exp(-d) approximation matching RTL
def bipartite_exp(d_q47, lut0, lut1):
    if d_q47 >= D_CLIP:
        return 0
    x0 = (d_q47 >> 6) & 0x1F
    x1 = (d_q47 >> 3) & 0x07
    x2 = d_q47 & 0x07
    val = lut0[(x0 << 3) | x1] + lut1[(x0 << 3) | x2]
    if val < 0:
        return 0
    if val > 32768:
        return 32768
    return val


# LOD: find leading one position in a 24-bit value
def lod24(val):
    if val == 0:
        return 0
    k = 0
    for i in range(24):
        if val & (1 << i):
            k = i
    return k


# Extract 4-bit mantissa below leading one
def lod_mantissa(val, k):
    if k >= 4:
        return (val >> (k - 4)) & 0xF
    else:
        return (val << (4 - k)) & 0xF


# Full hardware softmax (bipartite LUT + LOD division)
def rtl_softmax(inputs, lut0, lut1):
    max_val = max(inputs)

    # EXP_ACC pass
    sum_acc = 0
    for x in inputs:
        d_raw = max_val - x
        d_int = d_raw >> FRAC_W
        d_frac = d_raw & ((1 << FRAC_W) - 1)
        if d_int >= 16:
            d_q47 = D_CLIP
        else:
            d_q47 = (d_int << FRAC_W) | d_frac
        sum_acc += bipartite_exp(d_q47, lut0, lut1)
    sum_acc = min(sum_acc, 0xFFFFFF)

    # LN_SUM
    if sum_acc == 0:
        ln_offset = D_CLIP
    else:
        k = lod24(sum_acc)
        s = lod_mantissa(sum_acc, k)
        k_minus_15 = k - 15
        kln2 = k_minus_15 * LN2_Q7
        ln1ps = LN1PS_LUT[s]
        ln_raw = kln2 + ln1ps
        if ln_raw < 0:
            ln_offset = 0
        elif ln_raw >= D_CLIP:
            ln_offset = D_CLIP
        else:
            ln_offset = ln_raw

    # NORM pass
    outputs = []
    for x in inputs:
        d_raw = max_val - x
        d_int = d_raw >> FRAC_W
        d_frac = d_raw & ((1 << FRAC_W) - 1)
        d_overflow = d_int >= 16
        if d_overflow:
            d_q47 = D_CLIP
        else:
            d_q47 = (d_int << FRAC_W) | d_frac
        d_plus_ln = d_q47 + ln_offset
        if d_overflow or d_plus_ln >= D_CLIP:
            d_norm = D_CLIP
        else:
            d_norm = d_plus_ln
        outputs.append(bipartite_exp(d_norm, lut0, lut1))

    return outputs


# Load PWL coefficients from gelu_pwl.hex
def load_gelu_pwl(path=None):
    if path is None:
        path = os.path.join(MEM, "gelu_pwl.hex")
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            vals.append(int(s, 16))
    breaks = vals[0:15]
    slopes = [vals[15 + 2 * i] for i in range(16)]
    icepts = [vals[15 + 2 * i + 1] for i in range(16)]
    return breaks, slopes, icepts


# FP16 unsigned magnitude comparison (works for positive fp16)
def _fp16_ge(a_bits, b_bits):
    return (a_bits & 0x7FFF) >= (b_bits & 0x7FFF)


# FP16 PWL GELU matching RTL gelu.v bit-for-bit. Returns fp16 bits
def rtl_gelu_fp16(x_bits, breaks, slopes, icepts):
    SAT_THRESH = 0x44F3
    sign = (x_bits >> 15) & 1
    abs_x = x_bits & 0x7FFF
    is_zero = (abs_x == 0)

    # Saturation check
    saturated = (abs_x >= SAT_THRESH)

    # Segment index = count of breakpoints exceeded
    seg = 0
    for k in range(15):
        if abs_x >= breaks[k]:
            seg += 1

    # erf_pos = slope * |x| + intercept
    product = fp16_mul(slopes[seg], abs_x)
    erf_pos = fp16_add(product, icepts[seg])

    # Saturated override
    if saturated:
        erf_pos = 0x3C00  # 1.0

    # Sign restore
    if sign and not is_zero:
        erf_val = 0x8000 | (erf_pos & 0x7FFF)
    else:
        erf_val = erf_pos

    # 1.0 + erf
    one_plus_erf = fp16_add(0x3C00, erf_val)

    # half_x = x * 0.5 via exponent decrement
    x_exp = (x_bits >> 10) & 0x1F
    if x_exp <= 0:
        half_x = 0x0000
    else:
        half_x = (x_bits & 0x8000) | ((x_exp - 1) << 10) | (x_bits & 0x03FF)

    # y = half_x * (1 + erf)
    return fp16_mul(half_x, one_plus_erf)


# FP16 matvec matching RTL matvec_fp16.v bit-for-bit
# int8 weights dequanted to fp16 at runtime: fp16_from_int(w) * scale
def rtl_matvec_fp16(in_vec_fp16, weights_u8, out_dim, in_dim, scale_bits):
    out = []
    for r in range(out_dim):
        acc = 0x0000  # fp16 zero
        for c in range(in_dim):
            w_int = to_signed8(weights_u8[r * in_dim + c])
            w_fp16 = fp16_from_int(w_int)
            w_dequant = fp16_mul(w_fp16, scale_bits)
            prod = fp16_mul(w_dequant, in_vec_fp16[c])
            acc = fp16_add(acc, prod)
        out.append(acc)
    return out


# Convert fp16 bit pattern to signed Q16.7 integer (24-bit, value * 128, rounded)
# Matches RTL fp16_to_q167.v. Full fp16 range fits without clamping
def fp16_to_q167(bits):
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    full_mant = (1 << 10) | mant  # 11 bits: 1.mant

    MAX_MAG = 0x7FFFFF  # 8388607

    if exp == 0:
        return 0
    if exp == 31:
        return (-MAX_MAG - 1) & 0xFFFFFF if sign else MAX_MAG

    # Q16.7 = round(full_mant * 2^(exp-18))
    shift = 18 - exp
    if shift <= 0:
        # Left shift (no rounding needed)
        mag = full_mant << (-shift)
    elif shift >= 12:
        # full_mant is 11 bits, shift >= 12 means result rounds to 0 or 1
        if shift == 11:
            guard = 1  # implicit bit is always 1
            sticky = 1 if (mant != 0) else 0
            mag = 1 if (guard and sticky) else 0
        else:
            mag = 0
        if sign:
            return (-mag) & 0xFFFFFF
        return mag & 0xFFFFFF
    else:
        mag = full_mant >> shift
        guard = (full_mant >> (shift - 1)) & 1
        if shift >= 2:
            below_guard = full_mant & ((1 << (shift - 1)) - 1)
            sticky = 1 if below_guard else 0
        else:
            sticky = 0
        round_up = guard & (sticky | (mag & 1))
        mag = mag + round_up

    if mag > MAX_MAG:
        mag = MAX_MAG

    if sign:
        return (-mag) & 0xFFFFFF
    return mag & 0xFFFFFF


# Convert unsigned Q1.15 (0..32768) to fp16 bit pattern
# Matches RTL q115_to_fp16.v
def q115_to_fp16(val):
    if val == 0:
        return 0x0000

    # LOD: find MSB position
    k = 0
    for i in range(16):
        if val & (1 << i):
            k = i

    # fp16 exponent = k (val/32768 = val*2^-15, MSB at k means ~2^(k-15))
    exp_val = k

    # Extract mantissa bits below leading one
    if k <= 10:
        if k == 0:
            frac = 0
        else:
            frac = ((val << (10 - k)) & 0x3FF)
    else:
        rshift = k - 10
        frac = (val >> rshift) & 0x3FF
        guard_pos = rshift - 1
        guard = (val >> guard_pos) & 1
        if guard_pos > 0:
            sticky = 1 if (val & ((1 << guard_pos) - 1)) else 0
        else:
            sticky = 0
        round_up = guard & (sticky | (frac & 1))
        frac = frac + round_up
        if frac >= 1024:
            frac = 0
            exp_val = exp_val + 1

    if exp_val >= 31:
        return 0x7C00

    return (exp_val << 10) | (frac & 0x3FF)


# Helper: interpret 24-bit unsigned Q16.7 as signed Python int
def _q167_to_signed(val):
    val = val & 0xFFFFFF
    if val >= 0x800000:
        return val - 0x1000000
    return val


# FP16 attention matching W8A16 RTL. Returns list of fp16 bit patterns
def rtl_attention_fp16(x_fp16, layer, pos, kv_cache, qkv_w, proj_w,
                          lut0, lut1, qkv_scale_bits, proj_scale_bits):
    INV_SQRT_DK = 0x3400  # 0.25 = 1/sqrt(16)

    # QKV projection: matvec_fp16
    qkv = rtl_matvec_fp16(x_fp16, qkv_w, 384, DIM, qkv_scale_bits)

    q_fp16 = qkv[0:128]
    k_fp16 = qkv[128:256]
    v_fp16 = qkv[256:384]

    # Store K and V directly as fp16
    for i in range(128):
        h = i // HEAD_DIM
        d = i % HEAD_DIM
        kv_cache[(layer, 0, h, pos, d)] = k_fp16[i]
        kv_cache[(layer, 1, h, pos, d)] = v_fp16[i]

    head_out = [0x0000] * 128
    for h in range(N_HEADS):
        q_h = [q_fp16[h * HEAD_DIM + d] for d in range(HEAD_DIM)]

        # Score: fp16 dot product + 1/sqrt(d_k) scaling -> Q16.7 for softmax
        scores_q167 = []
        for p in range(pos + 1):
            acc = 0x0000
            for d in range(HEAD_DIM):
                k_val = kv_cache.get((layer, 0, h, p, d), 0x0000)
                prod = fp16_mul(q_h[d], k_val)
                acc = fp16_add(acc, prod)
            score_scaled = fp16_mul(acc, INV_SQRT_DK)
            scores_q167.append(_q167_to_signed(fp16_to_q167(score_scaled)))

        # Pad with min Q16.7 (24-bit)
        sm_input = scores_q167 + [-8388608] * (N_SM - len(scores_q167))

        # Softmax (unchanged bipartite LUT)
        attn = rtl_softmax(sm_input, lut0, lut1)

        # AV: fp16 accumulation
        for d in range(HEAD_DIM):
            acc = 0x0000
            for p in range(pos + 1):
                attn_fp16 = q115_to_fp16(attn[p])
                v_val = kv_cache.get((layer, 1, h, p, d), 0x0000)
                prod = fp16_mul(attn_fp16, v_val)
                acc = fp16_add(acc, prod)
            head_out[h * HEAD_DIM + d] = acc

    # Proj projection: matvec_fp16
    return rtl_matvec_fp16(head_out, proj_w, DIM, DIM, proj_scale_bits)


# Dequant LN gamma/beta bytes to fp16 using scale
def _dequant_ln_fp16(ln_bytes, scale_bits):
    return [fp16_mul(fp16_from_int(to_signed8(b)), scale_bits) for b in ln_bytes]


# FP16 transformer layer matching W8A16 RTL bit-for-bit
# Returns list of 128 fp16 bit patterns
def rtl_transformer_layer_fp16(x_fp16, layer, pos, kv_cache,
                                   ln_mem, isqrt_lut,
                                   qkv_w, proj_w, ff_up_w, ff_down_w,
                                   lut0, lut1, scales,
                                   breaks, slopes, icepts):
    # LN1
    gamma_sel = layer * 8 + 2
    gamma_bytes = ln_mem[LN_OFFSETS[gamma_sel]:LN_OFFSETS[gamma_sel] + DIM]
    beta_bytes = ln_mem[LN_OFFSETS[gamma_sel + 1]:LN_OFFSETS[gamma_sel + 1] + DIM]
    gamma_fp16 = _dequant_ln_fp16(gamma_bytes, scales['ln1_gamma'])
    beta_fp16 = _dequant_ln_fp16(beta_bytes, scales['ln1_beta'])
    ln1_out, _, _, _ = rtl_layernorm_fp16(x_fp16, gamma_fp16, beta_fp16, isqrt_lut)

    # Attention
    attn_out = rtl_attention_fp16(ln1_out, layer, pos, kv_cache,
                                      qkv_w, proj_w, lut0, lut1,
                                      scales['qkv'], scales['proj'])

    # Residual 1: fp16 add
    res1 = [fp16_add(attn_out[i], x_fp16[i]) for i in range(DIM)]

    # LN2
    gamma_sel2 = layer * 8 + 6
    gamma2_bytes = ln_mem[LN_OFFSETS[gamma_sel2]:LN_OFFSETS[gamma_sel2] + DIM]
    beta2_bytes = ln_mem[LN_OFFSETS[gamma_sel2 + 1]:LN_OFFSETS[gamma_sel2 + 1] + DIM]
    gamma2_fp16 = _dequant_ln_fp16(gamma2_bytes, scales['ln2_gamma'])
    beta2_fp16 = _dequant_ln_fp16(beta2_bytes, scales['ln2_beta'])
    ln2_out, _, _, _ = rtl_layernorm_fp16(res1, gamma2_fp16, beta2_fp16, isqrt_lut)

    # FF_up: matvec_fp16 (128->512)
    ff_up_out = rtl_matvec_fp16(ln2_out, ff_up_w, 512, DIM, scales['ff_up'])

    # GELU: fp16 PWL per element
    gelu_out = [rtl_gelu_fp16(ff_up_out[i], breaks, slopes, icepts) for i in range(512)]

    # FF_down: matvec_fp16 (512->128)
    ff_down_out = rtl_matvec_fp16(gelu_out, ff_down_w, DIM, 512, scales['ff_down'])

    # Residual 2: fp16 add
    return [fp16_add(ff_down_out[i], res1[i]) for i in range(DIM)]


# FP16 embedding matching RTL embedding.v bit-for-bit
# tok_emb_w, pos_emb_w: flat uint8 lists from hex files
# Returns list of 128 fp16 bit patterns
def rtl_embedding_fp16(token_id, position, tok_emb_w, pos_emb_w,
                           tok_scale_bits, pos_scale_bits):
    out = []
    for i in range(DIM):
        tok_byte = tok_emb_w[token_id * DIM + i]
        pos_byte = pos_emb_w[position * DIM + i]
        tok_fp16 = fp16_from_int(to_signed8(tok_byte))
        pos_fp16 = fp16_from_int(to_signed8(pos_byte))
        tok_dq = fp16_mul(tok_fp16, tok_scale_bits)
        pos_dq = fp16_mul(pos_fp16, pos_scale_bits)
        out.append(fp16_add(tok_dq, pos_dq))
    return out


# FP16 head projection matching RTL (weight-tied with tok_emb)
# 256 output logits from 128-dim input, returns list of 256 fp16 bit patterns
def rtl_head_projection_fp16(x_fp16, tok_emb_w, tok_scale_bits):
    return rtl_matvec_fp16(x_fp16, tok_emb_w, VOCAB, DIM, tok_scale_bits)


# Full RTL-exact inference: one token through embedding -> 4 layers -> ln_f -> head
# Returns list of 256 fp16 logit bit patterns
def rtl_forward_fp16(token_id, position, kv_cache,
                         tok_emb_w, pos_emb_w, ln_mem, isqrt_lut,
                         layer_weights, lut0, lut1,
                         breaks, slopes, icepts,
                         tok_scale_bits, pos_scale_bits,
                         lnf_gamma_scale_bits, lnf_beta_scale_bits):
    # Embedding
    x = rtl_embedding_fp16(token_id, position, tok_emb_w, pos_emb_w,
                               tok_scale_bits, pos_scale_bits)

    # 4 transformer layers
    for layer in range(N_LAYERS):
        lw = layer_weights[layer]
        x = rtl_transformer_layer_fp16(
            x, layer, position, kv_cache,
            ln_mem, isqrt_lut,
            lw['qkv'], lw['proj'], lw['ff_up'], lw['ff_down'],
            lut0, lut1, lw['scales'],
            breaks, slopes, icepts)

    # Final LayerNorm
    lnf_gamma_bytes = ln_mem[LN_OFFSETS[34]:LN_OFFSETS[34] + DIM]
    lnf_beta_bytes = ln_mem[LN_OFFSETS[35]:LN_OFFSETS[35] + DIM]
    lnf_gamma_fp16 = _dequant_ln_fp16(lnf_gamma_bytes, lnf_gamma_scale_bits)
    lnf_beta_fp16 = _dequant_ln_fp16(lnf_beta_bytes, lnf_beta_scale_bits)
    x, _, _, _ = rtl_layernorm_fp16(x, lnf_gamma_fp16, lnf_beta_fp16, isqrt_lut)

    # Head projection (weight-tied with tok_emb)
    return rtl_head_projection_fp16(x, tok_emb_w, tok_scale_bits)