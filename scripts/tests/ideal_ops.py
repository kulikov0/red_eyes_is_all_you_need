"""
Ideal float64 reference models for error analysis

These use real arithmetic (float64 LN, real softmax, real GELU) to show
how much quantization/rounding error the RTL-exact models introduce.
Not bit-exact with RTL - use rtl_ops.py for that.
"""

import math

from rtl_ops import (
    DIM, N_HEADS, HEAD_DIM, N_LAYERS, VOCAB, LN_OFFSETS, FRAC_W,
    to_signed8,
    fp16_to_float, fp16_from_float, fp16_from_int, fp16_mul, fp16_pack,
    _dequant_ln_fp16,
)


# 1/sqrt(x) via float math, returns fp16 (ideal, not RTL-exact)
def fp16_rsqrt(bits):
    f = fp16_to_float(bits)
    if f <= 0.0: return fp16_pack(0, 31, 0)
    return fp16_from_float(1.0 / math.sqrt(f))


# Float64 softmax for error analysis. Input: Q8.7 ints, output: float probabilities
def ideal_softmax(inputs):
    floats = [x / (1 << FRAC_W) for x in inputs]
    max_f = max(floats)
    exps = [math.exp(f - max_f) for f in floats]
    s = sum(exps)
    return [e / s for e in exps]


# GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
def gelu_float(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Ideal FP16 LayerNorm: dequant to fp16, then float64 math
def ideal_layernorm_fp16(x_fp16, gamma_fp16, beta_fp16):
    x_f = [fp16_to_float(v) for v in x_fp16]
    g_f = [fp16_to_float(v) for v in gamma_fp16]
    b_f = [fp16_to_float(v) for v in beta_fp16]
    n = len(x_f)
    mean = sum(x_f) / n
    var = sum((x - mean) ** 2 for x in x_f) / n
    inv_std = 1.0 / math.sqrt(var) if var > 1e-30 else 0.0
    return [(x_f[i] - mean) * inv_std * g_f[i] + b_f[i] for i in range(n)]


# Ideal GELU: float64 math. Returns float
def ideal_gelu_fp16(x_bits):
    x = fp16_to_float(x_bits)
    return gelu_float(x)


# Float64 ideal attention for error analysis. Returns list of fp16 bit patterns
def ideal_attention_fp16(x_fp16, layer, pos, kv_cache_f, qkv_w, proj_w,
                         qkv_scale_bits, proj_scale_bits):
    qkv_scale = fp16_to_float(qkv_scale_bits)
    proj_scale = fp16_to_float(proj_scale_bits)

    # QKV: float dequant
    qkv_float = []
    for r in range(384):
        acc = 0.0
        for c in range(DIM):
            x_f = fp16_to_float(x_fp16[c])
            w_f = to_signed8(qkv_w[r * DIM + c]) * qkv_scale
            acc += w_f * x_f
        qkv_float.append(acc)

    q_f = qkv_float[0:128]
    k_f = qkv_float[128:256]
    v_f = qkv_float[256:384]

    for i in range(128):
        h = i // HEAD_DIM
        d = i % HEAD_DIM
        kv_cache_f[(layer, 0, h, pos, d)] = k_f[i]
        kv_cache_f[(layer, 1, h, pos, d)] = v_f[i]

    sqrt_dk = math.sqrt(HEAD_DIM)
    head_out_f = [0.0] * 128

    for h in range(N_HEADS):
        q_h = [q_f[h * HEAD_DIM + d] for d in range(HEAD_DIM)]

        scores = []
        for p in range(pos + 1):
            dot = 0.0
            for d in range(HEAD_DIM):
                k_val = kv_cache_f.get((layer, 0, h, p, d), 0.0)
                dot += q_h[d] * k_val
            scores.append(dot / sqrt_dk)

        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        sum_e = sum(exps)
        attn_w = [e / sum_e for e in exps]

        for d in range(HEAD_DIM):
            acc = 0.0
            for p in range(pos + 1):
                v_val = kv_cache_f.get((layer, 1, h, p, d), 0.0)
                acc += attn_w[p] * v_val
            head_out_f[h * HEAD_DIM + d] = acc

    # Proj: float dequant
    out_f = []
    for r in range(DIM):
        acc = 0.0
        for c in range(DIM):
            w_f = to_signed8(proj_w[r * DIM + c]) * proj_scale
            acc += w_f * head_out_f[c]
        out_f.append(acc)

    return [fp16_from_float(v) for v in out_f]


# Float64 ideal transformer layer for error analysis. Returns list of fp16 bit patterns
def ideal_transformer_layer_fp16(x_fp16, layer, pos, kv_cache_f,
                                  ln_mem, qkv_w, proj_w, ff_up_w, ff_down_w,
                                  scales):
    # Ideal LN1
    gamma_sel = layer * 8 + 2
    gamma_bytes = ln_mem[LN_OFFSETS[gamma_sel]:LN_OFFSETS[gamma_sel] + DIM]
    beta_bytes = ln_mem[LN_OFFSETS[gamma_sel + 1]:LN_OFFSETS[gamma_sel + 1] + DIM]
    gamma_fp16 = _dequant_ln_fp16(gamma_bytes, scales['ln1_gamma'])
    beta_fp16 = _dequant_ln_fp16(beta_bytes, scales['ln1_beta'])
    ln1_f = ideal_layernorm_fp16(x_fp16, gamma_fp16, beta_fp16)
    ln1_out = [fp16_from_float(v) for v in ln1_f]

    # Ideal attention
    attn_out = ideal_attention_fp16(ln1_out, layer, pos, kv_cache_f,
                                     qkv_w, proj_w,
                                     scales['qkv'], scales['proj'])

    # Residual 1
    x_f = [fp16_to_float(v) for v in x_fp16]
    attn_f = [fp16_to_float(v) for v in attn_out]
    res1_f = [attn_f[i] + x_f[i] for i in range(DIM)]
    res1 = [fp16_from_float(v) for v in res1_f]

    # Ideal LN2
    gamma_sel2 = layer * 8 + 6
    gamma2_bytes = ln_mem[LN_OFFSETS[gamma_sel2]:LN_OFFSETS[gamma_sel2] + DIM]
    beta2_bytes = ln_mem[LN_OFFSETS[gamma_sel2 + 1]:LN_OFFSETS[gamma_sel2 + 1] + DIM]
    gamma2_fp16 = _dequant_ln_fp16(gamma2_bytes, scales['ln2_gamma'])
    beta2_fp16 = _dequant_ln_fp16(beta2_bytes, scales['ln2_beta'])
    ln2_f = ideal_layernorm_fp16(res1, gamma2_fp16, beta2_fp16)
    ln2_out = [fp16_from_float(v) for v in ln2_f]

    # FF_up: float dequant matvec
    ff_up_scale = fp16_to_float(scales['ff_up'])
    ff_up_f = []
    for r in range(512):
        acc = 0.0
        for c in range(DIM):
            w_f = to_signed8(ff_up_w[r * DIM + c]) * ff_up_scale
            acc += w_f * fp16_to_float(ln2_out[c])
        ff_up_f.append(acc)

    # Ideal GELU: float64 math
    gelu_f = [gelu_float(v) for v in ff_up_f]

    # FF_down: float dequant matvec
    ff_down_scale = fp16_to_float(scales['ff_down'])
    ff_down_f = []
    for r in range(DIM):
        acc = 0.0
        for c in range(512):
            w_f = to_signed8(ff_down_w[r * 512 + c]) * ff_down_scale
            acc += w_f * gelu_f[c]
        ff_down_f.append(acc)

    # Residual 2
    res1_f2 = [fp16_to_float(v) for v in res1]
    out_f = [ff_down_f[i] + res1_f2[i] for i in range(DIM)]
    return [fp16_from_float(v) for v in out_f]


# Ideal float64 embedding. Returns list of fp16 bit patterns
def ideal_embedding_fp16(token_id, position, tok_emb_w, pos_emb_w,
                         tok_scale_bits, pos_scale_bits):
    tok_scale = fp16_to_float(tok_scale_bits)
    pos_scale = fp16_to_float(pos_scale_bits)
    out = []
    for i in range(DIM):
        t = to_signed8(tok_emb_w[token_id * DIM + i]) * tok_scale
        p = to_signed8(pos_emb_w[position * DIM + i]) * pos_scale
        out.append(fp16_from_float(t + p))
    return out


# Ideal float64 matvec. Returns list of floats (not fp16)
def ideal_matvec_float(in_fp16, weights_u8, out_dim, in_dim, scale_bits):
    scale = fp16_to_float(scale_bits)
    out = []
    for r in range(out_dim):
        acc = 0.0
        for c in range(in_dim):
            w_f = to_signed8(weights_u8[r * in_dim + c]) * scale
            acc += w_f * fp16_to_float(in_fp16[c])
        out.append(acc)
    return out


# Ideal float64 full forward. Returns list of 256 fp16 logit bit patterns
def ideal_forward_fp16(token_id, position, kv_cache_f,
                       tok_emb_w, pos_emb_w, ln_mem,
                       layer_weights,
                       tok_scale_bits, pos_scale_bits,
                       lnf_gamma_scale_bits, lnf_beta_scale_bits):
    # Embedding
    x = ideal_embedding_fp16(token_id, position, tok_emb_w, pos_emb_w,
                             tok_scale_bits, pos_scale_bits)

    # 4 transformer layers
    for layer in range(N_LAYERS):
        lw = layer_weights[layer]
        x = ideal_transformer_layer_fp16(
            x, layer, position, kv_cache_f,
            ln_mem,
            lw['qkv'], lw['proj'], lw['ff_up'], lw['ff_down'],
            lw['scales'])

    # Final LayerNorm
    lnf_gamma_bytes = ln_mem[LN_OFFSETS[34]:LN_OFFSETS[34] + DIM]
    lnf_beta_bytes = ln_mem[LN_OFFSETS[35]:LN_OFFSETS[35] + DIM]
    lnf_gamma_fp16 = _dequant_ln_fp16(lnf_gamma_bytes, lnf_gamma_scale_bits)
    lnf_beta_fp16 = _dequant_ln_fp16(lnf_beta_bytes, lnf_beta_scale_bits)
    ln_f = ideal_layernorm_fp16(x, lnf_gamma_fp16, lnf_beta_fp16)
    x = [fp16_from_float(v) for v in ln_f]

    # Head projection (weight-tied with tok_emb)
    head_f = ideal_matvec_float(x, tok_emb_w, VOCAB, DIM, tok_scale_bits)
    return [fp16_from_float(v) for v in head_f]