"""
RTL-exact inference using pure Python fp16 primitives

Runs autoregressive text generation matching the hardware pipeline
bit-for-bit: embedding -> 4 transformer layers -> ln_f -> head proj -> argmax
No xsim, no ideal model - just the RTL golden model from rtl_ops.py

Usage:
  python3 rtl_inference.py                  # default: token 65 ('A'), 20 tokens
  python3 rtl_inference.py 72 50            # token 72 ('H'), 50 tokens
  python3 rtl_inference.py "Hello"          # prompt string, 20 tokens
  python3 rtl_inference.py "Hello" 50       # prompt string, 50 tokens
"""

import sys
import os
import time

from rtl_ops import (
    DIM, N_LAYERS, VOCAB, MEM,
    fp16_to_float,
    load_hex, load_lut16, load_gelu_pwl, parse_scale_bits,
    rtl_forward_fp16,
)


# Human-readable token: printable ASCII or hex
def token_repr(tok):
    if 0x20 <= tok < 0x7F:
        return chr(tok)
    if tok == 0x0A:
        return "\\n"
    if tok == 0x0D:
        return "\\r"
    if tok == 0x09:
        return "\\t"
    return f"<0x{tok:02x}>"


# FP16 argmax matching RTL sampler behavior
def fp16_argmax(logits_fp16):
    best_val = 0xFC00
    best_idx = 0
    for i in range(len(logits_fp16)):
        cur = logits_fp16[i]
        a_neg = (cur >> 15) & 1
        b_neg = (best_val >> 15) & 1
        mag_gt = (cur & 0x7FFF) > (best_val & 0x7FFF)
        if a_neg != b_neg:
            a_gt_b = b_neg == 1
        elif a_neg:
            a_gt_b = not mag_gt
        else:
            a_gt_b = mag_gt
        if a_gt_b:
            best_val = cur
            best_idx = i
    return best_idx


def load_all_resources():
    ln_mem = load_hex(os.path.join(MEM, "ln_params.hex"))
    isqrt_lut = load_lut16(os.path.join(MEM, "inv_sqrt_lut.hex"), signed=False)
    lut0 = load_lut16(os.path.join(MEM, "exp_lut0.hex"), signed=False)
    lut1 = load_lut16(os.path.join(MEM, "exp_lut1.hex"), signed=True)
    breaks, slopes, icepts = load_gelu_pwl()

    tok_emb_w = load_hex(os.path.join(MEM, "tok_emb_weight.hex"))
    pos_emb_w = load_hex(os.path.join(MEM, "pos_emb_weight.hex"))

    tok_scale = parse_scale_bits("SCALE_TOK_EMB_WEIGHT")
    pos_scale = parse_scale_bits("SCALE_POS_EMB_WEIGHT")
    lnf_g_scale = parse_scale_bits("SCALE_LN_F_WEIGHT")
    lnf_b_scale = parse_scale_bits("SCALE_LN_F_BIAS")

    layer_weights = {}
    for L in range(N_LAYERS):
        qkv_w = load_hex(os.path.join(MEM, f"block{L}_attn_qkv_weight.hex"))
        proj_w = load_hex(os.path.join(MEM, f"block{L}_attn_proj_weight.hex"))
        ff_up_w = load_hex(os.path.join(MEM, f"block{L}_ff_up_weight.hex"))
        ff_down_w = load_hex(os.path.join(MEM, f"block{L}_ff_down_weight.hex"))
        scales = {
            'ln1_gamma': parse_scale_bits(f"SCALE_BLOCK{L}_LN1_WEIGHT"),
            'ln1_beta':  parse_scale_bits(f"SCALE_BLOCK{L}_LN1_BIAS"),
            'ln2_gamma': parse_scale_bits(f"SCALE_BLOCK{L}_LN2_WEIGHT"),
            'ln2_beta':  parse_scale_bits(f"SCALE_BLOCK{L}_LN2_BIAS"),
            'qkv':       parse_scale_bits(f"SCALE_BLOCK{L}_ATTN_QKV_WEIGHT"),
            'proj':      parse_scale_bits(f"SCALE_BLOCK{L}_ATTN_PROJ_WEIGHT"),
            'ff_up':     parse_scale_bits(f"SCALE_BLOCK{L}_FF_UP_WEIGHT"),
            'ff_down':   parse_scale_bits(f"SCALE_BLOCK{L}_FF_DOWN_WEIGHT"),
        }
        layer_weights[L] = {
            "qkv": qkv_w, "proj": proj_w,
            "ff_up": ff_up_w, "ff_down": ff_down_w,
            "scales": scales,
        }

    return {
        "tok_emb_w": tok_emb_w, "pos_emb_w": pos_emb_w,
        "ln_mem": ln_mem, "isqrt_lut": isqrt_lut,
        "lut0": lut0, "lut1": lut1,
        "breaks": breaks, "slopes": slopes, "icepts": icepts,
        "layer_weights": layer_weights,
        "tok_scale": tok_scale, "pos_scale": pos_scale,
        "lnf_g_scale": lnf_g_scale, "lnf_b_scale": lnf_b_scale,
    }


def run_forward(token_id, position, kv_cache, res):
    return rtl_forward_fp16(
        token_id, position, kv_cache,
        res["tok_emb_w"], res["pos_emb_w"],
        res["ln_mem"], res["isqrt_lut"],
        res["layer_weights"],
        res["lut0"], res["lut1"],
        res["breaks"], res["slopes"], res["icepts"],
        res["tok_scale"], res["pos_scale"],
        res["lnf_g_scale"], res["lnf_b_scale"])


def parse_prompt(arg):
    # Try as integer token ID first
    try:
        tok = int(arg)
        if 0 <= tok < VOCAB:
            return [tok]
    except ValueError:
        pass
    # Treat as string, convert each char to its byte value
    tokens = []
    for ch in arg:
        b = ord(ch)
        if b >= VOCAB:
            print(f"Warning: character '{ch}' (U+{b:04X}) outside vocab, skipping")
            continue
        tokens.append(b)
    return tokens


def generate(prompt_tokens, n_gen, res):
    kv_cache = {}
    pos = 0

    # Prompt phase: fill KV cache
    if len(prompt_tokens) > 1:
        print(f"Prompt: ", end="", flush=True)
        for tok in prompt_tokens[:-1]:
            print(token_repr(tok), end="", flush=True)
            t0 = time.time()
            _ = run_forward(tok, pos, kv_cache, res)
            elapsed = time.time() - t0
            pos += 1
        print(f"  ({pos} tokens prefilled)")

    # Generation phase
    cur_token = prompt_tokens[-1]
    print(f"Generating from token {cur_token} ({token_repr(cur_token)}):")
    print()

    output_tokens = []
    output_text = token_repr(cur_token)

    for step in range(n_gen):
        t0 = time.time()
        logits = run_forward(cur_token, pos, kv_cache, res)
        elapsed = time.time() - t0

        next_token = fp16_argmax(logits)
        output_tokens.append(next_token)

        top_val = fp16_to_float(logits[next_token])
        tr = token_repr(next_token)
        output_text += tr

        print(f"  [{step:3d}] pos={pos:3d}  token={next_token:3d}  "
              f"{tr:>6s}  logit={top_val:+8.3f}  {elapsed:.2f}s")

        cur_token = next_token
        pos += 1

    print()
    print(f"Output: {output_text}")
    return output_tokens


if __name__ == "__main__":
    n_gen = 20
    prompt_arg = "65"

    if len(sys.argv) >= 2:
        prompt_arg = sys.argv[1]
    if len(sys.argv) >= 3:
        n_gen = int(sys.argv[2])

    prompt_tokens = parse_prompt(prompt_arg)
    if not prompt_tokens:
        print("Error: empty prompt")
        sys.exit(1)

    print("Loading resources...")
    res = load_all_resources()
    print(f"Loaded {N_LAYERS} layers, vocab={VOCAB}, dim={DIM}")
    print()

    generate(prompt_tokens, n_gen, res)