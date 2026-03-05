"""
Validate tb_transformer_top xsim output against Python rtl model (W8A16)

Two reference models:
  rtl: exact W8A16 hardware fp16 arithmetic (rtl_forward_fp16)
          xsim must match rtl exactly (bit-for-bit fp16)
  ideal: float64 reference for error analysis

Works with both tb_transformer_top.log and tb_transformer_top_stress.log.
Events are grouped by test number; prompt events fill KV cache,
generate events produce logits and tokens to validate.
"""

import re
import os
import sys

from rtl_ops import (
    DIM, N_LAYERS, VOCAB, MEM,
    to_signed8, fp16_to_float, fp16_from_int,
    load_hex, load_lut16, load_gelu_pwl, parse_scale_bits,
    rtl_forward_fp16,
)
from ideal_ops import ideal_forward_fp16

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOG = os.path.join(PROJ, "logs", "tb_transformer_top.log")


# Human-readable token: printable ASCII or hex
def token_repr(tok):
    if 0x20 <= tok < 0x7F:
        return f"'{chr(tok)}'"
    if tok == 0x0A:
        return "'\\n'"
    if tok == 0x0D:
        return "'\\r'"
    if tok == 0x09:
        return "'\\t'"
    return f"0x{tok:02x}"


def parse_log(path):
    test_pat = re.compile(r"TEST (\d+) TOKEN=(\d+) POS=(\d+) (GENERATE|PROMPT)")
    logit_pat = re.compile(r"LOGITS\[(\d+)\]=([0-9a-fA-F]{4})")
    out_pat = re.compile(r"OUT_TOKEN=([0-9a-fA-F]{2})")
    gen_pat = re.compile(r"GEN_TOKEN\[(\d+)\]=([0-9a-fA-F]{2})")

    events = []
    current = None

    with open(path) as f:
        for line in f:
            m = test_pat.search(line)
            if m:
                if current is not None:
                    events.append(current)
                current = {
                    "test": int(m.group(1)),
                    "token": int(m.group(2)),
                    "pos": int(m.group(3)),
                    "mode": m.group(4),
                    "logits": {},
                    "out_token": None,
                    "gen_tokens": {},
                }
                continue
            if current is None:
                continue
            m = logit_pat.search(line)
            if m:
                current["logits"][int(m.group(1))] = int(m.group(2), 16)
                continue
            m = out_pat.search(line)
            if m:
                current["out_token"] = int(m.group(1), 16)
                continue
            m = gen_pat.search(line)
            if m:
                current["gen_tokens"][int(m.group(1))] = int(m.group(2), 16)
                continue

    if current is not None:
        events.append(current)
    return events


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


def run_rtl_forward(token_id, position, kv_cache, res):
    return rtl_forward_fp16(
        token_id, position, kv_cache,
        res["tok_emb_w"], res["pos_emb_w"],
        res["ln_mem"], res["isqrt_lut"],
        res["layer_weights"],
        res["lut0"], res["lut1"],
        res["breaks"], res["slopes"], res["icepts"],
        res["tok_scale"], res["pos_scale"],
        res["lnf_g_scale"], res["lnf_b_scale"])


def run_ideal_forward(token_id, position, kv_cache_f, res):
    return ideal_forward_fp16(
        token_id, position, kv_cache_f,
        res["tok_emb_w"], res["pos_emb_w"],
        res["ln_mem"],
        res["layer_weights"],
        res["tok_scale"], res["pos_scale"],
        res["lnf_g_scale"], res["lnf_b_scale"])


# Compare 256 fp16 logits with rtl and ideal references, return error count
def compare_logits(xsim_logits, rtl_logits, ideal_logits):
    errors = 0
    max_xi_delta = 0.0
    sum_xi_delta = 0.0
    n_delta = 0

    print(f"  {'idx':>5s}  {'xsim':>6s}  {'rtl':>6s}  {'ideal':>6s}"
          f"  {'x-i delta':>10s}  {'status'}")

    for i in range(VOCAB):
        xv = xsim_logits.get(i, 0)
        rv = rtl_logits[i]
        iv = ideal_logits[i]
        match = (xv == rv)
        if not match:
            errors += 1
            status = "MISMATCH"
        else:
            status = "OK"

        xv_special = (xv & 0x7FFF) >= 0x7C00
        iv_special = (iv & 0x7FFF) >= 0x7C00
        if not xv_special and not iv_special:
            xf = fp16_to_float(xv)
            ivf = fp16_to_float(iv)
            xi_delta = xf - ivf
            max_xi_delta = max(max_xi_delta, abs(xi_delta))
            sum_xi_delta += abs(xi_delta)
            n_delta += 1
            print(f"  {i:5d}  {xv:04x}    {rv:04x}    {iv:04x}"
                  f"    {xi_delta:+10.6f}  {status}")
        else:
            print(f"  {i:5d}  {xv:04x}    {rv:04x}    {iv:04x}"
                  f"    {'---':>10s}  {status}")

    print()
    print(f"  RTL match:       {VOCAB - errors}/{VOCAB}")
    if n_delta > 0:
        print(f"  Xsim vs ideal:      max={max_xi_delta:.6f}"
              f"  mean={sum_xi_delta / n_delta:.6f}")

    return errors


def print_token_result(label, xsim_tok, rtl_tok, ideal_tok=None):
    match = (xsim_tok == rtl_tok)
    parts = [f"  {label}: xsim={xsim_tok:3d} {token_repr(xsim_tok):>6s}"
             f"  rtl={rtl_tok:3d} {token_repr(rtl_tok):>6s}"]
    if ideal_tok is not None:
        parts.append(f"  ideal={ideal_tok:3d} {token_repr(ideal_tok):>6s}")
    parts.append(f"  {'OK' if match else 'MISMATCH'}")
    print("".join(parts))
    return 0 if match else 1


# Process all events for one test number, return (errors, count)
def process_test_group(test_id, test_events, res):
    total_errors = 0
    total_count = 0

    prompts = [e for e in test_events if e["mode"] == "PROMPT"]
    generates = [e for e in test_events if e["mode"] == "GENERATE"]

    if not generates:
        print(f"  (prompt-only test, no output to validate)")
        return 0, 0

    kv_cache = {}
    kv_ideal = {}
    pos = 0

    # Run prompt events (fill KV cache, no output)
    for ev in prompts:
        print(f"  Prompt: token={ev['token']} pos={pos}")
        _ = run_rtl_forward(ev["token"], pos, kv_cache, res)
        _ = run_ideal_forward(ev["token"], pos, kv_ideal, res)
        pos += 1

    for ev in generates:
        start_token = ev["token"]
        gen_tokens = ev["gen_tokens"]
        n_gen = max(gen_tokens.keys()) + 1 if gen_tokens else 0

        # Compare logits if logged
        if ev["logits"]:
            rtl_logits = run_rtl_forward(start_token, pos, kv_cache, res)
            ideal_logits = run_ideal_forward(start_token, pos, kv_ideal, res)
            rtl_tok = fp16_argmax(rtl_logits)
            ideal_tok = fp16_argmax(ideal_logits)

            errs = compare_logits(ev["logits"], rtl_logits, ideal_logits)
            total_errors += errs
            total_count += VOCAB

            if ev["out_token"] is not None:
                total_errors += print_token_result(
                    "Output token", ev["out_token"], rtl_tok, ideal_tok)
                total_count += 1

            # Logit step already advanced pos
            if n_gen > 0:
                xsim_tok_0 = gen_tokens.get(0)
                if xsim_tok_0 is not None:
                    total_errors += print_token_result(
                        "GEN_TOKEN[0]", xsim_tok_0, rtl_tok, ideal_tok)
                    total_count += 1
                    cur_token = xsim_tok_0
                else:
                    cur_token = rtl_tok
                pos += 1

                for step in range(1, n_gen):
                    rtl_logits = run_rtl_forward(cur_token, pos, kv_cache, res)
                    ideal_logits = run_ideal_forward(cur_token, pos, kv_ideal, res)
                    rtl_tok = fp16_argmax(rtl_logits)
                    ideal_tok = fp16_argmax(ideal_logits)

                    xsim_tok = gen_tokens.get(step)
                    if xsim_tok is not None:
                        total_errors += print_token_result(
                            f"GEN_TOKEN[{step}]", xsim_tok, rtl_tok, ideal_tok)
                        total_count += 1
                    else:
                        print(f"  GEN_TOKEN[{step}]: xsim=MISSING rtl={rtl_tok:02x}")
                        total_errors += 1
                        total_count += 1

                    cur_token = xsim_tok if xsim_tok is not None else rtl_tok
                    pos += 1
            else:
                pos += 1

        # No logits logged, just autoregressive tokens
        elif n_gen > 0:
            cur_token = start_token
            for step in range(n_gen):
                rtl_logits = run_rtl_forward(cur_token, pos, kv_cache, res)
                ideal_logits = run_ideal_forward(cur_token, pos, kv_ideal, res)
                rtl_tok = fp16_argmax(rtl_logits)
                ideal_tok = fp16_argmax(ideal_logits)

                xsim_tok = gen_tokens.get(step)
                if xsim_tok is not None:
                    total_errors += print_token_result(
                        f"GEN_TOKEN[{step}]", xsim_tok, rtl_tok, ideal_tok)
                    total_count += 1
                else:
                    print(f"  GEN_TOKEN[{step}]: xsim=MISSING rtl={rtl_tok:02x}")
                    total_errors += 1
                    total_count += 1

                cur_token = xsim_tok if xsim_tok is not None else rtl_tok
                pos += 1

    return total_errors, total_count


if __name__ == "__main__":
    LOG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG

    if not os.path.exists(LOG):
        print(f"Log not found: {LOG}")
        print("Run tb_transformer_top simulation first")
        sys.exit(1)

    print("Loading resources...")
    res = load_all_resources()
    print(f"Loaded tok_emb: {len(res['tok_emb_w'])} bytes")
    print(f"Loaded pos_emb: {len(res['pos_emb_w'])} bytes")
    for L in range(N_LAYERS):
        lw = res["layer_weights"][L]
        print(f"Layer {L}: QKV={len(lw['qkv'])}, proj={len(lw['proj'])}, "
              f"ff_up={len(lw['ff_up'])}, ff_down={len(lw['ff_down'])}")

    print(f"\nReading: {LOG}\n")
    events = parse_log(LOG)

    if not events:
        print("No test data found in log")
        sys.exit(1)

    # Group events by test number
    test_groups = {}
    for ev in events:
        tid = ev["test"]
        if tid not in test_groups:
            test_groups[tid] = []
        test_groups[tid].append(ev)

    total_errors = 0
    total_count = 0

    for tid in sorted(test_groups.keys()):
        group = test_groups[tid]
        modes = [e["mode"] for e in group]
        tokens = [e["token"] for e in group]
        n_gen = 0
        for e in group:
            if e["gen_tokens"]:
                n_gen = max(n_gen, max(e["gen_tokens"].keys()) + 1)

        if n_gen > 0:
            print(f"Test {tid}: token={tokens[0]}"
                  f" {token_repr(tokens[0])}, {n_gen} generated tokens")
        elif "PROMPT" in modes and "GENERATE" in modes:
            print(f"Test {tid}: prompt+generate, tokens={tokens}")
        else:
            print(f"Test {tid}: tokens={tokens} modes={modes}")

        errs, cnt = process_test_group(tid, group, res)
        total_errors += errs
        total_count += cnt
        print()

    if total_errors == 0:
        print(f"PASSED - all {total_count} outputs match rtl model")
    else:
        print(f"FAILED - {total_errors} mismatches vs rtl model")
    sys.exit(0 if total_errors == 0 else 1)