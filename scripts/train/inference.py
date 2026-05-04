"""
Load int8 quantized weights, dequantize, and run inference

Usage:
  python3 inference.py --weights weights_int8.bin --prompt "Q" --tokens 255 --temperature 0.4 --top-k 10 --repeat-penalty 1.3 --seed 44257 --device mps
"""

import argparse
import struct
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import Config, TinyTransformer, generate_with_penalty, load_weights_int8

cfg = Config()


def main():
    parser = argparse.ArgumentParser(description="Run int8 transformer inference")
    parser.add_argument("--weights", default="weights_int8.bin", help="Path to weights_int8.bin")
    parser.add_argument("--prompt", default="VINCENTIO:\n", help="Text prompt to seed generation")
    parser.add_argument("--tokens", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling")
    parser.add_argument("--repeat-penalty", type=float, default=1.3, help="Repetition penalty")
    parser.add_argument("--seed", type=int, default=None, help="Torch RNG seed for reproducible sampling")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                        help="Override device. Default: cfg.device (auto-detected)")
    args = parser.parse_args()

    if args.device is not None:
        cfg.device = args.device

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {cfg.device}")
    print(f"Loading weights: {args.weights}")

    state_dict = load_weights_int8(args.weights)
    model = TinyTransformer(cfg).to(cfg.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Parameters: {model.count_params():,}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Generating {args.tokens} tokens (temp={args.temperature}, top_k={args.top_k})\n")

    prompt_t = torch.tensor(
        list(args.prompt.encode("utf-8")), dtype=torch.long, device=cfg.device
    ).unsqueeze(0)

    t0 = time.time()
    out = generate_with_penalty(
        model, prompt_t,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
    )
    elapsed = time.time() - t0

    generated = bytes(out[0][len(prompt_t[0]):].tolist()).decode("utf-8", errors="replace")
    print(args.prompt + generated)
    print(f"\nGenerated {args.tokens} tokens in {elapsed:.3f}s ({args.tokens / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()