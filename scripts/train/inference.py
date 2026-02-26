"""
Load int8 quantized weights, dequantize, and run inference.

Usage:
    python3 scripts/inference.py
    python3 scripts/inference.py --prompt "VINCENTIO:" --tokens 500
    python3 scripts/inference.py --weights path/to/weights_int8.bin
"""

import argparse
import struct
import math
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
    args = parser.parse_args()

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

    out = generate_with_penalty(
        model, prompt_t,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
    )

    generated = bytes(out[0][len(prompt_t[0]):].tolist()).decode("utf-8", errors="replace")
    print(args.prompt + generated)


if __name__ == "__main__":
    main()
