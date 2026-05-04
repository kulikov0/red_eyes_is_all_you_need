"""
UART inference client for FPGA transformer

Sends prompt bytes over UART, then 0xFF to trigger generation
Receives and prints generated tokens

Usage:
  python3 uart_inference.py "Q" 255 --port /dev/cu.usbserial-110 --temp 0.4 --top-k 10 --repeat-penalty 1.3 --seed 44257
"""

import sys
import os
import argparse
import random
import struct
import time
import serial


CMD_GENERATE = 0xFF
CMD_CONFIG   = 0xFE
SAMPLER_K_MAX = 16


# Convert positive Python float to IEEE 754 fp16 bit pattern
def fp16_bits(f):
    return struct.unpack("<H", struct.pack("<e", float(f)))[0]


# Human-readable token: printable ASCII or hex
def token_repr(tok):
    if 0x20 <= tok < 0x7F:
        return chr(tok)
    if tok == 0x0A:
        return "\n"
    if tok == 0x0D:
        return "\r"
    if tok == 0x09:
        return "\t"
    return f"<0x{tok:02x}>"


def main():
    parser = argparse.ArgumentParser(description="UART inference client")
    parser.add_argument("prompt", nargs="?", default="A",
                        help="Prompt string (default: 'A')")
    parser.add_argument("max_tokens", nargs="?", type=int, default=50,
                        help="Max tokens to generate (default: 50)")
    parser.add_argument("--port", default="/dev/cu.usbserial-110",
                        help="Serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--temp", type=float, default=None,
                        help="Sampling temperature; default = greedy")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"Top-k cutoff; 1 = greedy, 0 = full vocab, max = {SAMPLER_K_MAX}")
    parser.add_argument("--seed", type=int, default=None,
                        help="LFSR seed integer 1..65535; random if omitted")
    parser.add_argument("--repeat-penalty", type=float, default=None,
                        help="Repetition penalty (>1 demotes seen tokens); 1.0 = off")
    args = parser.parse_args()

    prompt_bytes = [ord(c) for c in args.prompt if ord(c) < 256]
    if not prompt_bytes:
        print("Error: empty prompt")
        sys.exit(1)

    # 0xFE and 0xFF are reserved for CMD_CONFIG and CMD_GENERATE
    for reserved in (CMD_GENERATE, CMD_CONFIG):
        if reserved in prompt_bytes:
            print(f"Error: prompt cannot contain 0x{reserved:02X}")
            sys.exit(1)

    send_cfg = (args.temp is not None or args.top_k is not None
                or args.seed is not None or args.repeat_penalty is not None)
    if send_cfg:
        temp = args.temp if args.temp is not None else 1.0
        if temp <= 0.0:
            print("Error: --temp must be positive")
            sys.exit(1)
        inv_temp_bits = fp16_bits(1.0 / temp)
        top_k = args.top_k if args.top_k is not None else 1
        if top_k != 0 and not (1 <= top_k <= SAMPLER_K_MAX):
            print(f"Error: --top-k must be 0 or in 1..{SAMPLER_K_MAX}")
            sys.exit(1)
        seed = args.seed if args.seed is not None else random.randint(1, 0xFFFF)
        if not (1 <= seed <= 0xFFFF):
            print("Error: --seed must be in 1..65535")
            sys.exit(1)
        penalty = args.repeat_penalty if args.repeat_penalty is not None else 1.0
        if penalty <= 0.0:
            print("Error: --repeat-penalty must be positive")
            sys.exit(1)
        inv_penalty_bits = fp16_bits(1.0 / penalty)
        cfg_bytes = bytes([
            inv_temp_bits & 0xFF, (inv_temp_bits >> 8) & 0xFF,
            top_k,
            seed & 0xFF, (seed >> 8) & 0xFF,
            inv_penalty_bits & 0xFF, (inv_penalty_bits >> 8) & 0xFF,
        ])

    print(f"Port: {args.port}")
    print(f"Prompt: {repr(args.prompt)} ({len(prompt_bytes)} tokens)")
    print(f"Max generate: {args.max_tokens}")
    if send_cfg:
        print(f"Config: temp={temp} top_k={top_k} seed={seed} penalty={penalty}")
    print()

    ser = serial.Serial(args.port, args.baud, timeout=30)
    ser.reset_input_buffer()

    if send_cfg:
        ser.write(bytes([CMD_CONFIG]))
        time.sleep(0.05)
        for b in cfg_bytes:
            ser.write(bytes([b]))
            time.sleep(0.05)
        print("Config sent")

    print("Sending prompt...", end="", flush=True)
    for i, tok in enumerate(prompt_bytes):
        ser.write(bytes([tok]))
        time.sleep(0.15)
        print(f" {token_repr(tok)}", end="", flush=True)
    print()

    # Send generate command
    print("Starting generation...")
    ser.write(bytes([CMD_GENERATE]))
    print()

    # FPGA stops at pos_r=255, so emits at most (256 - prompt_len) tokens
    expected = min(args.max_tokens, 256 - len(prompt_bytes))

    print("Output:\n")
    sys.stdout.write("".join(token_repr(b) for b in prompt_bytes))
    sys.stdout.flush()
    t0 = time.time()
    received = 0
    for _ in range(expected):
        data = ser.read(1)
        if not data:
            print("\n[timeout - no response]")
            break
        tok = data[0]
        received += 1
        sys.stdout.write(token_repr(tok))
        sys.stdout.flush()

    elapsed = time.time() - t0
    print()
    print()
    if received > 0:
        print(f"Generated {received} tokens in {elapsed:.1f}s "
              f"({received/elapsed:.1f} tok/s)")

    ser.close()


if __name__ == "__main__":
    main()