"""
UART inference client for FPGA transformer

Sends prompt bytes over UART, then 0xFF to trigger generation
Receives and prints generated tokens

Usage:
  python3 uart_inference.py                    # default: "A", 50 tokens
  python3 uart_inference.py "Hello" 100        # prompt string, 100 tokens
  python3 uart_inference.py --port /dev/cu.usbserial-110 "Hi"
"""

import sys
import os
import argparse
import time
import serial


CMD_GENERATE = 0xFF


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
    args = parser.parse_args()

    prompt_bytes = [ord(c) for c in args.prompt if ord(c) < 256]
    if not prompt_bytes:
        print("Error: empty prompt")
        sys.exit(1)

    # Check for 0xFF in prompt (reserved for generate command)
    if CMD_GENERATE in prompt_bytes:
        print("Error: prompt cannot contain 0xFF (reserved)")
        sys.exit(1)

    print(f"Port: {args.port}")
    print(f"Prompt: {repr(args.prompt)} ({len(prompt_bytes)} tokens)")
    print(f"Max generate: {args.max_tokens}")
    print()

    ser = serial.Serial(args.port, args.baud, timeout=30)
    ser.reset_input_buffer()

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

    # Receive generated tokens
    print("Output: ", end="", flush=True)
    t0 = time.time()
    received = 0
    for _ in range(args.max_tokens):
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