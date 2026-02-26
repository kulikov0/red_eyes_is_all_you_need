"""
Extract int8 weights from weights_int8.bin into individual .hex files
for Verilog $readmemh, plus weight_scales.vh and manifest.txt

We need weight_scales.vh because the int8 weights in the ROMs are quantized -
they're not the actual values the neural network uses. Each tensor was scaled down
to fit into the -128..127 range, and each tensor has its own scale factor.
For example, if tok_emb has scale 0.005333, then a stored byte of 0x24 (36 decimal)
actually means 36 * 0.005333 = 0.192 in the real model.
When the FPGA computes a matrix-vector multiply, it works in int8 for speed.
But at some point after accumulation it needs to convert back to real values (or at least scale-aware values)
for the next layer.
Without weight_scales.vh, the hardware would have no way to know what the int8 numbers actually represent.
The weights would just be meaningless bytes.

We need manifest.txt for debugging and bookkeeping. When something goes wrong - a tensor loads incorrectly,
a simulation fails, or BRAM utilization doesn't match expectations - we need a single place to look up which tensor
is which index, how big it is, what file it maps to, and what scale it uses.

Binary format:
  Global header: 8-byte magic "TFPGA001" + uint32 num_tensors
  Per tensor:    uint32 name_len, name bytes, uint32 ndim,
                 ndim x uint32 shape, float32 scale, then raw int8 data
"""

import struct
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(PROJ, "scripts", "train", "weights_int8.bin")
MEM = os.path.join(PROJ, "mem")
RTL = os.path.join(PROJ, "rtl")

"""Convert tensor name to a clean filename stem.
e.g. 'blocks.0.attn.qkv.weight' => 'block0_attn_qkv_weight'
     'blocks.0.ff.net.0.weight' => 'block0_ff_up_weight'
     'blocks.0.ff.net.2.weight' => 'block0_ff_down_weight'
     'tok_emb.weight'           => 'tok_emb_weight'
     'ln_f.weight'              => 'ln_f_weight'
"""
def sanitize_name(tensor_name):
    n = tensor_name
    n = n.replace("ff.net.0.", "ff_up_")
    n = n.replace("ff.net.2.", "ff_down_")
    n = n.replace("blocks.", "block")
    n = n.replace(".", "_")
    return n

def verilog_param_name(stem):
    # Convert stem to UPPER_CASE Verilog localparam name for scale
    return "SCALE_" + stem.upper()

def main():
    os.makedirs(MEM, exist_ok=True)
    os.makedirs(RTL, exist_ok=True)

    with open(BIN, "rb") as f:
        # Global header
        magic = f.read(8)
        if magic != b"TFPGA001":
            print(f"ERROR: bad magic {magic!r}", file=sys.stderr)
            sys.exit(1)
        num_tensors = struct.unpack("<I", f.read(4))[0]
        print(f"Magic OK, {num_tensors} tensors")

        tensors = []
        for i in range(num_tensors):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("ascii")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
            scale_bytes = f.read(4)
            scale = struct.unpack("<f", scale_bytes)[0]
            scale_hex = struct.unpack("<I", scale_bytes)[0]  # IEEE 754 bits
            size = 1
            for s in shape:
                size *= s
            data = f.read(size)
            if len(data) != size:
                print(f"ERROR: tensor {name} expected {size} bytes, got {len(data)}", file=sys.stderr)
                sys.exit(1)

            stem = sanitize_name(name)
            hex_file = stem + ".hex"
            tensors.append({
                "index": i,
                "name": name,
                "stem": stem,
                "shape": shape,
                "size": size,
                "scale": scale,
                "scale_hex": scale_hex,
                "hex_file": hex_file,
                "data": data,
            })

    for t in tensors:
        hex_path = os.path.join(MEM, t["hex_file"])
        with open(hex_path, "w") as hf:
            for byte in t["data"]:
                # int8 stored as unsigned byte; write as 2-digit hex
                hf.write(f"{byte:02x}\n")
        print(f"  wrote {hex_path} ({t['size']} bytes)")

    # Re-read hex and compare
    errors = 0
    for t in tensors:
        hex_path = os.path.join(MEM, t["hex_file"])
        with open(hex_path, "r") as hf:
            lines = hf.read().strip().split("\n")
        if len(lines) != t["size"]:
            print(f"VERIFY FAIL: {t['name']} line count {len(lines)} != {t['size']}")
            errors += 1
            continue
        for j, line in enumerate(lines):
            val = int(line, 16)
            if val != t["data"][j]:
                print(f"VERIFY FAIL: {t['name']}[{j}] hex={val:02x} bin={t['data'][j]:02x}")
                errors += 1
                break
    if errors:
        print(f"\n*** {errors} verification errors! ***", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tensors)} hex files verified")

    manifest_path = os.path.join(MEM, "manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write(f"{'#':>3s}  {'Name':40s}  {'Shape':20s}  {'Size':>8s}  {'Scale':>12s}  {'Hex File'}\n")
        mf.write("-" * 110 + "\n")
        for t in tensors:
            shape_str = "x".join(str(s) for s in t["shape"])
            mf.write(f"{t['index']:3d}  {t['name']:40s}  {shape_str:20s}  {t['size']:8d}  {t['scale']:12.6f}  {t['hex_file']}\n")
    print(f"Wrote {manifest_path}")

    vh_path = os.path.join(RTL, "weight_scales.vh")
    with open(vh_path, "w") as vf:
        vf.write("// Auto-generated by extract_weights.py - DO NOT EDIT\n")
        vf.write("// IEEE 754 scale factors for int8 tensors\n\n")
        for t in tensors:
            param = verilog_param_name(t["stem"])
            vf.write(f"localparam [31:0] {param:50s} = 32'h{t['scale_hex']:08x};  // {t['scale']:.6f}\n")
    print(f"Wrote {vh_path}")

    # Combined LayerNorm hex: pack all 18 LN tensors (128 bytes each) into one file
    ln_indices = [i for i, t in enumerate(tensors) if t["size"] == 128]
    ln_hex_path = os.path.join(MEM, "ln_params.hex")
    with open(ln_hex_path, "w") as hf:
        for idx in ln_indices:
            for byte in tensors[idx]["data"]:
                hf.write(f"{byte:02x}\n")
    print(f"Wrote {ln_hex_path} ({len(ln_indices)} tensors, {len(ln_indices) * 128} bytes)")

    print(f"\nDone. {len(tensors)} tensors extracted")

if __name__ == "__main__":
    main()
