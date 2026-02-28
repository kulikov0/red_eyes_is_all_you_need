"""
Generate combined GELU LUT via calibration

Loads the trained checkpoint, runs calibration data through the model,
hooks into each ff_up output (before GELU) to measure activation scales,
then generates int8 -> int8 GELU LUTs packed into a single BRAM

Output: mem/gelu_lut.hex (1024 entries: 4 layers x 256, 1 byte per line)

BRAM layout: addr[9:8] = layer, addr[7:0] = input byte (unsigned)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts", "train"))
from train import TinyTransformer, Config, ByteDataset

cfg = Config()
CKPT = os.path.join(PROJ, "scripts", "train", "ckpt.pt")
DATA = os.path.join(PROJ, "scripts", "train", "input.txt")
MEM_DIR = os.path.join(PROJ, "mem")
# 32 batches x 64 = 2048 samples (~0.2% of data, sufficient per stability test)
N_CALIB_BATCHES = 32


"""
Build a 256-entry int8 -> int8 GELU LUT for a given activation scale

The ff_up output int8 value v represents float v * act_scale
LUT maps: v -> clamp(round(gelu(v * act_scale) / act_scale), -128, 127)
"""
def build_gelu_lut(act_scale):
    inputs = torch.tensor([i - 256 if i >= 128 else i for i in range(256)],
                          dtype=torch.float32)
    x = inputs * act_scale
    y = F.gelu(x)
    q = torch.clamp(torch.round(y / act_scale), -128, 127).to(torch.int8)
    return [(v.item() & 0xFF) for v in q]


def write_hex(path, lut, act_scales):
    with open(path, "w") as f:
        for i, s in enumerate(act_scales):
            f.write(f"// layer {i} act_scale = {s:.10f}\n")
        for val in lut:
            f.write(f"{val:02x}\n")


def main():
    device = cfg.device
    print(f"Device: {device}")

    # Load model
    model = TinyTransformer(cfg).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (step {ckpt.get('step', '?')})")

    # Load calibration data
    raw = open(DATA, "rb").read()
    ds = ByteDataset(raw, cfg.context_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    print(f"Calibration data: {len(ds):,} samples, using {N_CALIB_BATCHES} batches")

    # Hook into ff_up outputs (net[0] is the Linear, before GELU)
    act_stats = {i: {"min": float("inf"), "max": float("-inf")}
                 for i in range(cfg.n_layers)}

    hooks = []
    for layer_idx in range(cfg.n_layers):
        block = model.blocks[layer_idx]

        def make_hook(idx):
            def hook_fn(module, input, output):
                with torch.no_grad():
                    act_stats[idx]["min"] = min(act_stats[idx]["min"],
                                                output.min().item())
                    act_stats[idx]["max"] = max(act_stats[idx]["max"],
                                                output.max().item())
            return hook_fn

        h = block.ff.net[0].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Run calibration
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= N_CALIB_BATCHES:
                break
            model(x.to(device))
    for h in hooks:
        h.remove()

    print(f"\nCalibration results (ff_up output ranges):")
    print(f"{'layer':>5s}  {'min':>8s}  {'max':>8s}  {'act_scale':>10s}")
    print("-" * 38)

    act_scales = []
    for i in range(cfg.n_layers):
        abs_max = max(abs(act_stats[i]["min"]), abs(act_stats[i]["max"]))
        # Symmetric quantization: scale = abs_max / 127
        act_scale = abs_max / 127.0
        act_scales.append(act_scale)
        print(f"{i:5d}  {act_stats[i]['min']:8.3f}  {act_stats[i]['max']:8.3f}  {act_scale:10.6f}")

    # Generate combined GELU LUT (4 layers x 256 = 1024 entries in 1 BRAM)
    os.makedirs(MEM_DIR, exist_ok=True)
    combined = []
    for i in range(cfg.n_layers):
        combined.extend(build_gelu_lut(act_scales[i]))

    hex_path = os.path.join(MEM_DIR, "gelu_lut.hex")
    write_hex(hex_path, combined, act_scales)
    print(f"\nWrote {hex_path} ({len(combined)} entries, 1 BRAM18)")

    print("\nDone")


if __name__ == "__main__":
    main()
