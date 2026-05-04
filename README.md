# red_eyes_is_all_you_need

A tiny transformer LLM running entirely on an Artix-7 FPGA (XC7A200T). Character-level text generation trained on Shakespeare, with the full inference pipeline implemented in synthesizable Verilog.

## Performance

Runs on FPGA (XC7A200T @ 95 MHz) with ~1100 tok/s

Per-token cycle budget:

| Region | cycles/token | % |
|---|---:|---:|
| Attention | 32,580 | 46% |
| FF_up | 11,020 | 15% |
| FF_down | 10,828 | 15% |
| LayerNorm x2 | 5,832 | 8% |
| HEAD_PROJ | 2,459 | 3% |
| sampler | 1,634 | 2% |
| ln_f | 729 | 1% |
| embed | 271 | <1% |

## Architecture

| Parameter     | Value |
|---------------|-------|
| Vocab size    | 256, byte-level |
| Embedding dim | 128 |
| Attention heads | 8, head_dim 16 |
| Layers        | 4 |
| Context length | 256 |
| FF hidden dim | 512 |
| Weight format | W8A16, int8 weights, fp16 activations |
| Total params  | ~854K, 854,272 bytes quantized |

### Inference pipeline

```
Token in -> Embedding -> 4x Transformer Layer -> LayerNorm ln_f -> Head Projection -> Sampler -> Token out
                              |                                          |
                         KV Cache, BRAM                          Weight-tied with tok_emb
```

Each transformer layer:
```
x -> LN1 -> Attention -> +residual -> LN2 -> FF_up -> GELU -> FF_down -> +residual -> out
```

### Quantization: W8A16

Weights are stored as int8 in BRAM and dequantized to fp16 at runtime:

```
fp16_activation = fp16_from_int8(weight_byte) * fp16_scale
```

All intermediate activations, KV cache entries, and accumulations use IEEE 754 half-precision fp16 with flush-to-zero denormals and round-to-nearest-even.

### Top level

| Module | Description |
|--------|-------------|
| `top.v` | System top: MMCM 200 MHz LVDS to 95 MHz, UART rx/tx, transformer_top, LEDs |
| `transformer_top.v` | Full inference FSM: embed -> 4 layers -> ln_f -> head proj -> sampler |
| `transformer_layer.v` | LN -> attention -> residual -> LN -> FF -> residual |
| `attention.v` | 8-head self-attention, 4-way packed KV, per-position 4-way score+AV |
| `sampler.v` | FP16 mul x 1/temp -> Q16.7 -> softmax -> top-k -> Fibonacci LFSR multinomial |

### Functional units

| Module | Description |
|--------|-------------|
| `matvec_fp16_w32.v` | 32-way SIMD matrix-vector multiply for QKV, proj, FF_up, FF_down |
| `matvec_fp16_w16.v` | 16-way SIMD matrix-vector multiply, used by head_proj |
| `layernorm.v` | FP16 LayerNorm with int8 gamma/beta dequant, ~646 cycles |
| `gelu.v` | ISPA piecewise-linear erf, 16 segments, 2-cycle pipeline |
| `softmax.v` | SafeSoftmax, bipartite exp LUT, Q1.15 output, ~2N+5 cycles |
| `embedding.v` | tok_emb + pos_emb dequant to fp16 |

### FP16 primitives

| Module | Description | Latency |
|--------|-------------|---------|
| `fp16_add.v` | IEEE 754 half-precision adder, RNE rounding | 4 cycles |
| `fp16_mul.v` | IEEE 754 half-precision multiplier, 1 DSP48 | 2 cycles |
| `fp16_reduce_k8.v` | K=8 partial-sum reducer with balanced tree | streaming |
| `fp16_rsqrt.v` | Reciprocal square root via LOD-LUT-shift | 2 cycles |
| `fp16_from_int8.v` | Signed int8 to fp16, exact | combinational |
| `fp16_to_int8.v` | FP16 to signed int8 with RNE + clamp | combinational |
| `fp16_to_q167.v` | FP16 to signed Q16.7 for softmax input | combinational |
| `q115_to_fp16.v` | Unsigned Q1.15 to fp16 for softmax output | combinational |

### Storage

| Module | Description |
|--------|-------------|
| `weight_store.v` | 18 LN tensors packed in 1 BRAM + pos_emb byte ROM |
| `weight_store_qkv/proj/ff_up/ff_down.v` | 256-bit packed banks for matvec_fp16_w32 |
| `weight_store_tok_emb.v` | 128-bit packed bank for matvec_fp16_w16 head_proj |
| `weight_rom.v` | Synchronous-read BRAM ROM primitive |
| `kv_cache.v` | 4 layers x 8 heads, fp16, 64-bit 4-way packed words |
| `kv_ram.v` | Read-write BRAM primitive with byte-write enable |

## Resource Usage

Post-route at 95 MHz target, fmax ~98.8 MHz, all 132,895 endpoints meet timing.

| Resource | Used |
|----------|---:|
| Block RAM Tile | 338.5 |
| Slice LUTs | 84,595 |
| Slice Registers | 63,534 |
| DSP48E1 | 316 |

## Training

Character-level GPT trained on Shakespeare, ~1.1 MB.

`scripts/train/train.py`: Standard transformer, 30K steps, AdamW. Post-training int8 quantization, per-tensor symmetric, scale = max/127. Weights exported to `weights_int8.bin` with custom binary format containing per-tensor int8 data + float32 scales.

### Weight extraction

```bash
python3 scripts/extract_weights.py
```

Reads `weights_int8.bin`, writes 36 `.hex` files to `mem/`:
- 4 banks x 4 layers in 256-bit packed format pairing col and col+IN/2 for matvec_fp16_w32
- tok_emb in 128-bit packed format for matvec_fp16_w16
- pos_emb as flat byte ROM
- ln_params combined into one 18-tensor BRAM

Also generates `rtl/weight_scales.vh` with fp16 scale localparams and regenerates `tb/tb_weight_store.v` with expected first/last byte assertions.

## Simulation

Requires Vivado 2025.2 in a Docker container (for MacOS).

### Run all tests

```bash
./scripts/run_tests.sh
```

### Run specific tests

```bash
./scripts/run_tests.sh transformer_top
./scripts/run_tests.sh attention transformer_layer
```

`tb_profile` is excluded from the default sweep, run it explicitly when needed.

The test flow:
1. xsim, Vivado simulator: compiles RTL + testbench, runs simulation, writes log
2. Validation, Python: parses xsim log, compares against RTL-exact Python model in `rtl_ops.py`

## Inference

Three ways to run the model.

### PyTorch on the host

The original training model. `load_weights_int8` reads the int8 file and dequantizes each tensor on load, so the model itself runs in plain float32 on CPU, MPS, or CUDA. This is the reference output before any of the RTL approximations come into play.

```bash
source scripts/train/venv/bin/activate
python3 scripts/train/inference.py --weights scripts/train/weights_int8.bin --prompt "Q" --tokens 255 --temperature 0.4 --top-k 10 --repeat-penalty 1.3 --seed 44257 --device mps
```

### RTL-exact, pure Python

Same fp16 rounding, LOD-LUT rsqrt, bipartite softmax, PWL GELU and LFSR sampler as the hardware, written out in Python on top of `rtl_ops.py`. No Vivado required, and the output matches the FPGA byte for byte at a given seed.

```bash
source scripts/train/venv/bin/activate
python3 scripts/tests/rtl_inference.py "Q" 255
```

### UART to the FPGA

Talks to the synthesized design on the AX7203 over the CP2102 USB UART. An optional `0xFE` config packet sets temperature, top-k, repeat penalty and LFSR seed. The prompt is streamed byte by byte to fill the KV cache, then `0xFF` kicks off autoregressive generation and each token comes back as it is sampled.

```bash
source scripts/train/venv/bin/activate
python3 scripts/uart_inference.py "Q" 255 --port /dev/cu.usbserial-110 --temp 0.4 --top-k 10 --repeat-penalty 1.3 --seed 44257
```

## Validation

Every RTL module has a corresponding testbench and Python validation script:

| RTL Module | Testbench | Validator |
|------------|-----------|-----------|
| `weight_store.v` | `tb_weight_store.v` | `validate_weights.py` |
| `fp16_*.v` | `tb_fp16.v` | `validate_fp16.py` |
| `embedding.v` | `tb_embedding.v` | `validate_embedding.py` |
| `layernorm.v` | `tb_layernorm.v` | `validate_layernorm.py` |
| `softmax.v` | `tb_softmax.v` | `validate_softmax.py` |
| `gelu.v` | `tb_gelu.v` | `validate_gelu.py` |
| `kv_cache.v` | `tb_kv_cache.v` | `validate_kv_cache.py` |
| `attention.v` | `tb_attention.v` | `validate_attention.py` |
| `transformer_layer.v` | `tb_transformer_layer.v` | `validate_transformer_layer.py` |
| `transformer_top.v` | `tb_transformer_top.v` | `validate_transformer_top.py` |
| `sampler.v` | `tb_sampler.v` | `validate_sampler.py` |

Stress tests `tb_*_stress.v` run extended sequences, 300 positions for attention, 50 for transformer_layer, 20 autoregressive tokens for transformer_top, and reuse the same validators. All are bit-exact against the Python model.

### RTL-exact model

- `rtl_ops.py`: Pure-Python fp16 primitives that replicate RTL rounding bit-for-bit. All validation compares xsim output against this model.
- `ideal_ops.py`: Float64 reference models for error analysis, not used for pass/fail.

`tb_profile.v` instruments the FSM states and dumps cycles per state per token, used for performance analysis but not validated.

## Synthesis

```bash
./scripts/synth/run_synth.sh
```

Vivado 2025.2 in Docker. Outputs `build/post_synth.dcp`, `build/post_route.dcp`, `build/top.bit`. Floorplanning constraints in `constraints/ax7203.xdc` keep each matvec near its dedicated weight bank to control route delay.

## Project Structure

```
rtl/                    30 Verilog modules
  weight_scales.vh      Auto-generated fp16 scale localparams
  gelu_pwl_coeffs.vh    Auto-generated GELU breakpoint and slope/intercept
tb/                     15 testbenches
mem/                    Weight hex files, LUTs, test vectors
constraints/            ax7203.xdc + ax7203_impl.xdc
scripts/
  extract_weights.py    Binary weights -> hex + scales + tb_weight_store
  run_tests.sh          Simulation + validation runner
  gen_*.py              LUT and test vector generators
  synth/                Vivado synth tcl
  tests/                Validation scripts + RTL-exact model
  train/                Training code, checkpoint, data, host inference
docs/                   Style guide
```

## Paper-based implementations

Three modules are based on published hardware designs:

| Module | Paper | What it provides | Generator script |
|--------|-------|------------------|------------------|
| `gelu.v` | [Huang et al., Electronics 2025, 14(9), 1825](https://www.mdpi.com/2079-9292/14/9/1825) | ISPA piecewise-linear erf approximation, 16 non-uniform segments, EPSS breakpoint optimization | `scripts/gen_gelu_pwl_coeffs.py` |
| `softmax.v` | [Kang & Wang, Micromachines 2026, 17(1), 84](https://www.mdpi.com/2072-666X/17/1/84) | Division-free SafeSoftmax via bipartite exp(-d) LUT with compensated initialization | `scripts/gen_softmax_luts.py` |
| `fp16_rsqrt.v` | [Kang & Wang, Micromachines 2026, 17(1), 84](https://www.mdpi.com/2072-666X/17/1/84) | LOD-LUT-shift reciprocal square root, reused inside `layernorm.v` | `scripts/gen_inv_sqrt_lut.py` |

## Board

Target: ALINX AX7203, XC7A200TFBG484-2, 200 MHz LVDS clock, USB UART CP2102, active-low LEDs/buttons. Pin assignments in `constraints/ax7203.xdc`.

UART protocol over 115200 8N1:
- PC sends 1 byte -> FPGA runs prompt pass per byte, fills KV cache
- PC sends 0xFE + N config bytes -> sets inv_temp, top_k, inv_penalty, RNG seed
- PC sends 0xFF -> FPGA runs autoregressive generation, sends each token back
