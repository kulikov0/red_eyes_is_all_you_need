# red_eyes_is_all_you_need

A tiny transformer LLM running entirely on an Artix-7 FPGA (XC7A200T). Character-level text generation trained on Shakespeare, with the full inference pipeline implemented in synthesizable Verilog.

## Architecture

| Parameter     | Value |
|---------------|-------|
| Vocab size    | 256 (byte-level) |
| Embedding dim | 128 |
| Attention heads | 8 (head_dim = 16) |
| Layers        | 4 |
| Context length | 256 |
| FF hidden dim | 512 |
| Weight format | W8A16 (int8 weights, fp16 activations) |
| Total params  | ~854K (854,272 bytes quantized) |

### Inference pipeline

```
Token in -> Embedding -> 4x Transformer Layer -> LayerNorm (ln_f) -> Head Projection -> Sampler -> Token out
                              |                                           |
                         KV Cache (BRAM)                          Weight-tied with tok_emb
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

All intermediate activations, KV cache entries, and accumulations use IEEE 754 half-precision (fp16) with flush-to-zero denormals and round-to-nearest-even.

### Top level

| Module | Description | Latency |
|--------|-------------|---------|
| `transformer_top.v` | Full inference FSM: embed -> 4 layers -> ln_f -> head proj -> sampler | ~1.1M cycles @ P=255 |
| `transformer_layer.v` | LN -> attention -> residual -> LN -> FF -> residual | ~277K cycles @ P=255 |
| `sampler.v` | FP16 argmax (stub for temperature/top-k/LFSR sampling) | 258 cycles |

### Functional units

| Module | Description | Latency |
|--------|-------------|---------|
| `attention.v` | Multi-head self-attention (W8A16), 8 heads, softmax bridge | ~144K cycles @ P=255 |
| `layernorm.v` | FP16 LayerNorm with int8 gamma/beta dequant | ~646 cycles |
| `gelu.v` | ISPA piecewise-linear erf, 16 segments | 2 cycles |
| `softmax.v` | SafeSoftmax, bipartite exp LUT, Q1.15 output | ~2N+5 cycles |
| `embedding.v` | tok_emb + pos_emb dequant to fp16 | 258 cycles |
| `matvec_fp16.v` | W8A16 matrix-vector multiply | OUT*IN+2 cycles |

### FP16 primitives

| Module | Description | Latency |
|--------|-------------|---------|
| `fp16_add.v` | IEEE 754 half-precision adder, RNE rounding | 1 cycle |
| `fp16_mul.v` | IEEE 754 half-precision multiplier, 1 DSP48 | 1 cycle |
| `fp16_mac.v` | Multiply-accumulate (comb mul + reg add) | 1 cycle |
| `fp16_rsqrt.v` | Reciprocal square root via LOD-LUT-shift | 2 cycles |
| `fp16_from_int8.v` | Signed int8 to fp16 (exact) | combinational |
| `fp16_to_int8.v` | FP16 to signed int8 with RNE + clamp | combinational |
| `fp16_to_q167.v` | FP16 to signed Q16.7 (softmax input) | combinational |
| `q115_to_fp16.v` | Unsigned Q1.15 to fp16 (softmax output) | combinational |

### Storage

| Module | Description | Latency |
|--------|-------------|---------|
| `weight_store.v` | 18 BRAM ROMs + 1 shared LN BRAM, tensor mux | 1 cycle |
| `weight_rom.v` | Synchronous-read BRAM ROM primitive | 1 cycle |
| `kv_cache.v` | 4 layers x 8 heads, fp16, 256 positions | 2 cycles |
| `kv_ram.v` | Read-write BRAM primitive | 1 cycle |

## BRAM Budget

339 / 365 BRAM36 used (93%).

| Resource | BRAM36 | Notes |
|----------|--------|-------|
| Weight BRAMs | 209 | 18 individual + 1 shared LN (2304 bytes) |
| KV cache | 128 | 64 K + 64 V (fp16, 32 banks each, 2 BRAM36/bank) |
| Softmax LUTs | 1.5 | Bipartite exp: input buf + lut0 + lut1 |
| inv_sqrt LUT | 0.5 | 512x16-bit for fp16_rsqrt |
| GELU | 0 | PWL coefficients as localparams |
| **Available** | **26** | |

## Training

Character-level GPT trained on Shakespeare (~1.1 MB).

`scripts/train/train.py`: Standard transformer, 30K steps, AdamW. Post-training int8 quantization (per-tensor symmetric, scale = max/127). Weights exported to `weights_int8.bin` (custom binary format with per-tensor int8 data + float32 scales).

### Weight extraction

```bash
python3 scripts/extract_weights.py
```

Reads `weights_int8.bin`, writes 36 `.hex` files to `mem/` for `$readmemh`, and generates `rtl/weight_scales.vh` with fp16 scale localparams.

## Simulation

Requires Vivado 2025.2 (runs in Docker container).

### Run all tests

```bash
./scripts/run_tests.sh
```

### Run a specific test

```bash
./scripts/run_tests.sh transformer_top
./scripts/run_tests.sh transformer_top_stress
```

The test flow:
1. **xsim** (Vivado simulator): compiles RTL + testbench, runs simulation, writes log
2. **Validation** (Python): parses xsim log, compares against RTL-exact Python model (`rtl_ops.py`)

### RTL inference (pure Python)

Run the bit-exact RTL-exact model without Vivado:

```bash
source scripts/train/venv/bin/activate
cd scripts/tests
python3 rtl_inference.py              # token 65 ('A'), 20 tokens
python3 rtl_inference.py "Hello" 50   # prompt "Hello", 50 tokens
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

Stress tests (`tb_*_stress.v`) run extended sequences (50 positions, 20 autoregressive tokens) and reuse the same validators.

### RTL-exact model

- **`rtl_ops.py`**: Pure-Python fp16 primitives that replicate RTL rounding bit-for-bit. All validation compares xsim output against this model.
- **`ideal_ops.py`**: Float64 reference models for error analysis (not used for pass/fail).

## Project Structure

```
rtl/                    21 Verilog modules
  weight_scales.vh      Auto-generated fp16 scale localparams
tb/                     13 testbenches
mem/                    Weight hex files, LUTs, test vectors
scripts/
  extract_weights.py    Binary weights -> hex + scales
  run_tests.sh          Simulation + validation runner
  gen_*.py              LUT and test vector generators
  tests/                Validation scripts + RTL-exact model
  train/                Training code, checkpoint, data
docs/                   Style guide
```

## Paper-based implementations

Three modules are based on published hardware designs:

| Module | Paper | What it provides | Generator script |
|--------|-------|------------------|------------------|
| `gelu.v` | [Huang et al., Electronics 2025, 14(9), 1825](https://www.mdpi.com/2079-9292/14/9/1825) | ISPA piecewise-linear erf approximation, 16 non-uniform segments, EPSS breakpoint optimization | `scripts/gen_gelu_pwl_coeffs.py` |
| `softmax.v` | [Kang & Wang, Micromachines 2026, 17(1), 84](https://www.mdpi.com/2072-666X/17/1/84) | Division-free SafeSoftmax via bipartite exp(-d) LUT with compensated initialization | `scripts/gen_softmax_luts.py` |
| `fp16_rsqrt.v` | [Kang & Wang, Micromachines 2026, 17(1), 84](https://www.mdpi.com/2072-666X/17/1/84) | LOD-LUT-shift reciprocal square root, reused inside `layernorm.v` | `scripts/gen_inv_sqrt_lut.py` |


### TODO
- Full sampler (temperature scaling, repetition penalty, top-k, LFSR sampling)
- UART interface (rx, tx, FIFO)
- System top (UART <-> transformer_top, clock/reset)
- LayerNorm optimization: overlap LOAD_GAMMA/LOAD_BETA with MEAN_ACC/VAR_ACC (~258 cycles saved per LN, ~2K per layer)
- Performance: use DSP48 fp16 multiply for parallel MAC lanes in matvec, reducing per-token latency
- FPGA synthesis and board bring-up