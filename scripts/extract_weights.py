"""
Extract int8 weights from weights_int8.bin into individual .hex files
for Verilog $readmemh, plus weight_scales.vh with fp16 scale factors

W8A16 pipeline: int8 weights stay in BRAM, dequantized to fp16 at runtime
via fp16_from_int8(w) * fp16_scale. Scales are stored as 16-bit IEEE 754.

Binary format:
  Global header: 8-byte magic "TFPGA001" + uint32 num_tensors
  Per tensor:    uint32 name_len, name bytes, uint32 ndim,
                 ndim x uint32 shape, float32 scale, then raw int8 data
"""

import struct
import os
import sys
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(PROJ, "scripts", "train", "weights_int8.bin")
MEM = os.path.join(PROJ, "mem")
RTL = os.path.join(PROJ, "rtl")
TB = os.path.join(PROJ, "tb")

# Base path used in Vivado/xsim on the build machine for $readmemh, log paths, etc.
VIVADO_BASE = "/home/user/red_eyes_is_all_you_need"

"""Convert tensor name to a clean filename stem
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

W16_SUFFIXES = (
    "_attn_qkv_weight",
    "_attn_proj_weight",
    "_ff_up_weight",
    "_ff_down_weight",
    "tok_emb_weight",
)

# Banks consumed by matvec_fp16_w32 use 256-bit packed words pairing col and
# col+IN/2 for the same 16 rows. tok_emb stays 128-bit for the w16 head_proj
W32_SUFFIXES = (
    "_attn_qkv_weight",
    "_attn_proj_weight",
    "_ff_up_weight",
    "_ff_down_weight",
)

def is_w16(stem):
    return any(stem.endswith(s) for s in W16_SUFFIXES)

def is_w32(stem):
    return any(stem.endswith(s) for s in W32_SUFFIXES)


# Generate tb/tb_weight_store.v with expected values derived from tensor data.
# Tests both the 8-bit weight_store for LN gamma/beta and the per-tensor
# 128-bit banks for qkv/proj/ff_up/ff_down/tok_emb.
def generate_tb_weight_store(tensors):
    log_path = f"{VIVADO_BASE}/logs/tb_weight_store.log"

    keep = [(i, t) for i, t in enumerate(tensors) if not is_w16(t["stem"])]
    n = len(keep)

    lines_first = []
    lines_last  = []
    lines_addr  = []
    lines_tsel  = []
    for j, (orig_i, t) in enumerate(keep):
        fb = t["data"][0]
        lb = t["data"][-1]
        la = t["size"] - 1
        lines_first.append(f"exp_first[{j:2d}] = 8'h{fb:02x};")
        lines_last.append(f"exp_last[{j:2d}] = 8'h{lb:02x};")
        lines_addr.append(f"last_addr[{j:2d}] = 16'd{la};")
        lines_tsel.append(f"tsel_lut[{j:2d}] = 6'd{orig_i};")

    # W16 banks: gather per-layer first/last bytes plus original tensor index
    bucket = {"qkv": [], "proj": [], "ff_up": [], "ff_down": []}
    tok_emb = None
    for i, t in enumerate(tensors):
        stem = t["stem"]
        if stem == "tok_emb_weight":
            tok_emb = (i, t)
            continue
        for kind in bucket:
            if stem.endswith(f"_{kind}_weight") or stem.endswith(f"_attn_{kind}_weight"):
                layer = int(stem.replace("block", "").split("_")[0])
                bucket[kind].append((layer, i, t))
                break
    for k in bucket:
        bucket[k].sort(key=lambda x: x[0])

    bank_meta = {
        "qkv":     {"depth": 1536, "addr_w": 11},
        "proj":    {"depth": 512,  "addr_w": 9},
        "ff_up":   {"depth": 2048, "addr_w": 11},
        "ff_down": {"depth": 2048, "addr_w": 11},
    }
    tok_depth = (tok_emb[1]["shape"][0] // 16) * tok_emb[1]["shape"][1]

    def bank_init(items, name):
        out = []
        for layer, orig_i, t in items:
            fb = t["data"][0]
            lb = t["data"][-1]
            out.append(f"{name}_first[{layer}] = 8'h{fb:02x};")
            out.append(f"{name}_last[{layer}]  = 8'h{lb:02x};")
            out.append(f"{name}_idx[{layer}]   = 6'd{orig_i};")
        return out

    qkv_init     = bank_init(bucket["qkv"],     "qkv")
    proj_init    = bank_init(bucket["proj"],    "proj")
    ff_up_init   = bank_init(bucket["ff_up"],   "ff_up")
    ff_down_init = bank_init(bucket["ff_down"], "ff_down")

    tok_first = tok_emb[1]["data"][0]
    tok_last  = tok_emb[1]["data"][-1]
    tok_idx   = tok_emb[0]

    bank_total_tests = 4 * 4 * 2 + 2  # 4 banks x 4 layers x (first,last) + tok_emb (first,last)
    total_tests = n * 2 + bank_total_tests

    def paired(items, indent="    "):
        out = []
        for j in range(0, len(items), 2):
            if j + 1 < len(items):
                out.append(f"{indent}{items[j]}  {items[j+1]}")
            else:
                out.append(f"{indent}{items[j]}")
        return "\n".join(out)

    tb = f"""`timescale 1ns / 1ps
// Auto-generated by extract_weights.py - DO NOT EDIT

module tb_weight_store;

  reg         clk;
  reg  [ 5:0] tensor_sel;
  reg  [15:0] addr;
  wire [ 7:0] data;
  wire [31:0] scale;

  weight_store uut (
    .clk_i       (clk),
    .tensor_sel_i(tensor_sel),
    .addr_i      (addr[14:0]),
    .data_o      (data),
    .scale_o     (scale)
  );

  // W32 banks: 256-bit packed words pairing col and col+IN/2 per row group.
  // tok_emb stays in 128-bit w16 packing because head_proj uses matvec_fp16_w16
  reg [1:0]  layer;
  reg [{bank_meta["qkv"]["addr_w"]-1}:0] qkv_addr;
  reg [{bank_meta["proj"]["addr_w"]-1}:0]  proj_addr;
  reg [{bank_meta["ff_up"]["addr_w"]-1}:0] ff_up_addr;
  reg [{bank_meta["ff_down"]["addr_w"]-1}:0] ff_down_addr;
  reg [10:0] tok_emb_addr;
  wire [255:0] qkv_data, proj_data, ff_up_data, ff_down_data;
  wire [127:0] tok_emb_data;
  wire [15:0]  qkv_scale, proj_scale, ff_up_scale, ff_down_scale, tok_emb_scale;

  weight_store_qkv     u_ws_qkv     (.clk_i(clk), .layer_i(layer),
    .addr_i(qkv_addr), .data_o(qkv_data), .scale_o(qkv_scale));
  weight_store_proj    u_ws_proj    (.clk_i(clk), .layer_i(layer),
    .addr_i(proj_addr), .data_o(proj_data), .scale_o(proj_scale));
  weight_store_ff_up   u_ws_ff_up   (.clk_i(clk), .layer_i(layer),
    .addr_i(ff_up_addr), .data_o(ff_up_data), .scale_o(ff_up_scale));
  weight_store_ff_down u_ws_ff_down (.clk_i(clk), .layer_i(layer),
    .addr_i(ff_down_addr), .data_o(ff_down_data), .scale_o(ff_down_scale));
  weight_store_tok_emb u_ws_tok_emb (.clk_i(clk), .addr_i(tok_emb_addr), .data_o(tok_emb_data), .scale_o(tok_emb_scale));

  initial clk = 1'b0;
  always #5 clk = ~clk;

  reg [7:0]  exp_first [0:{n-1}];
  reg [7:0]  exp_last  [0:{n-1}];
  reg [15:0] last_addr [0:{n-1}];
  reg [5:0]  tsel_lut  [0:{n-1}];

  reg [7:0] qkv_first     [0:3], qkv_last     [0:3];
  reg [7:0] proj_first    [0:3], proj_last    [0:3];
  reg [7:0] ff_up_first   [0:3], ff_up_last   [0:3];
  reg [7:0] ff_down_first [0:3], ff_down_last [0:3];
  reg [5:0] qkv_idx [0:3], proj_idx [0:3], ff_up_idx [0:3], ff_down_idx [0:3];

  integer errors;
  integer i;
  integer fd;

  initial begin
{paired(lines_first)}

{paired(lines_last)}

{paired(lines_addr)}

{paired(lines_tsel)}

{paired(qkv_init)}

{paired(proj_init)}

{paired(ff_up_init)}

{paired(ff_down_init)}

    errors = 0;
    layer        = 2'd0;
    qkv_addr     = {bank_meta["qkv"]["addr_w"]}'d0;
    proj_addr    = {bank_meta["proj"]["addr_w"]}'d0;
    ff_up_addr   = {bank_meta["ff_up"]["addr_w"]}'d0;
    ff_down_addr = {bank_meta["ff_down"]["addr_w"]}'d0;
    tok_emb_addr = 11'd0;

    fd = $fopen("{log_path}", "w");

    #20;

    $display("=== Weight Store Testbench ===");
    $fwrite(fd, "=== Weight Store Testbench ===\\n");

    // 8-bit LN store: first and last byte of each tensor
    for (i = 0; i < {n}; i = i + 1) begin
      tensor_sel = tsel_lut[i];
      addr       = 16'd0;
      @(posedge clk);
      @(posedge clk);
      #1;

      if (data !== exp_first[i]) begin
        $display("FAIL tensor %0d first: got 0x%02x, expected 0x%02x", tsel_lut[i], data, exp_first[i]);
        $fwrite(fd, "FAIL tensor %0d first: got 0x%02x, expected 0x%02x\\n", tsel_lut[i], data, exp_first[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=0  data=0x%02x  scale=0x%08x", tsel_lut[i], data, scale);
        $fwrite(fd, "OK   tensor %0d  addr=0  data=0x%02x  scale=0x%08x\\n", tsel_lut[i], data, scale);
      end

      addr = last_addr[i];
      @(posedge clk);
      @(posedge clk);
      #1;

      if (data !== exp_last[i]) begin
        $display("FAIL tensor %0d last: got 0x%02x, exp 0x%02x (addr=%0d)",
                 tsel_lut[i], data, exp_last[i], last_addr[i]);
        $fwrite(fd, "FAIL tensor %0d last: got 0x%02x, exp 0x%02x (addr=%0d)\\n",
                tsel_lut[i], data, exp_last[i], last_addr[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=%0d  data=0x%02x", tsel_lut[i], last_addr[i], data);
        $fwrite(fd, "OK   tensor %0d  addr=%0d  data=0x%02x\\n", tsel_lut[i], last_addr[i], data);
      end
    end

    // W16 banks: per layer, byte 0 of word 0 (first tensor byte) and byte 15
    // of word depth-1 (last tensor byte). 2-cycle read latency: addr_r reg
    // plus BRAM output reg
    for (i = 0; i < 4; i = i + 1) begin
      layer    = i[1:0];
      qkv_addr = {bank_meta["qkv"]["addr_w"]}'d0;
      @(posedge clk);
      @(posedge clk);
      #1;
      if (qkv_data[7:0] !== qkv_first[i]) begin
        $display("FAIL tensor %0d first: got 0x%02x, expected 0x%02x", qkv_idx[i], qkv_data[7:0], qkv_first[i]);
        $fwrite(fd, "FAIL tensor %0d first: got 0x%02x, expected 0x%02x\\n", qkv_idx[i], qkv_data[7:0], qkv_first[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=0  data=0x%02x", qkv_idx[i], qkv_data[7:0]);
        $fwrite(fd, "OK   tensor %0d  addr=0  data=0x%02x\\n", qkv_idx[i], qkv_data[7:0]);
      end

      qkv_addr = {bank_meta["qkv"]["addr_w"]}'d{bank_meta["qkv"]["depth"] - 1};
      @(posedge clk);
      @(posedge clk);
      #1;
      if (qkv_data[255:248] !== qkv_last[i]) begin
        $display("FAIL tensor %0d last: got 0x%02x, exp 0x%02x", qkv_idx[i], qkv_data[255:248], qkv_last[i]);
        $fwrite(fd, "FAIL tensor %0d last: got 0x%02x, exp 0x%02x\\n", qkv_idx[i], qkv_data[255:248], qkv_last[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=%0d  data=0x%02x", qkv_idx[i], qkv_addr, qkv_data[255:248]);
        $fwrite(fd, "OK   tensor %0d  addr=%0d  data=0x%02x\\n", qkv_idx[i], qkv_addr, qkv_data[255:248]);
      end
    end

    for (i = 0; i < 4; i = i + 1) begin
      layer     = i[1:0];
      proj_addr = {bank_meta["proj"]["addr_w"]}'d0;
      @(posedge clk);
      @(posedge clk);
      #1;
      if (proj_data[7:0] !== proj_first[i]) begin
        $display("FAIL tensor %0d first: got 0x%02x, expected 0x%02x", proj_idx[i], proj_data[7:0], proj_first[i]);
        $fwrite(fd, "FAIL tensor %0d first: got 0x%02x, expected 0x%02x\\n", proj_idx[i], proj_data[7:0], proj_first[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=0  data=0x%02x", proj_idx[i], proj_data[7:0]);
        $fwrite(fd, "OK   tensor %0d  addr=0  data=0x%02x\\n", proj_idx[i], proj_data[7:0]);
      end

      proj_addr = {bank_meta["proj"]["addr_w"]}'d{bank_meta["proj"]["depth"] - 1};
      @(posedge clk);
      @(posedge clk);
      #1;
      if (proj_data[255:248] !== proj_last[i]) begin
        $display("FAIL tensor %0d last: got 0x%02x, exp 0x%02x", proj_idx[i], proj_data[255:248], proj_last[i]);
        $fwrite(fd, "FAIL tensor %0d last: got 0x%02x, exp 0x%02x\\n", proj_idx[i], proj_data[255:248], proj_last[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=%0d  data=0x%02x", proj_idx[i], proj_addr, proj_data[255:248]);
        $fwrite(fd, "OK   tensor %0d  addr=%0d  data=0x%02x\\n", proj_idx[i], proj_addr, proj_data[255:248]);
      end
    end

    for (i = 0; i < 4; i = i + 1) begin
      layer      = i[1:0];
      ff_up_addr = {bank_meta["ff_up"]["addr_w"]}'d0;
      @(posedge clk);
      @(posedge clk);
      #1;
      if (ff_up_data[7:0] !== ff_up_first[i]) begin
        $display("FAIL tensor %0d first: got 0x%02x, expected 0x%02x", ff_up_idx[i], ff_up_data[7:0], ff_up_first[i]);
        $fwrite(fd, "FAIL tensor %0d first: got 0x%02x, expected 0x%02x\\n", ff_up_idx[i], ff_up_data[7:0], ff_up_first[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=0  data=0x%02x", ff_up_idx[i], ff_up_data[7:0]);
        $fwrite(fd, "OK   tensor %0d  addr=0  data=0x%02x\\n", ff_up_idx[i], ff_up_data[7:0]);
      end

      ff_up_addr = {bank_meta["ff_up"]["addr_w"]}'d{bank_meta["ff_up"]["depth"] - 1};
      @(posedge clk);
      @(posedge clk);
      #1;
      if (ff_up_data[255:248] !== ff_up_last[i]) begin
        $display("FAIL tensor %0d last: got 0x%02x, exp 0x%02x", ff_up_idx[i], ff_up_data[255:248], ff_up_last[i]);
        $fwrite(fd, "FAIL tensor %0d last: got 0x%02x, exp 0x%02x\\n", ff_up_idx[i], ff_up_data[255:248], ff_up_last[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=%0d  data=0x%02x", ff_up_idx[i], ff_up_addr, ff_up_data[255:248]);
        $fwrite(fd, "OK   tensor %0d  addr=%0d  data=0x%02x\\n", ff_up_idx[i], ff_up_addr, ff_up_data[255:248]);
      end
    end

    for (i = 0; i < 4; i = i + 1) begin
      layer        = i[1:0];
      ff_down_addr = {bank_meta["ff_down"]["addr_w"]}'d0;
      @(posedge clk);
      @(posedge clk);
      #1;
      if (ff_down_data[7:0] !== ff_down_first[i]) begin
        $display("FAIL tensor %0d first: got 0x%02x, expected 0x%02x", ff_down_idx[i], ff_down_data[7:0], ff_down_first[i]);
        $fwrite(fd, "FAIL tensor %0d first: got 0x%02x, expected 0x%02x\\n", ff_down_idx[i], ff_down_data[7:0], ff_down_first[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=0  data=0x%02x", ff_down_idx[i], ff_down_data[7:0]);
        $fwrite(fd, "OK   tensor %0d  addr=0  data=0x%02x\\n", ff_down_idx[i], ff_down_data[7:0]);
      end

      ff_down_addr = {bank_meta["ff_down"]["addr_w"]}'d{bank_meta["ff_down"]["depth"] - 1};
      @(posedge clk);
      @(posedge clk);
      #1;
      if (ff_down_data[255:248] !== ff_down_last[i]) begin
        $display("FAIL tensor %0d last: got 0x%02x, exp 0x%02x", ff_down_idx[i], ff_down_data[255:248], ff_down_last[i]);
        $fwrite(fd, "FAIL tensor %0d last: got 0x%02x, exp 0x%02x\\n", ff_down_idx[i], ff_down_data[255:248], ff_down_last[i]);
        errors = errors + 1;
      end else begin
        $display("OK   tensor %0d  addr=%0d  data=0x%02x", ff_down_idx[i], ff_down_addr, ff_down_data[255:248]);
        $fwrite(fd, "OK   tensor %0d  addr=%0d  data=0x%02x\\n", ff_down_idx[i], ff_down_addr, ff_down_data[255:248]);
      end
    end

    // tok_emb has no layer dimension
    tok_emb_addr = 11'd0;
    @(posedge clk);
    @(posedge clk);
    #1;
    if (tok_emb_data[7:0] !== 8'h{tok_first:02x}) begin
      $display("FAIL tensor {tok_idx} first: got 0x%02x, expected 0x{tok_first:02x}", tok_emb_data[7:0]);
      $fwrite(fd, "FAIL tensor {tok_idx} first: got 0x%02x, expected 0x{tok_first:02x}\\n", tok_emb_data[7:0]);
      errors = errors + 1;
    end else begin
      $display("OK   tensor {tok_idx}  addr=0  data=0x%02x", tok_emb_data[7:0]);
      $fwrite(fd, "OK   tensor {tok_idx}  addr=0  data=0x%02x\\n", tok_emb_data[7:0]);
    end

    tok_emb_addr = 11'd{tok_depth - 1};
    @(posedge clk);
    @(posedge clk);
    #1;
    if (tok_emb_data[127:120] !== 8'h{tok_last:02x}) begin
      $display("FAIL tensor {tok_idx} last: got 0x%02x, exp 0x{tok_last:02x}", tok_emb_data[127:120]);
      $fwrite(fd, "FAIL tensor {tok_idx} last: got 0x%02x, exp 0x{tok_last:02x}\\n", tok_emb_data[127:120]);
      errors = errors + 1;
    end else begin
      $display("OK   tensor {tok_idx}  addr=%0d  data=0x%02x", tok_emb_addr, tok_emb_data[127:120]);
      $fwrite(fd, "OK   tensor {tok_idx}  addr=%0d  data=0x%02x\\n", tok_emb_addr, tok_emb_data[127:120]);
    end

    if (errors == 0) begin
      $display("=== All {total_tests} tests passed ===");
      $fwrite(fd, "=== All {total_tests} tests passed ===\\n");
    end else begin
      $display("=== %0d errors out of {total_tests} ===", errors);
      $fwrite(fd, "=== %0d errors out of {total_tests} ===\\n", errors);
    end

    $fclose(fd);
    $finish;
  end

endmodule"""

    os.makedirs(TB, exist_ok=True)
    tb_path = os.path.join(TB, "tb_weight_store.v")
    with open(tb_path, "w") as f:
        f.write(tb)
    print(f"Wrote {tb_path}")


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
            scale = struct.unpack("<f", f.read(4))[0]
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
                "hex_file": hex_file,
                "data": data,
            })

    for t in tensors:
        # Skip individual LN files; packed into ln_params.hex instead
        if t["size"] == 128:
            continue
        hex_path = os.path.join(MEM, t["hex_file"])
        if is_w32(t["stem"]):
            out_dim, in_dim = t["shape"]
            data = np.frombuffer(t["data"], dtype=np.uint8).reshape(out_dim, in_dim)
            with open(hex_path, "w") as hf:
                for group_idx in range(out_dim // 16):
                    rows = [data[16 * group_idx + lane] for lane in range(16)]
                    for col in range(in_dim // 2):
                        word = 0
                        for lane in range(16):
                            word |= int(rows[lane][col])              << (lane * 8)
                            word |= int(rows[lane][col + in_dim // 2]) << (lane * 8 + 128)
                        hf.write(f"{word:064x}\n")
            print(f"  wrote {hex_path}: {t['size']} bytes packed into {(out_dim // 16) * (in_dim // 2)} 256-bit words")
        elif is_w16(t["stem"]):
            out_dim, in_dim = t["shape"]
            data = np.frombuffer(t["data"], dtype=np.uint8).reshape(out_dim, in_dim)
            with open(hex_path, "w") as hf:
                for group_idx in range(out_dim // 16):
                    rows = [data[16 * group_idx + lane] for lane in range(16)]
                    for col in range(in_dim):
                        word = 0
                        for lane in range(16):
                            word |= int(rows[lane][col]) << (lane * 8)
                        hf.write(f"{word:032x}\n")
            print(f"  wrote {hex_path}: {t['size']} bytes packed into {(out_dim // 16) * in_dim} 128-bit words")
        else:
            with open(hex_path, "w") as hf:
                for byte in t["data"]:
                    # int8 stored as unsigned byte; write as 2-digit hex
                    hf.write(f"{byte:02x}\n")
            print(f"  wrote {hex_path}: {t['size']} bytes")

    # Re-read hex and compare
    errors = 0
    for t in tensors:
        if t["size"] == 128:
            continue
        hex_path = os.path.join(MEM, t["hex_file"])
        with open(hex_path, "r") as hf:
            lines = hf.read().strip().split("\n")
        if is_w32(t["stem"]):
            out_dim, in_dim = t["shape"]
            expected_lines = (out_dim // 16) * (in_dim // 2)
            data = np.frombuffer(t["data"], dtype=np.uint8).reshape(out_dim, in_dim)
            if len(lines) != expected_lines:
                print(f"VERIFY FAIL: {t['name']} line count {len(lines)} != {expected_lines}")
                errors += 1
                continue
            stop = False
            for group_idx in range(out_dim // 16):
                rows = [data[16 * group_idx + lane] for lane in range(16)]
                for col in range(in_dim // 2):
                    word_idx = group_idx * (in_dim // 2) + col
                    word = int(lines[word_idx], 16)
                    expected = 0
                    for lane in range(16):
                        expected |= int(rows[lane][col])              << (lane * 8)
                        expected |= int(rows[lane][col + in_dim // 2]) << (lane * 8 + 128)
                    if word != expected:
                        print(f"VERIFY FAIL: {t['name']} word[{word_idx}] hex={word:064x} expected {expected:064x}")
                        errors += 1
                        stop = True
                        break
                if stop:
                    break
        elif is_w16(t["stem"]):
            out_dim, in_dim = t["shape"]
            expected_lines = (out_dim // 16) * in_dim
            data = np.frombuffer(t["data"], dtype=np.uint8).reshape(out_dim, in_dim)
            if len(lines) != expected_lines:
                print(f"VERIFY FAIL: {t['name']} line count {len(lines)} != {expected_lines}")
                errors += 1
                continue
            stop = False
            for group_idx in range(out_dim // 16):
                rows = [data[16 * group_idx + lane] for lane in range(16)]
                for col in range(in_dim):
                    word_idx = group_idx * in_dim + col
                    word = int(lines[word_idx], 16)
                    expected = 0
                    for lane in range(16):
                        expected |= int(rows[lane][col]) << (lane * 8)
                    if word != expected:
                        print(f"VERIFY FAIL: {t['name']} word[{word_idx}] hex={word:032x} expected {expected:032x}")
                        errors += 1
                        stop = True
                        break
                if stop:
                    break
        else:
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
        print(f"\n{errors} verification errors", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tensors)} hex files verified")

    vh_path = os.path.join(RTL, "weight_scales.vh")
    with open(vh_path, "w") as vf:
        vf.write("// Auto-generated by extract_weights.py - DO NOT EDIT\n")
        vf.write("// FP16 scale factors for W8A16 dequant: fp16_from_int8(w) * scale\n\n")
        for t in tensors:
            param = verilog_param_name(t["stem"])
            fp16_val = np.float16(t["scale"])
            fp16_bits = int(fp16_val.view(np.uint16))
            vf.write(f"localparam [15:0] {param:50s} = 16'h{fp16_bits:04x};"
                     f"  // {float(fp16_val)}\n")
    print(f"Wrote {vh_path}")

    # Generate testbench with expected values from tensor data
    generate_tb_weight_store(tensors)

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