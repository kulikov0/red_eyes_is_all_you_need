// 18 large BRAM ROMs + 1 combined LN BRAM (18x128 = 2304 bytes)
// tensor_sel[5:0] + addr -> 8-bit weight data + 32-bit IEEE754 scale
//
// Tensor map (matches extract_weights.py ordering):
// 0  tok_emb         256x128  = 32768   BRAM (individual)
// 1  pos_emb         256x128  = 32768   BRAM (individual)
// 2  block0_ln1_w    128               BRAM (shared ln_params)
// 3  block0_ln1_b    128               BRAM (shared ln_params)
// 4  block0_qkv      384x128 = 49152   BRAM (individual)
// 5  block0_proj     128x128 = 16384   BRAM (individual)
// 6  block0_ln2_w    128               BRAM (shared ln_params)
// 7  block0_ln2_b    128               BRAM (shared ln_params)
// 8  block0_ff_up    512x128 = 65536   BRAM (individual)
// 9  block0_ff_down  128x512 = 65536   BRAM (individual)
// 10  block1_ln1_w    128               BRAM (shared ln_params)
// 11  block1_ln1_b    128               BRAM (shared ln_params)
// 12  block1_qkv      384x128 = 49152   BRAM (individual)
// 13  block1_proj     128x128 = 16384   BRAM (individual)
// 14  block1_ln2_w    128               BRAM (shared ln_params)
// 15  block1_ln2_b    128               BRAM (shared ln_params)
// 16  block1_ff_up    512x128 = 65536   BRAM (individual)
// 17  block1_ff_down  128x512 = 65536   BRAM (individual)
// 18  block2_ln1_w    128               BRAM (shared ln_params)
// 19  block2_ln1_b    128               BRAM (shared ln_params)
// 20  block2_qkv      384x128 = 49152   BRAM (individual)
// 21  block2_proj     128x128 = 16384   BRAM (individual)
// 22  block2_ln2_w    128               BRAM (shared ln_params)
// 23  block2_ln2_b    128               BRAM (shared ln_params)
// 24  block2_ff_up    512x128 = 65536   BRAM (individual)
// 25  block2_ff_down  128x512 = 65536   BRAM (individual)
// 26  block3_ln1_w    128               BRAM (shared ln_params)
// 27  block3_ln1_b    128               BRAM (shared ln_params)
// 28  block3_qkv      384x128 = 49152   BRAM (individual)
// 29  block3_proj     128x128 = 16384   BRAM (individual)
// 30  block3_ln2_w    128               BRAM (shared ln_params)
// 31  block3_ln2_b    128               BRAM (shared ln_params)
// 32  block3_ff_up    512x128 = 65536   BRAM (individual)
// 33  block3_ff_down  128x512 = 65536   BRAM (individual)
// 34  ln_f_w          128               BRAM (shared ln_params)
// 35  ln_f_b          128               BRAM (shared ln_params)
//
// Note: head.weight is tied to tok_emb (index 0), no duplicate storage.

module weight_store (
  input  wire        clk_i,
  input  wire [ 5:0] tensor_sel_i,  // 0..35
  input  wire [15:0] addr_i,        // max depth 65536 -> 16 bits
  output reg  [ 7:0] data_o,
  output reg  [31:0] scale_o
);

  // Scale factors (IEEE 754)
  `include "weight_scales.vh"

  // BRAM ROM outputs (synchronous, 1-cycle latency)
  wire [7:0] d_tok_emb,       d_pos_emb;
  wire [7:0] d_b0_qkv,       d_b0_proj,    d_b0_ff_up,   d_b0_ff_down;
  wire [7:0] d_b1_qkv,       d_b1_proj,    d_b1_ff_up,   d_b1_ff_down;
  wire [7:0] d_b2_qkv,       d_b2_proj,    d_b2_ff_up,   d_b2_ff_down;
  wire [7:0] d_b3_qkv,       d_b3_proj,    d_b3_ff_up,   d_b3_ff_down;

  // Combined LayerNorm ROM: 18 tensors x 128 bytes = 2304 in one BRAM
  // Offset LUT maps tensor_sel to base address within ln_params.hex
  wire [7:0] d_ln;
  reg [11:0] ln_offset;
  wire [11:0] ln_addr = ln_offset + {5'd0, addr_i[6:0]};

  // Large weight ROMs

  // Embedding
  weight_rom #(
    .DEPTH(32768),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/tok_emb_weight.hex")
  ) u_tok_emb (
    .clk_i (clk_i),
    .addr_i(addr_i[14:0]),
    .data_o(d_tok_emb)
  );

  weight_rom #(
    .DEPTH(32768),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/pos_emb_weight.hex")
  ) u_pos_emb (
    .clk_i (clk_i),
    .addr_i(addr_i[14:0]),
    .data_o(d_pos_emb)
  );

  // Block 0
  weight_rom #(
    .DEPTH(49152),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_attn_qkv_weight.hex")
  ) u_b0_qkv (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b0_qkv)
  );

  weight_rom #(
    .DEPTH(16384),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_attn_proj_weight.hex")
  ) u_b0_proj (
    .clk_i (clk_i),
    .addr_i(addr_i[13:0]),
    .data_o(d_b0_proj)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_ff_up_weight.hex")
  ) u_b0_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b0_ff_up)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_ff_down_weight.hex")
  ) u_b0_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b0_ff_down)
  );

  // Block 1
  weight_rom #(
    .DEPTH(49152),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_attn_qkv_weight.hex")
  ) u_b1_qkv (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b1_qkv)
  );

  weight_rom #(
    .DEPTH(16384),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_attn_proj_weight.hex")
  ) u_b1_proj (
    .clk_i (clk_i),
    .addr_i(addr_i[13:0]),
    .data_o(d_b1_proj)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_ff_up_weight.hex")
  ) u_b1_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b1_ff_up)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_ff_down_weight.hex")
  ) u_b1_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b1_ff_down)
  );

  // Block 2
  weight_rom #(
    .DEPTH(49152),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_attn_qkv_weight.hex")
  ) u_b2_qkv (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b2_qkv)
  );

  weight_rom #(
    .DEPTH(16384),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_attn_proj_weight.hex")
  ) u_b2_proj (
    .clk_i (clk_i),
    .addr_i(addr_i[13:0]),
    .data_o(d_b2_proj)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_ff_up_weight.hex")
  ) u_b2_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b2_ff_up)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_ff_down_weight.hex")
  ) u_b2_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b2_ff_down)
  );

  // Block 3
  weight_rom #(
    .DEPTH(49152),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_attn_qkv_weight.hex")
  ) u_b3_qkv (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b3_qkv)
  );

  weight_rom #(
    .DEPTH(16384),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_attn_proj_weight.hex")
  ) u_b3_proj (
    .clk_i (clk_i),
    .addr_i(addr_i[13:0]),
    .data_o(d_b3_proj)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_ff_up_weight.hex")
  ) u_b3_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b3_ff_up)
  );

  weight_rom #(
    .DEPTH(65536),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_ff_down_weight.hex")
  ) u_b3_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_i[15:0]),
    .data_o(d_b3_ff_down)
  );

  //  Combined LayerNorm ROM
  weight_rom #(
    .DEPTH(2304),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/ln_params.hex")
  ) u_ln (
    .clk_i (clk_i),
    .addr_i(ln_addr),
    .data_o(d_ln)
  );

  // Offset LUT: tensor_sel -> base address in ln_params.hex
  // Order matches extract_weights.py: all size-128 tensors in bin order
  // idx 0: b0_ln1_w, 1: b0_ln1_b, 2: b0_ln2_w, ... 17: ln_f_b
  always @(*) begin
    case (tensor_sel_i)
      6'd2:  ln_offset = 12'd0;     // block0_ln1_weight
      6'd3:  ln_offset = 12'd128;   // block0_ln1_bias
      6'd6:  ln_offset = 12'd256;   // block0_ln2_weight
      6'd7:  ln_offset = 12'd384;   // block0_ln2_bias
      6'd10: ln_offset = 12'd512;   // block1_ln1_weight
      6'd11: ln_offset = 12'd640;   // block1_ln1_bias
      6'd14: ln_offset = 12'd768;   // block1_ln2_weight
      6'd15: ln_offset = 12'd896;   // block1_ln2_bias
      6'd18: ln_offset = 12'd1024;  // block2_ln1_weight
      6'd19: ln_offset = 12'd1152;  // block2_ln1_bias
      6'd22: ln_offset = 12'd1280;  // block2_ln2_weight
      6'd23: ln_offset = 12'd1408;  // block2_ln2_bias
      6'd26: ln_offset = 12'd1536;  // block3_ln1_weight
      6'd27: ln_offset = 12'd1664;  // block3_ln1_bias
      6'd30: ln_offset = 12'd1792;  // block3_ln2_weight
      6'd31: ln_offset = 12'd1920;  // block3_ln2_bias
      6'd34: ln_offset = 12'd2048;  // ln_f_weight
      6'd35: ln_offset = 12'd2176;  // ln_f_bias
      default: ln_offset = 12'd0;
    endcase
  end

  // Output MUX: tensor_sel -> data
  // Registered to match BRAM 1-cycle latency
  reg [5:0] sel_r;
  always @(posedge clk_i) sel_r <= tensor_sel_i;

  always @(*) begin
    case (sel_r)
      6'd0:  data_o = d_tok_emb;
      6'd1:  data_o = d_pos_emb;
      6'd2:  data_o = d_ln;
      6'd3:  data_o = d_ln;
      6'd4:  data_o = d_b0_qkv;
      6'd5:  data_o = d_b0_proj;
      6'd6:  data_o = d_ln;
      6'd7:  data_o = d_ln;
      6'd8:  data_o = d_b0_ff_up;
      6'd9:  data_o = d_b0_ff_down;
      6'd10: data_o = d_ln;
      6'd11: data_o = d_ln;
      6'd12: data_o = d_b1_qkv;
      6'd13: data_o = d_b1_proj;
      6'd14: data_o = d_ln;
      6'd15: data_o = d_ln;
      6'd16: data_o = d_b1_ff_up;
      6'd17: data_o = d_b1_ff_down;
      6'd18: data_o = d_ln;
      6'd19: data_o = d_ln;
      6'd20: data_o = d_b2_qkv;
      6'd21: data_o = d_b2_proj;
      6'd22: data_o = d_ln;
      6'd23: data_o = d_ln;
      6'd24: data_o = d_b2_ff_up;
      6'd25: data_o = d_b2_ff_down;
      6'd26: data_o = d_ln;
      6'd27: data_o = d_ln;
      6'd28: data_o = d_b3_qkv;
      6'd29: data_o = d_b3_proj;
      6'd30: data_o = d_ln;
      6'd31: data_o = d_ln;
      6'd32: data_o = d_b3_ff_up;
      6'd33: data_o = d_b3_ff_down;
      6'd34: data_o = d_ln;
      6'd35: data_o = d_ln;
      default: data_o = 8'd0;
    endcase
  end

  // Scale LUT: tensor_sel -> 32-bit IEEE 754 scale
  always @(posedge clk_i) begin
    case (tensor_sel_i)
      6'd0:  scale_o <= SCALE_TOK_EMB_WEIGHT;
      6'd1:  scale_o <= SCALE_POS_EMB_WEIGHT;
      6'd2:  scale_o <= SCALE_BLOCK0_LN1_WEIGHT;
      6'd3:  scale_o <= SCALE_BLOCK0_LN1_BIAS;
      6'd4:  scale_o <= SCALE_BLOCK0_ATTN_QKV_WEIGHT;
      6'd5:  scale_o <= SCALE_BLOCK0_ATTN_PROJ_WEIGHT;
      6'd6:  scale_o <= SCALE_BLOCK0_LN2_WEIGHT;
      6'd7:  scale_o <= SCALE_BLOCK0_LN2_BIAS;
      6'd8:  scale_o <= SCALE_BLOCK0_FF_UP_WEIGHT;
      6'd9:  scale_o <= SCALE_BLOCK0_FF_DOWN_WEIGHT;
      6'd10: scale_o <= SCALE_BLOCK1_LN1_WEIGHT;
      6'd11: scale_o <= SCALE_BLOCK1_LN1_BIAS;
      6'd12: scale_o <= SCALE_BLOCK1_ATTN_QKV_WEIGHT;
      6'd13: scale_o <= SCALE_BLOCK1_ATTN_PROJ_WEIGHT;
      6'd14: scale_o <= SCALE_BLOCK1_LN2_WEIGHT;
      6'd15: scale_o <= SCALE_BLOCK1_LN2_BIAS;
      6'd16: scale_o <= SCALE_BLOCK1_FF_UP_WEIGHT;
      6'd17: scale_o <= SCALE_BLOCK1_FF_DOWN_WEIGHT;
      6'd18: scale_o <= SCALE_BLOCK2_LN1_WEIGHT;
      6'd19: scale_o <= SCALE_BLOCK2_LN1_BIAS;
      6'd20: scale_o <= SCALE_BLOCK2_ATTN_QKV_WEIGHT;
      6'd21: scale_o <= SCALE_BLOCK2_ATTN_PROJ_WEIGHT;
      6'd22: scale_o <= SCALE_BLOCK2_LN2_WEIGHT;
      6'd23: scale_o <= SCALE_BLOCK2_LN2_BIAS;
      6'd24: scale_o <= SCALE_BLOCK2_FF_UP_WEIGHT;
      6'd25: scale_o <= SCALE_BLOCK2_FF_DOWN_WEIGHT;
      6'd26: scale_o <= SCALE_BLOCK3_LN1_WEIGHT;
      6'd27: scale_o <= SCALE_BLOCK3_LN1_BIAS;
      6'd28: scale_o <= SCALE_BLOCK3_ATTN_QKV_WEIGHT;
      6'd29: scale_o <= SCALE_BLOCK3_ATTN_PROJ_WEIGHT;
      6'd30: scale_o <= SCALE_BLOCK3_LN2_WEIGHT;
      6'd31: scale_o <= SCALE_BLOCK3_LN2_BIAS;
      6'd32: scale_o <= SCALE_BLOCK3_FF_UP_WEIGHT;
      6'd33: scale_o <= SCALE_BLOCK3_FF_DOWN_WEIGHT;
      6'd34: scale_o <= SCALE_LN_F_WEIGHT;
      6'd35: scale_o <= SCALE_LN_F_BIAS;
      default: scale_o <= 32'd0;
    endcase
  end

endmodule