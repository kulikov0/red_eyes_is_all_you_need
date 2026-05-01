// 128-bit weight ROM bank for the per-layer matvec tensor types and tok_emb.
// Each BRAM word packs 16 adjacent rows at the same column so matvec_fp16_w16
// reads 16 weights per cycle.
//
// w16_sel encoding: {layer[1:0], type[1:0]}
//   type 0: qkv      depth 3072
//   type 1: proj     depth 1024
//   type 2: ff_up    depth 4096
//   type 3: ff_down  depth 4096
//
// tok_emb has a dedicated port shared between embedding and head_proj.

module weight_store_w16 (
  input  wire         clk_i,
  input  wire [3:0]   w16_sel_i,
  input  wire [15:0]  w16_addr_i,
  output reg  [127:0] data_o,
  output reg  [15:0]  scale_o,

  input  wire [10:0]  tok_emb_addr_i,
  output wire [127:0] tok_emb_data_o,
  output reg  [15:0]  tok_emb_scale_o
);

  `include "weight_scales.vh"

  reg [15:0] addr_r;
  reg [3:0]  sel_r1;
  reg [10:0] tok_emb_addr_r;
  always @(posedge clk_i) begin
    addr_r         <= w16_addr_i;
    sel_r1         <= w16_sel_i;
    tok_emb_addr_r <= tok_emb_addr_i;
  end

  wire [127:0] d_b0_qkv,  d_b0_proj,  d_b0_ff_up,  d_b0_ff_down;
  wire [127:0] d_b1_qkv,  d_b1_proj,  d_b1_ff_up,  d_b1_ff_down;
  wire [127:0] d_b2_qkv,  d_b2_proj,  d_b2_ff_up,  d_b2_ff_down;
  wire [127:0] d_b3_qkv,  d_b3_proj,  d_b3_ff_up,  d_b3_ff_down;

  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_attn_qkv_weight.hex")
  ) u_b0_qkv (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b0_qkv)
  );

  weight_rom #(
    .DEPTH(1024), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_attn_proj_weight.hex")
  ) u_b0_proj (
    .clk_i (clk_i),
    .addr_i(addr_r[9:0]),
    .data_o(d_b0_proj)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_ff_up_weight.hex")
  ) u_b0_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b0_ff_up)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_ff_down_weight.hex")
  ) u_b0_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b0_ff_down)
  );

  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_attn_qkv_weight.hex")
  ) u_b1_qkv (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b1_qkv)
  );

  weight_rom #(
    .DEPTH(1024), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_attn_proj_weight.hex")
  ) u_b1_proj (
    .clk_i (clk_i),
    .addr_i(addr_r[9:0]),
    .data_o(d_b1_proj)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_ff_up_weight.hex")
  ) u_b1_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b1_ff_up)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_ff_down_weight.hex")
  ) u_b1_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b1_ff_down)
  );

  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_attn_qkv_weight.hex")
  ) u_b2_qkv (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b2_qkv)
  );

  weight_rom #(
    .DEPTH(1024), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_attn_proj_weight.hex")
  ) u_b2_proj (
    .clk_i (clk_i),
    .addr_i(addr_r[9:0]),
    .data_o(d_b2_proj)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_ff_up_weight.hex")
  ) u_b2_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b2_ff_up)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_ff_down_weight.hex")
  ) u_b2_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b2_ff_down)
  );

  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_attn_qkv_weight.hex")
  ) u_b3_qkv (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b3_qkv)
  );

  weight_rom #(
    .DEPTH(1024), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_attn_proj_weight.hex")
  ) u_b3_proj (
    .clk_i (clk_i),
    .addr_i(addr_r[9:0]),
    .data_o(d_b3_proj)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_ff_up_weight.hex")
  ) u_b3_ff_up (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b3_ff_up)
  );

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_ff_down_weight.hex")
  ) u_b3_ff_down (
    .clk_i (clk_i),
    .addr_i(addr_r[11:0]),
    .data_o(d_b3_ff_down)
  );

  weight_rom #(
    .DEPTH(2048), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/tok_emb_weight.hex")
  ) u_tok_emb (
    .clk_i (clk_i),
    .addr_i(tok_emb_addr_r),
    .data_o(tok_emb_data_o)
  );

  always @(posedge clk_i) tok_emb_scale_o <= SCALE_TOK_EMB_WEIGHT;

  reg [3:0] sel_r;
  always @(posedge clk_i) sel_r <= sel_r1;

  always @(*) begin
    case (sel_r)
      4'b0000: data_o = d_b0_qkv;
      4'b0001: data_o = d_b0_proj;
      4'b0010: data_o = d_b0_ff_up;
      4'b0011: data_o = d_b0_ff_down;
      4'b0100: data_o = d_b1_qkv;
      4'b0101: data_o = d_b1_proj;
      4'b0110: data_o = d_b1_ff_up;
      4'b0111: data_o = d_b1_ff_down;
      4'b1000: data_o = d_b2_qkv;
      4'b1001: data_o = d_b2_proj;
      4'b1010: data_o = d_b2_ff_up;
      4'b1011: data_o = d_b2_ff_down;
      4'b1100: data_o = d_b3_qkv;
      4'b1101: data_o = d_b3_proj;
      4'b1110: data_o = d_b3_ff_up;
      4'b1111: data_o = d_b3_ff_down;
      default: data_o = 128'd0;
    endcase
  end

  always @(posedge clk_i) begin
    case (sel_r1)
      4'b0000: scale_o <= SCALE_BLOCK0_ATTN_QKV_WEIGHT;
      4'b0001: scale_o <= SCALE_BLOCK0_ATTN_PROJ_WEIGHT;
      4'b0010: scale_o <= SCALE_BLOCK0_FF_UP_WEIGHT;
      4'b0011: scale_o <= SCALE_BLOCK0_FF_DOWN_WEIGHT;
      4'b0100: scale_o <= SCALE_BLOCK1_ATTN_QKV_WEIGHT;
      4'b0101: scale_o <= SCALE_BLOCK1_ATTN_PROJ_WEIGHT;
      4'b0110: scale_o <= SCALE_BLOCK1_FF_UP_WEIGHT;
      4'b0111: scale_o <= SCALE_BLOCK1_FF_DOWN_WEIGHT;
      4'b1000: scale_o <= SCALE_BLOCK2_ATTN_QKV_WEIGHT;
      4'b1001: scale_o <= SCALE_BLOCK2_ATTN_PROJ_WEIGHT;
      4'b1010: scale_o <= SCALE_BLOCK2_FF_UP_WEIGHT;
      4'b1011: scale_o <= SCALE_BLOCK2_FF_DOWN_WEIGHT;
      4'b1100: scale_o <= SCALE_BLOCK3_ATTN_QKV_WEIGHT;
      4'b1101: scale_o <= SCALE_BLOCK3_ATTN_PROJ_WEIGHT;
      4'b1110: scale_o <= SCALE_BLOCK3_FF_UP_WEIGHT;
      4'b1111: scale_o <= SCALE_BLOCK3_FF_DOWN_WEIGHT;
      default: scale_o <= 16'd0;
    endcase
  end

endmodule