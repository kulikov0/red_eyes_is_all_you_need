// QKV weight bank: 4 BRAMs holding all 4 layers' attention QKV weights.
// 256-bit packed words pair col and col+IN/2 for the same 16 rows so that a
// single BRAM read serves matvec_fp16_w32's 32-weight-per-cycle throughput.

module weight_store_qkv (
  input  wire        clk_i,
  input  wire [1:0]  layer_i,
  input  wire [10:0] addr_i,
  output reg  [255:0] data_o,
  (* MAX_FANOUT = "32" *) output reg [15:0] scale_o
);

  `include "weight_scales.vh"

  reg [10:0] addr_r;
  (* DONT_TOUCH = "true" *) reg [1:0]  layer_r1;
  (* DONT_TOUCH = "true" *) reg [1:0]  layer_r;
  always @(posedge clk_i) begin
    addr_r   <= addr_i;
    layer_r1 <= layer_i;
    layer_r  <= layer_r1;
  end

  wire [255:0] d0, d1, d2, d3;

  weight_rom #(
    .DEPTH(1536), .DATA_W(256),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_attn_qkv_weight.hex")
  ) u_b0 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d0));
  weight_rom #(
    .DEPTH(1536), .DATA_W(256),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_attn_qkv_weight.hex")
  ) u_b1 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d1));
  weight_rom #(
    .DEPTH(1536), .DATA_W(256),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_attn_qkv_weight.hex")
  ) u_b2 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d2));
  weight_rom #(
    .DEPTH(1536), .DATA_W(256),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_attn_qkv_weight.hex")
  ) u_b3 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d3));

  always @(*) begin
    case (layer_r)
      2'd0:    data_o = d0;
      2'd1:    data_o = d1;
      2'd2:    data_o = d2;
      default: data_o = d3;
    endcase
  end

  always @(posedge clk_i) begin
    case (layer_r1)
      2'd0:    scale_o <= SCALE_BLOCK0_ATTN_QKV_WEIGHT;
      2'd1:    scale_o <= SCALE_BLOCK1_ATTN_QKV_WEIGHT;
      2'd2:    scale_o <= SCALE_BLOCK2_ATTN_QKV_WEIGHT;
      default: scale_o <= SCALE_BLOCK3_ATTN_QKV_WEIGHT;
    endcase
  end

endmodule
