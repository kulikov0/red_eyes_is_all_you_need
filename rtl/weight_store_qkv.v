// QKV weight bank: 4 BRAMs holding all 4 layers' attention QKV weights.
// addr_i is shared across the 4 BRAMs, layer_i picks which layer's data
// and scale appears at the outputs. Designed to be co-located with
// attention's u_qkv matvec in a single pblock so the addr route stays
// short.

module weight_store_qkv (
  input  wire        clk_i,
  input  wire [1:0]  layer_i,
  input  wire [11:0] addr_i,
  output reg  [127:0] data_o,
  (* MAX_FANOUT = "16" *) output reg [15:0] scale_o
);

  `include "weight_scales.vh"

  reg [11:0] addr_r;
  // keeps each bank's layer pipeline distinct so Vivado does not
  // merge layer_r across banks and route a single replica into other banks
  (* DONT_TOUCH = "true" *) reg [1:0]  layer_r1;
  (* DONT_TOUCH = "true" *) reg [1:0]  layer_r;
  always @(posedge clk_i) begin
    addr_r   <= addr_i;
    layer_r1 <= layer_i;
    layer_r  <= layer_r1;
  end

  wire [127:0] d0, d1, d2, d3;

  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_attn_qkv_weight.hex")
  ) u_b0 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d0));
  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_attn_qkv_weight.hex")
  ) u_b1 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d1));
  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_attn_qkv_weight.hex")
  ) u_b2 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d2));
  weight_rom #(
    .DEPTH(3072), .DATA_W(128),
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
