// FF_down (512 -> 128) weight bank. Co-locates with transformer_layer's
// u_ff_down matvec.

module weight_store_ff_down (
  input  wire        clk_i,
  input  wire [1:0]  layer_i,
  input  wire [11:0] addr_i,
  output reg  [127:0] data_o,
  (* MAX_FANOUT = "16" *) output reg [15:0] scale_o
);

  `include "weight_scales.vh"

  reg [11:0] addr_r;
  reg [1:0]  layer_r1;
  reg [1:0]  layer_r;
  always @(posedge clk_i) begin
    addr_r   <= addr_i;
    layer_r1 <= layer_i;
    layer_r  <= layer_r1;
  end

  wire [127:0] d0, d1, d2, d3;

  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block0_ff_down_weight.hex")
  ) u_b0 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d0));
  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block1_ff_down_weight.hex")
  ) u_b1 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d1));
  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block2_ff_down_weight.hex")
  ) u_b2 (.clk_i(clk_i), .addr_i(addr_r), .data_o(d2));
  weight_rom #(
    .DEPTH(4096), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/block3_ff_down_weight.hex")
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
      2'd0:    scale_o <= SCALE_BLOCK0_FF_DOWN_WEIGHT;
      2'd1:    scale_o <= SCALE_BLOCK1_FF_DOWN_WEIGHT;
      2'd2:    scale_o <= SCALE_BLOCK2_FF_DOWN_WEIGHT;
      default: scale_o <= SCALE_BLOCK3_FF_DOWN_WEIGHT;
    endcase
  end

endmodule
