// tok_emb weight bank: single 128-bit packed BRAM, 256 vocab x 8 dims/word
// = 2048 entries. Shared between embedding read path and head_proj matvec.

module weight_store_tok_emb (
  input  wire        clk_i,
  input  wire [10:0] addr_i,
  output wire [127:0] data_o,
  output reg  [15:0]  scale_o
);

  `include "weight_scales.vh"

  reg [10:0] addr_r;
  always @(posedge clk_i) addr_r <= addr_i;

  weight_rom #(
    .DEPTH(2048), .DATA_W(128),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/tok_emb_weight.hex")
  ) u_tok_emb (
    .clk_i (clk_i),
    .addr_i(addr_r),
    .data_o(data_o)
  );

  always @(posedge clk_i) scale_o <= SCALE_TOK_EMB_WEIGHT;

endmodule
