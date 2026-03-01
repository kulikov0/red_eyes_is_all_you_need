// Requantize: rescale wide accumulator to int8 via scale multiply + shift
//
// Combinational (zero latency). Computes:
//   product = acc_i * unsigned(scale_i)   [signed x unsigned]
//   shifted = product >>> SHIFT
//   q_o     = clamp(shifted, -128, 127)
//
// Scale encoding: requant factor M = scale_i / 2^SHIFT
//   Shift-only: scale_i = 2^(SCALE_W-1), SHIFT = N + SCALE_W - 1
//   With scale: scale_i = round(M * 2^SHIFT)

module requant #(
  parameter ACC_W   = 24,
  parameter SCALE_W = 16,
  parameter SHIFT   = 22
) (
  input  wire signed [ACC_W-1:0]   acc_i,
  input  wire        [SCALE_W-1:0] scale_i,
  output wire signed [7:0]         q_o
);

  localparam PROD_W = ACC_W + SCALE_W;
  localparam signed [PROD_W-1:0] MAX_VAL =  127;
  localparam signed [PROD_W-1:0] MIN_VAL = -128;

  wire signed [PROD_W-1:0] product = acc_i * $signed({1'b0, scale_i});
  wire signed [PROD_W-1:0] shifted = product >>> SHIFT;

  assign q_o = (shifted > MAX_VAL) ?  8'sd127 :
               (shifted < MIN_VAL) ? -8'sd128 :
               shifted[7:0];

endmodule