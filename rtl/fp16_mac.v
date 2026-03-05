// FP16 multiply-accumulate: acc_o = acc_i + a_i * b_i
//
// Combinational multiply + registered add for 1-cycle accumulator feedback
// Pipeline: mul is combinational, add is registered
// Latency: 1 cycle (result available next clock edge)
// Throughput: 1 MAC per cycle
// Resources: ~1 DSP48 + ~600 LUTs + ~100 FFs

module fp16_mac (
  input  wire        clk_i,
  input  wire [15:0] a_i,
  input  wire [15:0] b_i,
  input  wire [15:0] acc_i,
  output wire [15:0] acc_o
);

  // Combinational multiply
  wire [15:0] prod;
  fp16_mul_comb u_mul (
    .a_i(a_i),
    .b_i(b_i),
    .prod_o(prod)
  );

  // Registered add (accumulate)
  fp16_add u_add (
    .clk_i(clk_i),
    .a_i(acc_i),
    .b_i(prod),
    .sum_o(acc_o)
  );

endmodule

// Combinational version of fp16 multiplier (no output register)
module fp16_mul_comb (
  input  wire [15:0] a_i,
  input  wire [15:0] b_i,
  output wire [15:0] prod_o
);

  wire        a_sign = a_i[15];
  wire [4:0]  a_exp  = a_i[14:10];
  wire [9:0]  a_mant = a_i[9:0];

  wire        b_sign = b_i[15];
  wire [4:0]  b_exp  = b_i[14:10];
  wire [9:0]  b_mant = b_i[9:0];

  wire res_sign = a_sign ^ b_sign;

  wire a_is_zero = (a_exp == 5'd0);
  wire b_is_zero = (b_exp == 5'd0);
  wire a_is_inf  = (a_exp == 5'd31) && (a_mant == 10'd0);
  wire b_is_inf  = (b_exp == 5'd31) && (b_mant == 10'd0);
  wire a_is_nan  = (a_exp == 5'd31) && (a_mant != 10'd0);
  wire b_is_nan  = (b_exp == 5'd31) && (b_mant != 10'd0);

  wire [10:0] a_full = {1'b1, a_mant};
  wire [10:0] b_full = {1'b1, b_mant};

  wire [21:0] mant_prod = a_full * b_full;

  wire signed [7:0] exp_raw = $signed({1'b0, a_exp}) + $signed({1'b0, b_exp}) - 8'sd15;

  wire norm_shift = mant_prod[21];

  wire [9:0] trunc_mant = norm_shift ? mant_prod[20:11] : mant_prod[19:10];
  wire       guard_bit  = norm_shift ? mant_prod[10]     : mant_prod[9];
  wire       round_bit  = norm_shift ? mant_prod[9]      : mant_prod[8];
  wire       sticky_bit = norm_shift ? |mant_prod[8:0]   : |mant_prod[7:0];

  wire round_up = guard_bit & (round_bit | sticky_bit | trunc_mant[0]);
  wire [10:0] rounded = {1'b0, trunc_mant} + {10'd0, round_up};
  wire round_ovf = rounded[10];

  wire signed [7:0] final_exp = exp_raw + {7'd0, norm_shift} + {7'd0, round_ovf};

  wire [15:0] normal_result = {res_sign, final_exp[4:0], rounded[9:0]};
  wire [15:0] inf_result    = {res_sign, 5'd31, 10'd0};
  wire [15:0] nan_result    = 16'h7E00;
  wire [15:0] zero_result   = {res_sign, 15'd0};

  wire exp_overflow  = (final_exp >= 8'sd31);
  wire exp_underflow = (final_exp <= 8'sd0);

  reg [15:0] result;
  always @(*) begin
    if (a_is_nan || b_is_nan) begin
      result = nan_result;
    end else if ((a_is_inf && b_is_zero) || (b_is_inf && a_is_zero)) begin
      result = nan_result;
    end else if (a_is_inf || b_is_inf) begin
      result = inf_result;
    end else if (a_is_zero || b_is_zero) begin
      result = zero_result;
    end else if (exp_underflow) begin
      result = zero_result;
    end else if (exp_overflow) begin
      result = inf_result;
    end else begin
      result = normal_result;
    end
  end
  assign prod_o = result;

endmodule