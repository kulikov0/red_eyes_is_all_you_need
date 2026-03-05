// FP16 multiplier, 1-cycle registered output
//
// IEEE 754 half-precision: sign(1) | exp(5) | mant(10)
// Flush-to-zero for denormals (exp==0 treated as zero)
// inf * 0 = NaN, inf * x = inf, NaN -> NaN
// 11x11 mantissa multiply fits in 1 DSP48E1 (25x18)
// Latency: 1 clock cycle

module fp16_mul (
  input  wire        clk_i,
  input  wire [15:0] a_i,
  input  wire [15:0] b_i,
  output reg  [15:0] prod_o
);

  // Unpack
  wire        a_sign = a_i[15];
  wire [4:0]  a_exp  = a_i[14:10];
  wire [9:0]  a_mant = a_i[9:0];

  wire        b_sign = b_i[15];
  wire [4:0]  b_exp  = b_i[14:10];
  wire [9:0]  b_mant = b_i[9:0];

  wire res_sign = a_sign ^ b_sign;

  // Special value detection
  wire a_is_zero = (a_exp == 5'd0);
  wire b_is_zero = (b_exp == 5'd0);
  wire a_is_inf  = (a_exp == 5'd31) && (a_mant == 10'd0);
  wire b_is_inf  = (b_exp == 5'd31) && (b_mant == 10'd0);
  wire a_is_nan  = (a_exp == 5'd31) && (a_mant != 10'd0);
  wire b_is_nan  = (b_exp == 5'd31) && (b_mant != 10'd0);

  // Implied 1.mant for normals
  wire [10:0] a_full = {1'b1, a_mant};
  wire [10:0] b_full = {1'b1, b_mant};

  // 11x11 unsigned multiply -> 22-bit product
  // 1.mmmmmmmmmm x 1.mmmmmmmmmm = XX.mmmmmmmmmmmmmmmmmmm
  // Product range: [1.0, 4.0), bit 21 may be set
  wire [21:0] mant_prod = a_full * b_full;

  // Exponent sum (biased): ea + eb - 15
  // 8-bit signed to detect over/underflow
  wire signed [7:0] exp_raw = $signed({1'b0, a_exp}) + $signed({1'b0, b_exp}) - 8'sd15;

  // Normalization: product is in [1.0, 4.0)
  // If bit 21 is set -> 1x.xxx, shift right 1, exp + 1
  // Else bit 20 is set -> 1.xxx, no shift
  wire norm_shift = mant_prod[21];

  // Extract mantissa bits for rounding
  // After normalization, format is 1.mmmmmmmmmm (10 mantissa bits needed)
  // If norm_shift: product[21:11] = 1.mmmmmmmmmm
  //   guard=product[10], round=product[9], sticky=|product[8:0]
  // Else: product[20:10] = 1.mmmmmmmmmm
  //   guard=product[9], round=product[8], sticky=|product[7:0]

  wire [9:0] trunc_mant = norm_shift ? mant_prod[20:11] : mant_prod[19:10];
  wire       guard_bit  = norm_shift ? mant_prod[10]    : mant_prod[9];
  wire       round_bit  = norm_shift ? mant_prod[9]     : mant_prod[8];
  wire       sticky_bit = norm_shift ? |mant_prod[8:0]  : |mant_prod[7:0];

  // RNE rounding
  wire round_up = guard_bit & (round_bit | sticky_bit | trunc_mant[0]);
  wire [10:0] rounded = {1'b0, trunc_mant} + {10'd0, round_up};
  wire round_ovf = rounded[10]; // rounding caused mantissa overflow

  // Final exponent
  wire signed [7:0] final_exp = exp_raw + {7'd0, norm_shift} + {7'd0, round_ovf};

  // Pack result
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

  always @(posedge clk_i) begin
    prod_o <= result;
  end

endmodule