// FP16 adder, 1-cycle registered output
//
// IEEE 754 half-precision: sign(1) | exp(5) | mant(10)
// Flush-to-zero for denormals (exp==0 treated as zero)
// inf + inf(same sign) = inf, inf + inf(diff sign) = NaN
// Any NaN input -> NaN output
// Latency: 1 clock cycle

module fp16_add (
  input  wire        clk_i,
  input  wire [15:0] a_i,
  input  wire [15:0] b_i,
  output reg  [15:0] sum_o
);

  // Unpack
  wire        a_sign = a_i[15];
  wire [4:0]  a_exp  = a_i[14:10];
  wire [9:0]  a_mant = a_i[9:0];

  wire        b_sign = b_i[15];
  wire [4:0]  b_exp  = b_i[14:10];
  wire [9:0]  b_mant = b_i[9:0];

  // Special value detection
  wire a_is_zero = (a_exp == 5'd0);
  wire b_is_zero = (b_exp == 5'd0);
  wire a_is_inf  = (a_exp == 5'd31) && (a_mant == 10'd0);
  wire b_is_inf  = (b_exp == 5'd31) && (b_mant == 10'd0);
  wire a_is_nan  = (a_exp == 5'd31) && (a_mant != 10'd0);
  wire b_is_nan  = (b_exp == 5'd31) && (b_mant != 10'd0);

  // Implied 1.mant for normals, 0 for zero/denorm
  wire [10:0] a_full = a_is_zero ? 11'd0 : {1'b1, a_mant};
  wire [10:0] b_full = b_is_zero ? 11'd0 : {1'b1, b_mant};

  // Swap so |A| >= |B| (larger exponent first, tie-break on mantissa)
  wire a_ge_b = (a_exp > b_exp) || ((a_exp == b_exp) && (a_full >= b_full));

  wire        lg_sign = a_ge_b ? a_sign : b_sign;
  wire [4:0]  lg_exp  = a_ge_b ? a_exp  : b_exp;
  wire [10:0] lg_mant = a_ge_b ? a_full : b_full;
  wire        sm_sign = a_ge_b ? b_sign : a_sign;
  wire [4:0]  sm_exp  = a_ge_b ? b_exp  : a_exp;
  wire [10:0] sm_mant = a_ge_b ? b_full : a_full;

  // Alignment: both operands in 14-bit format {0, 1.mmmmmmmmmm, G, R}
  // Extra trailing bits in sm for sticky calculation
  wire [4:0] exp_diff = lg_exp - sm_exp;

  // lg: 14 bits = {0, mant[10:0], guard=0, round=0}
  wire [13:0] lg_ext = {1'b0, lg_mant, 2'b00};

  // sm: start with same 14-bit format, extend with 13 trailing zeros for shift
  // Total 27 bits. After right-shift, top 14 = aligned value, bottom 13 = sticky source
  wire [26:0] sm_wide = {1'b0, sm_mant, 2'b00, 13'b0};
  wire [26:0] sm_shifted = sm_wide >> exp_diff;
  wire [13:0] sm_ext = sm_shifted[26:13];
  wire        sticky  = |sm_shifted[12:0];

  // Add or subtract based on effective operation
  wire eff_sub = lg_sign ^ sm_sign;
  wire [14:0] mant_sum = eff_sub ? ({1'b0, lg_ext} - {1'b0, sm_ext}) :
                                   ({1'b0, lg_ext} + {1'b0, sm_ext});
  // For subtraction, result is always positive since |lg| >= |sm|
  // Bit 14 is overflow bit (addition only)

  // Normalization: find leading one position in mant_sum[14:0]
  // Expected bit positions after add:
  //   bit 14: never set for subtraction, possible for addition overflow (>= 2.0)
  //   bit 13: overflow (addition): 1x.mmmmmmmmmm G R
  //   bit 12: normal:              01.mmmmmmmmmm G R
  //   bits < 12: cancellation (subtraction)
  reg [3:0] lod;
  reg       sum_is_zero;
  integer i;
  always @(*) begin
    lod = 4'd0;
    sum_is_zero = (mant_sum[14:0] == 15'd0);
    for (i = 0; i < 15; i = i + 1) begin
      if (mant_sum[i]) lod = i[3:0];
    end
  end

  // Target: leading 1 at bit 12 -> {XX, 1, mant[9:0], G, R}
  wire overflow = (lod == 4'd13) || (lod == 4'd14);

  // Shift to normalize: if lod > 12, shift right (lod-12); if lod < 12, shift left (12-lod)
  wire [3:0] rshift_amt = (lod > 4'd12) ? (lod - 4'd12) : 4'd0;
  wire [3:0] lshift_amt = (lod < 4'd12) ? (4'd12 - lod) : 4'd0;

  wire [14:0] norm_mant = sum_is_zero ? 15'd0 :
                          overflow    ? (mant_sum >> rshift_amt) :
                                        (mant_sum << lshift_amt);

  // Exponent adjustment
  wire signed [6:0] lg_exp_s = $signed({2'b0, lg_exp});
  wire signed [6:0] rsh_s    = $signed({3'b0, rshift_amt});
  wire signed [6:0] lsh_s    = $signed({3'b0, lshift_amt});
  wire signed [6:0] exp_adj_s = sum_is_zero ? 7'sd0 :
                                overflow    ? (lg_exp_s + rsh_s) :
                                              (lg_exp_s - lsh_s);

  // RNE rounding: extract guard and round from normalized mantissa
  // norm_mant[12] = implicit 1, [11:2] = mantissa
  // [1] = guard, [0] = round
  wire [9:0] trunc_mant = norm_mant[11:2];
  wire       guard_bit  = norm_mant[1];
  wire       round_bit  = norm_mant[0];
  // For right-shifts during normalization (overflow), bits shifted out contribute to sticky
  wire       extra_sticky = overflow ? |mant_sum[0] : 1'b0;
  wire       sticky_bit   = sticky | extra_sticky;
  // Disable sticky for effective subtraction (sticky is from alignment, unreliable after borrow)
  wire       use_sticky = sticky_bit & ~eff_sub;
  wire       round_up = guard_bit & (round_bit | use_sticky | trunc_mant[0]);

  wire [10:0] rounded_mant = {1'b0, trunc_mant} + {10'd0, round_up};
  wire        round_ovf = rounded_mant[10];
  wire signed [6:0] final_exp_s = round_ovf ? (exp_adj_s + 7'sd1) : exp_adj_s;

  // Pack result
  wire [15:0] normal_result = {lg_sign, final_exp_s[4:0], rounded_mant[9:0]};

  // Overflow to infinity
  wire exp_overflow = (final_exp_s >= 7'sd31);
  wire [15:0] inf_result = {lg_sign, 5'd31, 10'd0};

  // Underflow to zero (exponent <= 0)
  wire exp_underflow = (final_exp_s <= 7'sd0) && !sum_is_zero;
  wire [15:0] zero_result = {lg_sign, 15'd0};

  wire [15:0] nan_result = 16'h7E00;

  reg [15:0] result;
  always @(*) begin
    if (a_is_nan || b_is_nan) begin
      result = nan_result;
    end else if (a_is_inf && b_is_inf && eff_sub) begin
      result = nan_result;
    end else if (a_is_inf || b_is_inf) begin
      result = a_is_inf ? {a_sign, 5'd31, 10'd0} : {b_sign, 5'd31, 10'd0};
    end else if (a_is_zero && b_is_zero) begin
      result = {a_sign & b_sign, 15'd0};
    end else if (a_is_zero) begin
      result = b_i;
    end else if (b_is_zero) begin
      result = a_i;
    end else if (sum_is_zero) begin
      result = 16'd0;
    end else if (exp_underflow) begin
      result = zero_result;
    end else if (exp_overflow) begin
      result = inf_result;
    end else begin
      result = normal_result;
    end
  end

  always @(posedge clk_i) begin
    sum_o <= result;
  end

endmodule