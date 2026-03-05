// Convert IEEE 754 fp16 to signed int8 with rounding and clamping, combinational
//
// Latency: 0 (combinational)
// Rounds to nearest, ties to even (RNE)
// Clamps to [-128, 127]
// NaN/inf -> clamp to +127/-128 based on sign
// Denormals (exp==0) -> 0
// |val| < 0.5 -> 0

module fp16_to_int8 (
  input  wire [15:0] val_i,
  output wire [7:0]  int8_o
);

  wire        sign = val_i[15];
  wire [4:0]  exp  = val_i[14:10];
  wire [9:0]  mant = val_i[9:0];

  wire is_zero    = (exp == 5'd0);
  wire is_inf     = (exp == 5'd31) && (mant == 10'd0);
  wire is_nan     = (exp == 5'd31) && (mant != 10'd0);
  wire is_special = is_inf || is_nan;
  wire too_small  = (exp < 5'd14); // |val| < 0.5

  wire [10:0] full_mant = {1'b1, mant};

  // Extract integer magnitude, round bit, and sticky bits via case on exponent
  // e = exp - 15 (unbiased). Integer = full_mant >> (10 - e)
  // Round bit = MSB of fractional part
  // Sticky = OR of all fractional bits below round bit
  reg [7:0] mag;
  reg       round_bit;
  reg       sticky;
  always @(*) begin
    mag = 8'd0;
    round_bit = 1'b0;
    sticky = 1'b0;
    case (exp)
      5'd14: begin // e=-1, val in [0.5, 1.0)
        mag = 8'd0;
        round_bit = 1'b1;  // implicit 1 -> fractional MSB is 1
        sticky = |mant;    // any mantissa bits below implicit 1 -> not exactly 0.5
      end
      5'd15: begin // e=0, val in [1.0, 2.0)
        mag = {7'd0, full_mant[10]};
        round_bit = full_mant[9];
        sticky = |full_mant[8:0];
      end
      5'd16: begin // e=1
        mag = {6'd0, full_mant[10:9]};
        round_bit = full_mant[8];
        sticky = |full_mant[7:0];
      end
      5'd17: begin // e=2
        mag = {5'd0, full_mant[10:8]};
        round_bit = full_mant[7];
        sticky = |full_mant[6:0];
      end
      5'd18: begin // e=3
        mag = {4'd0, full_mant[10:7]};
        round_bit = full_mant[6];
        sticky = |full_mant[5:0];
      end
      5'd19: begin // e=4
        mag = {3'd0, full_mant[10:6]};
        round_bit = full_mant[5];
        sticky = |full_mant[4:0];
      end
      5'd20: begin // e=5
        mag = {2'd0, full_mant[10:5]};
        round_bit = full_mant[4];
        sticky = |full_mant[3:0];
      end
      5'd21: begin // e=6
        mag = {1'd0, full_mant[10:4]};
        round_bit = full_mant[3];
        sticky = |full_mant[2:0];
      end
      5'd22: begin // e=7
        mag = full_mant[10:3];
        round_bit = full_mant[2];
        sticky = |full_mant[1:0];
      end
      default: begin // e>=8: will clamp
        mag = 8'hFF;
        round_bit = 1'b0;
        sticky = 1'b0;
      end
    endcase
  end

  // RNE: round up if (round_bit=1) AND (sticky=1 OR mag[0]=1)
  // This is: round up if above midpoint, or at midpoint and result would be odd
  wire round_up = round_bit & (sticky | mag[0]);
  wire [8:0] rounded = {1'b0, mag} + {8'd0, round_up};

  // Clamp magnitude: positive max 127, negative max 128
  wire [8:0] max_mag = sign ? 9'd128 : 9'd127;
  wire       needs_clamp = (rounded > max_mag);
  wire [7:0] clamped_mag = needs_clamp ? max_mag[7:0] : rounded[7:0];

  // Apply sign (two's complement)
  wire [7:0] signed_val = sign ? (~clamped_mag + 8'd1) : clamped_mag;

  assign int8_o = is_special            ? (sign ? -8'sd128 : 8'sd127) :
                  (is_zero || too_small) ? 8'd0 :
                                           signed_val;

endmodule