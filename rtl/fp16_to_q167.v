// Convert IEEE 754 fp16 to signed Q16.7 fixed-point (24-bit), combinational
//
// Latency: 0 (combinational)
// Q16.7 = round(value * 128), clamped to signed 24-bit [-8388608, 8388607]
// Full fp16 range fits: max 65504 * 128 = 8384512 < 8388607
// RNE rounding for right shifts
// Denormals (exp==0) -> 0
// NaN/inf -> +8388607/-8388608 based on sign

module fp16_to_q167 (
  input  wire [15:0] val_i,
  output wire [23:0] q167_o  // signed Q16.7
);

  wire        sign = val_i[15];
  wire [4:0]  exp  = val_i[14:10];
  wire [9:0]  mant = val_i[9:0];
  wire [10:0] full_mant = {1'b1, mant};

  wire is_zero    = (exp == 5'd0);
  wire is_special = (exp == 5'd31);

  // Q16.7 = round(full_mant * 2^(exp-18))
  // exp < 18: right shift by (18-exp), with RNE rounding
  // exp >= 18: left shift by (exp-18)
  reg [23:0] mag;
  reg        round_bit;
  reg        sticky;
  always @(*) begin
    mag = 24'd0;
    round_bit = 1'b0;
    sticky = 1'b0;
    case (exp)
      // exp 1..6: shift >= 12, full_mant (11 bits) >> 12+ = 0
      5'd1, 5'd2, 5'd3, 5'd4, 5'd5, 5'd6: begin
        mag = 24'd0;
      end
      5'd7: begin // shift=11
        mag = 24'd0;
        round_bit = 1'b1;
        sticky = |mant;
      end
      5'd8: begin // shift=10
        mag = {23'd0, full_mant[10]};
        round_bit = full_mant[9];
        sticky = |full_mant[8:0];
      end
      5'd9: begin // shift=9
        mag = {22'd0, full_mant[10:9]};
        round_bit = full_mant[8];
        sticky = |full_mant[7:0];
      end
      5'd10: begin // shift=8
        mag = {21'd0, full_mant[10:8]};
        round_bit = full_mant[7];
        sticky = |full_mant[6:0];
      end
      5'd11: begin // shift=7
        mag = {20'd0, full_mant[10:7]};
        round_bit = full_mant[6];
        sticky = |full_mant[5:0];
      end
      5'd12: begin // shift=6
        mag = {19'd0, full_mant[10:6]};
        round_bit = full_mant[5];
        sticky = |full_mant[4:0];
      end
      5'd13: begin // shift=5
        mag = {18'd0, full_mant[10:5]};
        round_bit = full_mant[4];
        sticky = |full_mant[3:0];
      end
      5'd14: begin // shift=4
        mag = {17'd0, full_mant[10:4]};
        round_bit = full_mant[3];
        sticky = |full_mant[2:0];
      end
      5'd15: begin // shift=3
        mag = {16'd0, full_mant[10:3]};
        round_bit = full_mant[2];
        sticky = |full_mant[1:0];
      end
      5'd16: begin // shift=2
        mag = {15'd0, full_mant[10:2]};
        round_bit = full_mant[1];
        sticky = full_mant[0];
      end
      5'd17: begin // shift=1
        mag = {14'd0, full_mant[10:1]};
        round_bit = full_mant[0];
      end
      5'd18: begin // shift=0, exact
        mag = {13'd0, full_mant};
      end
      5'd19: mag = {12'd0, full_mant, 1'b0};
      5'd20: mag = {11'd0, full_mant, 2'b00};
      5'd21: mag = {10'd0, full_mant, 3'b000};
      5'd22: mag = {9'd0, full_mant, 4'b0000};
      5'd23: mag = {8'd0, full_mant, 5'b00000};
      5'd24: mag = {7'd0, full_mant, 6'b000000};
      5'd25: mag = {6'd0, full_mant, 7'b0000000};
      5'd26: mag = {5'd0, full_mant, 8'b00000000};
      5'd27: mag = {4'd0, full_mant, 9'b000000000};
      5'd28: mag = {3'd0, full_mant, 10'b0000000000};
      5'd29: mag = {2'd0, full_mant, 11'b00000000000};
      5'd30: mag = {1'd0, full_mant, 12'b000000000000};
      default: mag = 24'h7FFFFF;
    endcase
  end

  // RNE: round up if guard=1 AND (sticky=1 OR mag[0]=1)
  wire round_up = round_bit & (sticky | mag[0]);
  wire [24:0] rounded = {1'b0, mag} + {24'd0, round_up};
  wire        overflow = rounded[24];
  wire [23:0] clamped  = overflow ? 24'h7FFFFF : rounded[23:0];

  // Two's complement for negative
  wire [23:0] signed_val = sign ? (~clamped + 24'd1) : clamped;

  assign q167_o = is_special ? (sign ? 24'h800000 : 24'h7FFFFF) :
                  is_zero    ? 24'd0 :
                               signed_val;

endmodule