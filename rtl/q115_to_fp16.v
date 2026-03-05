// Convert unsigned Q1.15 fixed-point to IEEE 754 fp16, combinational
//
// Latency: 0 (combinational)
// Input: 16-bit unsigned, representing value/32768 (range [0, 1.0])
// Output: fp16 bit pattern
// Denormals flushed to zero (val < 2^4 = 16 maps to exp=0..3, too small)
// RNE rounding when mantissa bits must be discarded (val >= 2048)

module q115_to_fp16 (
  input  wire [15:0] val_i,
  output wire [15:0] fp16_o
);

  wire is_zero = (val_i == 16'd0);

  // LOD: find leading one position (0..15)
  reg [3:0] lod;
  integer i;
  always @(*) begin
    lod = 4'd0;
    for (i = 0; i < 16; i = i + 1) begin
      if (val_i[i]) lod = i[3:0];
    end
  end

  // fp16 exponent = lod (since val/32768 = val*2^-15, MSB at lod -> 2^(lod-15))
  wire [4:0] exp_raw = {1'b0, lod};

  // Mantissa extraction: bits below leading one, left-aligned to 10 bits
  // For lod <= 10: all bits fit, shift left
  // For lod >= 11: discard LSBs, with RNE rounding
  reg [9:0] frac;
  reg       guard, rnd_bit, stk;
  always @(*) begin
    frac = 10'd0;
    guard = 1'b0;
    rnd_bit = 1'b0;
    stk = 1'b0;
    case (lod)
      4'd0: frac = 10'd0;
      4'd1: frac = {val_i[0], 9'd0};
      4'd2: frac = {val_i[1:0], 8'd0};
      4'd3: frac = {val_i[2:0], 7'd0};
      4'd4: frac = {val_i[3:0], 6'd0};
      4'd5: frac = {val_i[4:0], 5'd0};
      4'd6: frac = {val_i[5:0], 4'd0};
      4'd7: frac = {val_i[6:0], 3'd0};
      4'd8: frac = {val_i[7:0], 2'd0};
      4'd9: frac = {val_i[8:0], 1'd0};
      4'd10: frac = val_i[9:0];
      4'd11: begin // discard 1 bit
        frac = val_i[10:1];
        guard = val_i[0];
      end
      4'd12: begin // discard 2 bits
        frac = val_i[11:2];
        guard = val_i[1];
        rnd_bit = val_i[0];
      end
      4'd13: begin // discard 3 bits
        frac = val_i[12:3];
        guard = val_i[2];
        rnd_bit = val_i[1];
        stk = val_i[0];
      end
      4'd14: begin // discard 4 bits
        frac = val_i[13:4];
        guard = val_i[3];
        rnd_bit = val_i[2];
        stk = |val_i[1:0];
      end
      4'd15: begin // discard 5 bits
        frac = val_i[14:5];
        guard = val_i[4];
        rnd_bit = val_i[3];
        stk = |val_i[2:0];
      end
    endcase
  end

  // RNE rounding
  wire stk_all = rnd_bit | stk;
  wire round_up = guard & (stk_all | frac[0]);
  wire [10:0] rounded = {1'b0, frac} + {10'd0, round_up};
  wire        round_ovf = rounded[10];
  wire [4:0]  final_exp = round_ovf ? (exp_raw + 5'd1) : exp_raw;
  wire [9:0]  final_frac = round_ovf ? 10'd0 : rounded[9:0];

  assign fp16_o = is_zero ? 16'd0 : {1'b0, final_exp, final_frac};

endmodule