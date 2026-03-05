// Convert signed int8 to IEEE 754 fp16, combinational
//
// Latency: 0 (combinational)
// int8 range: [-128, 127]
// fp16 can represent all int8 values exactly (mantissa has 10 bits, only need 7)
// Special case: 0 -> +0.0 (0x0000)
// -128 -> 0xD800 (sign=1, exp=22, mant=0 -> -1.0 * 2^7 = -128)

module fp16_from_int8 (
  input  wire [7:0]  val_i,
  output wire [15:0] fp16_o
);

  wire       is_neg = val_i[7];
  wire [7:0] abs_val = is_neg ? (~val_i + 8'd1) : val_i;
  wire       is_zero = (val_i == 8'd0);

  // Leading one detector for abs_val [7:0]
  // abs_val range: [0, 128], so up to bit 7
  reg [3:0] lod;
  integer i;
  always @(*) begin
    lod = 4'd0;
    for (i = 0; i < 8; i = i + 1) begin
      if (abs_val[i]) lod = i[3:0];
    end
  end

  // Exponent: lod + bias(15)
  wire [4:0] exp_val = lod[3:0] + 5'd15;

  // Mantissa: remove implicit 1, left-align remaining bits into 10-bit field
  // abs_val has leading 1 at position lod, bits below are lod bits wide
  // Shift left to place bit (lod-1) at mantissa MSB (bit 9)
  // shift_amt = 10 - lod (since we skip the implicit 1)
  wire [3:0] mant_shift = 4'd10 - lod;
  wire [9:0] mant_val = (abs_val << mant_shift) & 10'h3FF; // mask off implicit 1

  assign fp16_o = is_zero ? 16'd0 : {is_neg, exp_val, mant_val};

endmodule