// Computes 1/sqrt(d) via LOD-LUT-Shift
//
// Based on https://www.mdpi.com/2072-666X/17/1/84
//
// Algorithm:
//   d = 2^k * (1 + m/256)
//   1/sqrt(d) = LUT[{k[0], m}] >> floor(k/2)
//
//   k = position of leading one (LOD)
//   m = 8-bit mantissa (bits below leading one, MSB-aligned)
//   LUT stores 1/sqrt(1+m/256) for even k, 1/sqrt(2*(1+m/256)) for odd k
//
// Output format: Q1.15 unsigned fixed-point (1 integer, 15 fractional bits)
//   result = round(1/sqrt(d) * 32768)
//
// Latency: 2 clock cycles
// Resources: 512x16 BRAM LUT + priority encoder + barrel shifter
// Max error: ~0.2% over full input range (D_W=14)

module inv_sqrt #(
  parameter D_W      = 14,
  parameter HEX_FILE = "/home/user/red_eyes_is_all_you_need/mem/inv_sqrt_lut.hex"
) (
  input  wire            clk_i,
  input  wire            valid_i,
  input  wire [D_W-1:0]  d_i,
  output reg             valid_o,
  output reg  [15:0]     result_o
);

  // Stage 0: LOD + mantissa extraction

  // Leading one detector: find MSB position (0 = LSB)
  reg [4:0] k;
  reg       d_is_zero;
  integer i;
  always @(*) begin
    k = 5'd0;
    d_is_zero = (d_i == {D_W{1'b0}});
    for (i = 0; i < D_W; i = i + 1) begin
      if (d_i[i]) k = i[4:0];
    end
  end

  // Normalize: shift leading one to bit D_W-1, then extract 8-bit mantissa
  wire [D_W-1:0] d_norm   = d_i << ((D_W - 1) - k);
  wire [7:0]     mantissa = d_norm[D_W-2 -: 8];

  // LUT address: {k_lsb, mantissa}
  wire [8:0] lut_addr = {k[0], mantissa};

  // Stage 1: BRAM LUT read + pipeline registers

  (* ram_style = "block" *) reg [15:0] lut_mem [0:511];
  initial $readmemh(HEX_FILE, lut_mem);

  reg [15:0] lut_out;
  always @(posedge clk_i) lut_out <= lut_mem[lut_addr];

  reg [4:0] k_r;
  reg       zero_r;
  reg       valid_r;
  always @(posedge clk_i) begin
    k_r     <= k;
    zero_r  <= d_is_zero;
    valid_r <= valid_i;
  end

  // Stage 2: Barrel shift + output

  wire [3:0] shift_amt = k_r[4:1];  // floor(k/2)

  always @(posedge clk_i) begin
    if (zero_r) result_o <= 16'hFFFF;
    else result_o <= lut_out >> shift_amt;
    valid_o <= valid_r;
  end

endmodule