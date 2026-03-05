// fp16 reciprocal square root: 1/sqrt(x) via LOD-LUT-Shift
// Ref: https://www.mdpi.com/2072-666X/17/1/84
//
// LOD + bipartite LUT (512x16, inv_sqrt_lut.hex) + exponent adjust
//
// fp16 x = 2^(e-15) * (1 + f/1024), n = e - 15
//   Even n: 1/sqrt(x) = 2^(-n/2) * 1/sqrt(1.f)     -> LUT parity=0
//   Odd  n: 1/sqrt(x) = 2^(-(n-1)/2) * 1/sqrt(2*1.f) -> LUT parity=1
//
// LUT indexed by {parity, f[9:2]}
// Output exponent: (15 - e + parity) / 2 + leading_one_of_LUT
// Output mantissa: extracted from LUT Q1.15 value
//
// Latency: 2 cycles
// Resources: 1 BRAM18 (shared LUT)

module fp16_rsqrt #(
  parameter HEX_FILE = "/home/user/red_eyes_is_all_you_need/mem/inv_sqrt_lut.hex"
) (
  input  wire        clk_i,
  input  wire        valid_i,
  input  wire [15:0] val_i,
  output reg         valid_o,
  output reg  [15:0] result_o
);

  // Stage 0: extract fields, compute LUT address

  wire [4:0] e = val_i[14:10];
  wire [9:0] f = val_i[9:0];
  wire is_zero = (e == 5'd0);
  wire is_inf  = (e == 5'd31);

  // parity: ~e[0] because n = e - 15, and 15 is odd
  // n even when e is odd -> parity=0; n odd when e is even -> parity=1
  wire parity = ~e[0];

  // LUT: same 512x16 BRAM as inv_sqrt
  (* ram_style = "block" *) reg [15:0] lut_mem [0:511];
  initial $readmemh(HEX_FILE, lut_mem);

  wire [8:0] lut_addr = {parity, f[9:2]};

  // Stage 1: LUT read + pipeline registers

  reg [15:0] lut_out;
  always @(posedge clk_i) lut_out <= lut_mem[lut_addr];

  reg [4:0] e_r;
  reg       parity_r;
  reg       zero_r, inf_r;
  reg       valid_r;
  always @(posedge clk_i) begin
    e_r      <= e;
    parity_r <= parity;
    zero_r   <= is_zero;
    inf_r    <= is_inf;
    valid_r  <= valid_i;
  end

  // Stage 2: compute output exponent + mantissa

  // Leading one of LUT output: always bit 14, except entry {0,0} = 32768 (bit 15)
  wire k_is_15 = lut_out[15];

  // Output exponent: (15 - e + parity) / 2 + lead
  // (15 - e + parity) is always even, so division is exact
  wire signed [6:0] base = $signed({2'b0, 5'd15}) -
                            $signed({2'b0, e_r}) +
                            $signed({6'd0, parity_r});
  wire signed [6:0] half_base = base >>> 1;
  wire signed [6:0] out_e_s = half_base + (k_is_15 ? 7'sd15 : 7'sd14);

  // Mantissa from LUT: bit 14 is implicit leading 1, bits [13:4] are fraction
  // RNE rounding from 14 frac bits to 10
  wire [9:0] raw_frac   = lut_out[13:4];
  wire       guard       = lut_out[3];
  wire       round_bit   = lut_out[2];
  wire       sticky      = |lut_out[1:0];
  wire       round_up    = guard & (round_bit | sticky | raw_frac[0]);
  wire [10:0] rounded    = {1'b0, raw_frac} + {10'd0, round_up};
  wire        round_ovf  = rounded[10];

  wire [9:0] out_frac = k_is_15   ? 10'd0 :
                         round_ovf ? 10'd0 : rounded[9:0];
  // round_ovf promotes mantissa to next power -> exp+1
  wire signed [6:0] out_e_adj = round_ovf ? (out_e_s + 7'sd1) : out_e_s;

  always @(posedge clk_i) begin
    if (zero_r) begin
      result_o <= 16'h7C00;
    end else if (inf_r) begin
      result_o <= 16'h0000;
    end else if (out_e_adj <= 7'sd0) begin
      result_o <= 16'h0000;
    end else if (out_e_adj >= 7'sd31) begin
      result_o <= 16'h7C00;
    end else begin
      result_o <= {1'b0, out_e_adj[4:0], out_frac};
    end
    valid_o <= valid_r;
  end

endmodule