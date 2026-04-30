// KV cache: 32 BRAM banks for key or value storage across all layers and heads
//
// Organization: 4 layers x 8 heads = 32 banks.
// Each bank packs 2 consecutive positions into one 32-bit word so attention
// can read both halves in a single cycle for per-position 2-way parallelism.
// Bank depth: 128 pair_idx x 16 dim = 2048 entries x 32-bit
//
// Write side stays 16-bit per access. The cache derives pair_idx from
// pos_i[7:1] and selects the lower or upper half via byte-write-enable
// based on pos_i[0]. Read side returns the 32-bit pair {odd_pos, even_pos}
//
// Bank select: {layer[1:0], head[2:0]}. Read latency: 2 cycles

module kv_cache (
  input  wire        clk_i,
  input  wire [1:0]  layer_i,
  input  wire [2:0]  head_i,
  input  wire [7:0]  pos_i,
  input  wire [3:0]  dim_i,
  input  wire        we_i,
  input  wire [15:0] wdata_i,
  output wire [31:0] rdata_o
);

  wire [4:0]  sel       = {layer_i, head_i};
  wire [10:0] addr      = {pos_i[7:1], dim_i};
  wire [31:0] wdata_dup = {wdata_i, wdata_i};
  wire [3:0]  bwe       = pos_i[0] ? 4'b1100 : 4'b0011;

  reg [4:0] sel_r;
  always @(posedge clk_i) sel_r <= sel;

  wire [31:0] bank_rdata [0:31];

  genvar g;
  generate
    for (g = 0; g < 32; g = g + 1) begin : banks
      kv_ram #(.DEPTH(2048), .DATA_W(32)) u_ram (
        .clk_i  (clk_i),
        .addr_i (addr),
        .we_i   (we_i && sel == g[4:0]),
        .bwe_i  (bwe),
        .wdata_i(wdata_dup),
        .rdata_o(bank_rdata[g])
      );
    end
  endgenerate

  assign rdata_o = bank_rdata[sel_r];

endmodule