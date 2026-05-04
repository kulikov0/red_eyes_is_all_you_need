// KV cache: 32 BRAM banks for key or value storage across all layers and heads
//
// Organization: 4 layers x 8 heads = 32 banks.
// Each bank packs 4 consecutive positions into one 64-bit word so attention
// can read all four halves in a single cycle for per-position 4-way parallelism.
// Bank depth: 64 quad_idx x 16 dim = 1024 entries x 64-bit
//
// Write side stays 16-bit per access. The cache derives quad_idx from
// pos_i[7:2] and selects one of four quarters via byte-write-enable
// based on pos_i[1:0]. Read side returns the 64-bit quad
// {pos%4=3, pos%4=2, pos%4=1, pos%4=0}
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
  output wire [63:0] rdata_o
);

  wire [4:0]  sel       = {layer_i, head_i};
  wire [9:0]  addr      = {pos_i[7:2], dim_i};
  wire [63:0] wdata_dup = {wdata_i, wdata_i, wdata_i, wdata_i};
  reg  [7:0]  bwe;
  always @(*) begin
    case (pos_i[1:0])
      2'b00: bwe = 8'b00000011;
      2'b01: bwe = 8'b00001100;
      2'b10: bwe = 8'b00110000;
      2'b11: bwe = 8'b11000000;
    endcase
  end

  reg [4:0] sel_r;
  always @(posedge clk_i) sel_r <= sel;

  wire [63:0] bank_rdata [0:31];

  genvar g;
  generate
    for (g = 0; g < 32; g = g + 1) begin : banks
      kv_ram #(.DEPTH(1024), .DATA_W(64)) u_ram (
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