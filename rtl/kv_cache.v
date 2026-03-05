// KV cache: 32 BRAM banks for key or value storage across all layers and heads
//
// Organization: 4 layers x 8 heads = 32 banks
// Each bank: 256 positions x 16 dimensions = 4096 entries (DATA_W bits each)
// Instantiate separately for K and V, both with DATA_W=16 (fp16)
// Each bank uses 2 BRAM36 (4096 x 16-bit), total 64 BRAM36 per cache instance
//
// Bank select: {layer[1:0], head[2:0]} (5 bits)
// Bank address: {pos[7:0], dim[3:0]} (12 bits)
// Read latency: 2 cycles (registered addr + registered sel mux)

module kv_cache #(
  parameter DATA_W = 16
) (
  input  wire              clk_i,
  input  wire [1:0]        layer_i,
  input  wire [2:0]        head_i,
  input  wire [7:0]        pos_i,
  input  wire [3:0]        dim_i,
  input  wire              we_i,
  input  wire [DATA_W-1:0] wdata_i,
  output wire [DATA_W-1:0] rdata_o
);

  wire [4:0]  sel  = {layer_i, head_i};
  wire [11:0] addr = {pos_i, dim_i};

  // Registered sel for output mux (1-cycle BRAM latency alignment)
  reg [4:0] sel_r;
  always @(posedge clk_i) sel_r <= sel;

  // 32 BRAM banks
  wire [DATA_W-1:0] bank_rdata [0:31];

  genvar g;
  generate
    for (g = 0; g < 32; g = g + 1) begin : banks
      kv_ram #(.DEPTH(4096), .DATA_W(DATA_W)) u_ram (
        .clk_i  (clk_i),
        .addr_i (addr),
        .we_i   (we_i && sel == g[4:0]),
        .wdata_i(wdata_i),
        .rdata_o(bank_rdata[g])
      );
    end
  endgenerate

  assign rdata_o = bank_rdata[sel_r];

endmodule