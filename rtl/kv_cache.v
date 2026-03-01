// KV cache: 64 BRAM banks for key/value storage across all layers and heads
//
// Organization: 4 layers x 2 (K/V) x 8 heads = 64 banks
// Each bank: 256 positions x 16 dimensions = 4096 bytes (1 BRAM36)
// Total: 64 BRAM36
//
// Bank select: {layer[1:0], kv_sel, head[2:0]} (6 bits)
// Bank address: {pos[7:0], dim[3:0]} (12 bits)
// Read latency: 2 cycles (registered addr + registered sel mux)

module kv_cache (
  input  wire        clk_i,
  input  wire [1:0]  layer_i,
  input  wire        kv_sel_i,
  input  wire [2:0]  head_i,
  input  wire [7:0]  pos_i,
  input  wire [3:0]  dim_i,
  input  wire        we_i,
  input  wire [7:0]  wdata_i,
  output wire [7:0]  rdata_o
);

  wire [5:0]  sel  = {layer_i, kv_sel_i, head_i};
  wire [11:0] addr = {pos_i, dim_i};

  // Registered sel for output mux (1-cycle BRAM latency alignment)
  reg [5:0] sel_r;
  always @(posedge clk_i) sel_r <= sel;

  // 64 BRAM banks
  wire [7:0] bank_rdata [0:63];

  genvar g;
  generate
    for (g = 0; g < 64; g = g + 1) begin : banks
      kv_ram #(.DEPTH(4096)) u_ram (
        .clk_i  (clk_i),
        .addr_i (addr),
        .we_i   (we_i && sel == g[5:0]),
        .wdata_i(wdata_i),
        .rdata_o(bank_rdata[g])
      );
    end
  endgenerate

  assign rdata_o = bank_rdata[sel_r];

endmodule