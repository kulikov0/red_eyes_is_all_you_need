// Read-write BRAM primitive for KV cache.
// Synchronous registered read, byte-write-enable masked write.
// bwe_i has one bit per byte of DATA_W; tie to all-ones for full-word writes.
// Write-first: if reading and writing the same address, read returns new data

module kv_ram #(
  parameter DEPTH  = 4096,
  parameter DATA_W = 16,
  localparam BWE_W = (DATA_W + 7) / 8
) (
  input  wire                      clk_i,
  input  wire [$clog2(DEPTH)-1:0]  addr_i,
  input  wire                      we_i,
  input  wire [BWE_W-1:0]          bwe_i,
  input  wire [DATA_W-1:0]         wdata_i,
  output reg  [DATA_W-1:0]         rdata_o
);

  (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:DEPTH-1];

  integer b;
  always @(posedge clk_i) begin
    if (we_i) begin
      for (b = 0; b < BWE_W; b = b + 1) begin
        if (bwe_i[b]) begin
          mem[addr_i][b*8 +: 8] <= wdata_i[b*8 +: 8];
        end
      end
    end
    rdata_o <= mem[addr_i];
  end

endmodule
