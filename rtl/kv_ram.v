// Read-write BRAM primitive for KV cache
//
// Synchronous read (registered output), synchronous write
// Write-first: if reading and writing same address, read returns new data
// Latency: 1 cycle (registered read)

module kv_ram #(
  parameter DEPTH  = 4096,
  parameter DATA_W = 16
) (
  input  wire                      clk_i,
  input  wire [$clog2(DEPTH)-1:0]  addr_i,
  input  wire                      we_i,
  input  wire [DATA_W-1:0]         wdata_i,
  output reg  [DATA_W-1:0]         rdata_o
);

  (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:DEPTH-1];

  always @(posedge clk_i) begin
    if (we_i) begin
      mem[addr_i] <= wdata_i;
    end
    rdata_o <= mem[addr_i];
  end

endmodule