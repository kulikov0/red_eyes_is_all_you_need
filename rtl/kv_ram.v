// Read-write BRAM primitive for KV cache
//
// Synchronous read (registered output), synchronous write
// Write-first: if reading and writing same address, read returns new data

module kv_ram #(
  parameter DEPTH = 4096
) (
  input  wire                      clk_i,
  input  wire [$clog2(DEPTH)-1:0]  addr_i,
  input  wire                      we_i,
  input  wire [7:0]                wdata_i,
  output reg  [7:0]                rdata_o
);

  (* ram_style = "block" *) reg [7:0] mem [0:DEPTH-1];

  always @(posedge clk_i) begin
    if (we_i)
      mem[addr_i] <= wdata_i;
    rdata_o <= mem[addr_i];
  end

endmodule