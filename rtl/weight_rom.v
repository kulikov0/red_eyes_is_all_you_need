// Synchronous-read ROM for large weight tensors, maps to BRAM.
//
// Auto-instantiated by weight_store and weight_store_w16. DATA_W=8 for byte
// ROMs, DATA_W=128 for the 16-way packed matvec ROMs that hold 16 adjacent
// rows at the same column in one BRAM word.

module weight_rom #(
  parameter DEPTH    = 1024,
  parameter DATA_W   = 8,
  parameter HEX_FILE = "weights.hex"
) (
  input  wire                     clk_i,
  input  wire [$clog2(DEPTH)-1:0] addr_i,
  output reg  [DATA_W-1:0]        data_o
);

  (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:DEPTH-1];

  initial begin
    $readmemh(HEX_FILE, mem);
  end

  always @(posedge clk_i) begin
    data_o <= mem[addr_i];
  end

endmodule