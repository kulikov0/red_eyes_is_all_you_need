// weight_rom.v - Synchronous-read ROM for large weight tensors (maps to BRAM)
// Auto-instantiated by weight_store.v

module weight_rom #(
  parameter DEPTH    = 1024,
  parameter HEX_FILE = "weights.hex"
) (
  input  wire                     clk_i,
  input  wire [$clog2(DEPTH)-1:0] addr_i,
  output reg  [7:0]               data_o
);

  (* ram_style = "block" *) reg [7:0] mem [0:DEPTH-1];

  initial begin
    $readmemh(HEX_FILE, mem);
  end

  always @(posedge clk_i) begin
    data_o <= mem[addr_i];
  end

endmodule