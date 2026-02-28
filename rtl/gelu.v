module gelu #(
  parameter HEX_FILE = "/home/user/red_eyes_is_all_you_need/mem/gelu_lut.hex"
) (
  input  wire       clk_i,
  input  wire [1:0] layer_sel_i,
  input  wire [7:0] in_data_i,
  output reg  [7:0] out_data_o
);

  (* ram_style = "block" *) reg [7:0] lut_mem [0:1023];
  initial $readmemh(HEX_FILE, lut_mem);

  wire [9:0] addr = {layer_sel_i, in_data_i};

  always @(posedge clk_i) begin
    out_data_o <= lut_mem[addr];
  end

endmodule