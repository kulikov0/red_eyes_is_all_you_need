`timescale 1ns / 1ps

module tb_gelu;

  reg        clk;
  reg  [1:0] layer_sel;
  reg  [7:0] in_data;
  wire [7:0] out_data;

  gelu #(
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/gelu_lut.hex")
  ) uut (
    .clk_i      (clk),
    .layer_sel_i(layer_sel),
    .in_data_i  (in_data),
    .out_data_o (out_data)
  );

  // Clock: 10 ns period
  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer layer, i, fd;

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_gelu.log", "w");

    layer_sel = 2'd0;
    in_data   = 8'd0;
    #20;

    $display("=== GELU Testbench ===");
    $fwrite(fd, "=== GELU Testbench ===\n");

    for (layer = 0; layer < 4; layer = layer + 1) begin
      $fwrite(fd, "=== Layer %0d ===\n", layer);

      for (i = 0; i < 256; i = i + 1) begin
        @(posedge clk);
        layer_sel = layer[1:0];
        in_data   = i[7:0];

        // Wait 1 cycle for BRAM read
        @(posedge clk);
        #1;

        $fwrite(fd, "L=%0d IN=%02x OUT=%02x\n", layer, i[7:0], out_data);
      end

      $display("Layer %0d done (256 entries)", layer);
    end

    $display("=== Done: 1024 entries logged ===");
    $fwrite(fd, "=== Done: 1024 entries logged ===\n");

    $fclose(fd);
    $finish;
  end

  // Timeout
  initial begin
    #5_000_000;
    $display("TIMEOUT");
    $finish;
  end

endmodule
