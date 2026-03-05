`timescale 1ns / 1ps

module tb_gelu;

  reg        clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg        valid_in;
  reg [15:0] x_in;
  wire       valid_out;
  wire [15:0] y_out;

  gelu dut (
    .clk_i   (clk),
    .valid_i (valid_in),
    .x_i     (x_in),
    .valid_o (valid_out),
    .y_o     (y_out)
  );

  // Test inputs from hex file
  localparam N_TESTS = 556;
  reg [15:0] test_mem [0:N_TESTS-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/gelu_test_inputs.hex", test_mem);

  // Pipeline tracking: remember input that produced each output
  reg [15:0] pipe0, pipe1;

  integer fd, i, out_count;

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_gelu.log", "w");

    valid_in = 1'b0;
    x_in = 16'h0000;
    pipe0 = 16'h0000;
    pipe1 = 16'h0000;
    out_count = 0;

    // Reset
    repeat(5) @(posedge clk);

    $display("=== GELU FP16 Testbench (%0d tests) ===", N_TESTS);
    $fwrite(fd, "=== GELU FP16 Testbench (%0d tests) ===\n", N_TESTS);

    // Stream inputs continuously, 1 per cycle
    for (i = 0; i < N_TESTS + 2; i = i + 1) begin
      @(posedge clk);
      #1;
      // Capture output from 2 cycles ago
      if (valid_out) begin
        $fwrite(fd, "IN=%04x OUT=%04x\n", pipe1, y_out);
        out_count = out_count + 1;
      end
      // Shift pipeline tracker
      pipe1 = pipe0;
      pipe0 = (i < N_TESTS) ? test_mem[i] : 16'h0000;
      // Drive input
      if (i < N_TESTS) begin
        x_in = test_mem[i];
        valid_in = 1'b1;
      end else begin
        valid_in = 1'b0;
      end
    end

    // Flush remaining
    repeat(3) @(posedge clk);
    #1;
    if (valid_out) begin
      $fwrite(fd, "IN=%04x OUT=%04x\n", pipe1, y_out);
      out_count = out_count + 1;
    end

    $display("=== All %0d outputs captured ===", out_count);
    $fwrite(fd, "=== All %0d outputs captured ===\n", out_count);

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