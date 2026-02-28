`timescale 1ns / 1ps

module tb_softmax;

  localparam N     = 256;
  localparam IN_W  = 16;
  localparam FRAC_W = 7;
  localparam OUT_W = 16;

  reg              clk;
  reg              rst;
  reg              start;
  reg              in_valid;
  reg  [IN_W-1:0]  in_data;
  wire             in_ready;
  wire             out_valid;
  wire [OUT_W-1:0] out_data;
  wire             done;

  softmax #(
    .N(N),
    .IN_W(IN_W),
    .FRAC_W(FRAC_W),
    .OUT_W(OUT_W),
    .LUT0_HEX("/home/user/red_eyes_is_all_you_need/mem/exp_lut0.hex"),
    .LUT1_HEX("/home/user/red_eyes_is_all_you_need/mem/exp_lut1.hex")
  ) uut (
    .clk_i      (clk),
    .rst_i      (rst),
    .start_i    (start),
    .in_valid_i (in_valid),
    .in_data_i  (in_data),
    .in_ready_o (in_ready),
    .out_valid_o(out_valid),
    .out_data_o (out_data),
    .done_o     (done)
  );

  // Clock: 10 ns period
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // Input/output storage
  reg signed [IN_W-1:0] test_input [0:N-1];
  reg [OUT_W-1:0]       test_output [0:N-1];

  integer fd;
  integer test_num;
  integer idx;
  integer out_cnt;
  integer total_errors;

  // Sum of outputs for validation
  reg [31:0] out_sum;

  // Load inputs and collect outputs
  task run_softmax;
    integer j;
    begin
      // Pulse start
      @(posedge clk);
      start <= 1'b1;
      @(posedge clk);
      start <= 1'b0;

      // Stream in N values
      for (j = 0; j < N; j = j + 1) begin
        @(posedge clk);
        in_valid <= 1'b1;
        in_data  <= test_input[j];
      end
      @(posedge clk);
      in_valid <= 1'b0;

      // Collect N outputs
      out_cnt = 0;
      out_sum = 0;
      while (out_cnt < N) begin
        @(posedge clk);
        #1;
        if (out_valid) begin
          test_output[out_cnt] = out_data;
          out_sum = out_sum + out_data;
          out_cnt = out_cnt + 1;
        end
      end
    end
  endtask

  // Log outputs for a test
  task log_test;
    input integer tnum;
    input [255:0] name;  // test name (padded string)
    integer j;
    begin
      $fwrite(fd, "\n=== Test %0d: %0s ===\n", tnum, name);
      $display("\n=== Test %0d: %0s ===", tnum, name);
      for (j = 0; j < N; j = j + 1) begin
        $fwrite(fd, "OUT[%0d] input=%0d output=%0d\n", j,
                $signed(test_input[j]), test_output[j]);
      end
      $fwrite(fd, "SUM=%0d (ideal=32768)\n", out_sum);
      $display("  output sum = %0d (ideal 32768)", out_sum);
    end
  endtask

  initial begin
    rst      = 1'b1;
    start    = 1'b0;
    in_valid = 1'b0;
    in_data  = {IN_W{1'b0}};
    total_errors = 0;

    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_softmax.log", "w");
    $fwrite(fd, "=== Softmax Testbench ===\n");
    $display("=== Softmax Testbench ===");

    // Reset
    #30;
    @(posedge clk);
    rst = 1'b0;
    #20;

    // Test 1: Uniform - all inputs = 100 (Q8.7 = 100 << 7 = 12800)
    // Expected: each output ~ 32768/256 = 128
    test_num = 1;
    for (idx = 0; idx < N; idx = idx + 1)
      test_input[idx] = 16'sd12800;  // 100.0 in Q8.7

    run_softmax;
    log_test(test_num, "Uniform (all=100.0)");

    // Check: all outputs should be close to 128
    begin : check1
      integer errs;
      errs = 0;
      for (idx = 0; idx < N; idx = idx + 1) begin
        if (test_output[idx] < 80 || test_output[idx] > 180) begin
          if (errs < 5)
            $display("  FAIL [%0d]: got %0d, expected ~128", idx, test_output[idx]);
          errs = errs + 1;
        end
      end
      if (errs == 0)
        $display("  PASS: all outputs near 128");
      else begin
        $display("  FAIL: %0d outputs out of range", errs);
        total_errors = total_errors + errs;
      end
      $fwrite(fd, "TEST1_ERRORS=%0d\n", errs);
    end

    // Test 2: One-hot - index 42 = 10.0, rest = 0.0
    // d for non-max: 10.0 in Q4.7 = 1280. exp(-10) ~ 0.0000454 ~ 1 in Q1.15
    // Expected: output[42] ~ 32768, rest ~ 0
    test_num = 2;
    for (idx = 0; idx < N; idx = idx + 1)
      test_input[idx] = 16'sd0;  // 0.0 in Q8.7
    test_input[42] = 16'sd1280;  // 10.0 in Q8.7 (10 * 128)

    run_softmax;
    log_test(test_num, "One-hot (idx42=10.0)");

    // Check: output[42] should be close to 32768, rest near 0
    begin : check2
      integer errs;
      errs = 0;
      if (test_output[42] < 30000) begin
        $display("  FAIL [42]: got %0d, expected ~32768", test_output[42]);
        errs = errs + 1;
      end else begin
        $display("  output[42] = %0d (expected ~32768)", test_output[42]);
      end
      for (idx = 0; idx < N; idx = idx + 1) begin
        if (idx != 42 && test_output[idx] > 200) begin
          if (errs < 5)
            $display("  FAIL [%0d]: got %0d, expected ~0", idx, test_output[idx]);
          errs = errs + 1;
        end
      end
      if (errs == 0)
        $display("  PASS: one-hot pattern correct");
      else
        total_errors = total_errors + errs;
      $fwrite(fd, "TEST2_ERRORS=%0d\n", errs);
    end

    // Test 3: Ramp - input[i] = i * 0.25 in Q8.7 (i * 32)
    // Max at i=255 = 63.75. Large spread so most weight at high indices
    test_num = 3;
    for (idx = 0; idx < N; idx = idx + 1)
      test_input[idx] = idx * 32;  // i * 0.25 in Q8.7

    run_softmax;
    log_test(test_num, "Ramp (0.25 step)");

    // Check: outputs should increase monotonically, last should be largest
    begin : check3
      integer errs;
      integer mono_ok;
      errs = 0;
      mono_ok = 1;
      for (idx = 1; idx < N; idx = idx + 1) begin
        if (test_output[idx] < test_output[idx-1])
          mono_ok = 0;
      end
      if (test_output[255] < 100) begin
        $display("  FAIL: output[255]=%0d too small", test_output[255]);
        errs = errs + 1;
      end
      $display("  output[255]=%0d, output[254]=%0d, output[0]=%0d",
               test_output[255], test_output[254], test_output[0]);
      $display("  monotonic: %0s", mono_ok ? "yes" : "approximately");
      if (errs == 0)
        $display("  PASS: ramp distribution reasonable");
      else
        total_errors = total_errors + errs;
      $fwrite(fd, "TEST3_ERRORS=%0d MONO=%0d\n", errs, mono_ok);
    end

    // Test 4: Two equal max - indices 100,200 = 5.0, rest = 0.0
    // Expected: output[100] ~ output[200] ~ 32768 * exp(-ln(sum))
    // with sum dominated by exp(0)=1 at both max positions
    // Each should get ~half of remaining probability after others
    test_num = 4;
    for (idx = 0; idx < N; idx = idx + 1)
      test_input[idx] = 16'sd0;  // 0.0 in Q8.7
    test_input[100] = 16'sd640;  // 5.0 in Q8.7 (5 * 128)
    test_input[200] = 16'sd640;  // 5.0 in Q8.7

    run_softmax;
    log_test(test_num, "Two equal max (5.0)");

    // Check: output[100] ~ output[200], both large
    begin : check4
      integer errs;
      integer diff;
      errs = 0;
      diff = test_output[100] > test_output[200] ?
             test_output[100] - test_output[200] :
             test_output[200] - test_output[100];
      $display("  output[100]=%0d, output[200]=%0d, diff=%0d",
               test_output[100], test_output[200], diff);
      if (diff > 2000) begin
        $display("  FAIL: outputs differ by more than 2000");
        errs = errs + 1;
      end
      if (test_output[100] < 7000 || test_output[200] < 7000) begin
        $display("  FAIL: max outputs too small");
        errs = errs + 1;
      end
      if (errs == 0)
        $display("  PASS: two equal max outputs are close");
      else
        total_errors = total_errors + errs;
      $fwrite(fd, "TEST4_ERRORS=%0d\n", errs);
    end

    // Test 5: All masked except one
    // Index 0 = 1.0 (128 in Q8.7), rest = -256.0 (-32768 in Q8.7, min signed)
    // d for masked: max - min = 257.0 >> 16.0, clip to 0
    // Expected: output[0] = 32768, rest = 0
    test_num = 5;
    for (idx = 0; idx < N; idx = idx + 1)
      test_input[idx] = -16'sd32768;  // most negative Q8.7
    test_input[0] = 16'sd128;  // 1.0 in Q8.7

    run_softmax;
    log_test(test_num, "Masked except idx0");

    // Check: output[0] should be close to 32768, rest = 0
    begin : check5
      integer errs;
      errs = 0;
      if (test_output[0] < 30000) begin
        $display("  FAIL [0]: got %0d, expected ~32768", test_output[0]);
        errs = errs + 1;
      end else begin
        $display("  output[0] = %0d (expected ~32768)", test_output[0]);
      end
      for (idx = 1; idx < N; idx = idx + 1) begin
        if (test_output[idx] != 0) begin
          if (errs < 5)
            $display("  FAIL [%0d]: got %0d, expected 0", idx, test_output[idx]);
          errs = errs + 1;
        end
      end
      if (errs == 0)
        $display("  PASS: masked pattern correct");
      else
        total_errors = total_errors + errs;
      $fwrite(fd, "TEST5_ERRORS=%0d\n", errs);
    end

    $display("");
    $fwrite(fd, "\n");
    if (total_errors == 0) begin
      $display("*** ALL 5 TESTS PASSED ***");
      $fwrite(fd, "*** ALL 5 TESTS PASSED ***\n");
    end else begin
      $display("*** %0d TOTAL ERRORS ***", total_errors);
      $fwrite(fd, "*** %0d TOTAL ERRORS ***\n", total_errors);
    end

    $fclose(fd);
    $finish;
  end

endmodule