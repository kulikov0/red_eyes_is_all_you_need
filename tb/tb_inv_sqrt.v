`timescale 1ns / 1ps

module tb_inv_sqrt;

  reg         clk;
  reg         in_valid;
  reg  [13:0] d;
  wire        out_valid;
  wire [15:0] result;

  inv_sqrt #(
    .D_W(14),
    .HEX_FILE("/home/user/red_eyes_is_all_you_need/mem/inv_sqrt_lut.hex")
  ) uut (
    .clk_i   (clk),
    .valid_i (in_valid),
    .d_i     (d),
    .valid_o (out_valid),
    .result_o(result)
  );

  // Clock: 10 ns period
  initial clk = 1'b0;
  always #5 clk = ~clk;

  localparam N_TESTS = 30;
  reg [13:0] test_d   [0:N_TESTS-1];
  reg [15:0] test_exp [0:N_TESTS-1];

  integer errors;
  integer ti; // test index
  integer fd;

  initial begin
    // Basic values
    test_d[ 0] = 14'd0;      test_exp[ 0] = 16'hFFFF;   // d=0: saturate
    test_d[ 1] = 14'd1;      test_exp[ 1] = 16'd32768;  // 1/sqrt(1)
    test_d[ 2] = 14'd2;      test_exp[ 2] = 16'd23170;  // 1/sqrt(2)
    test_d[ 3] = 14'd3;      test_exp[ 3] = 16'd18919;  // 1/sqrt(3)
    test_d[ 4] = 14'd4;      test_exp[ 4] = 16'd16384;  // 1/sqrt(4)
    test_d[ 5] = 14'd9;      test_exp[ 5] = 16'd10922;  // 1/sqrt(9)
    test_d[ 6] = 14'd16;     test_exp[ 6] = 16'd8192;   // 1/sqrt(16)
    test_d[ 7] = 14'd25;     test_exp[ 7] = 16'd6553;   // 1/sqrt(25)
    test_d[ 8] = 14'd64;     test_exp[ 8] = 16'd4096;   // 1/sqrt(64)
    test_d[ 9] = 14'd100;    test_exp[ 9] = 16'd3276;   // 1/sqrt(100)
    test_d[10] = 14'd128;    test_exp[10] = 16'd2896;   // 1/sqrt(128)
    test_d[11] = 14'd256;    test_exp[11] = 16'd2048;   // 1/sqrt(256)
    test_d[12] = 14'd1000;   test_exp[12] = 16'd1036;   // 1/sqrt(1000)
    test_d[13] = 14'd10000;  test_exp[13] = 16'd327;    // 1/sqrt(10000)
    test_d[14] = 14'd16383;  test_exp[14] = 16'd256;    // 1/sqrt(16383) max input

    // Edge cases: odd values between powers of 2
    test_d[15] = 14'd5;      test_exp[15] = 16'd14654;  // 5
    test_d[16] = 14'd17;     test_exp[16] = 16'd7947;   // 2^4 + 1

    // Edge cases: all-ones patterns (2^n - 1)
    test_d[17] = 14'd7;      test_exp[17] = 16'd12385;  // 0b111
    test_d[18] = 14'd15;     test_exp[18] = 16'd8460;   // 0b1111
    test_d[19] = 14'd127;    test_exp[19] = 16'd2907;   // 0b1111111
    test_d[20] = 14'd255;    test_exp[20] = 16'd2052;   // 0b11111111
    test_d[21] = 14'd511;    test_exp[21] = 16'd1449;   // 0b111111111
    test_d[22] = 14'd1023;   test_exp[22] = 16'd1025;   // 0b1111111111
    test_d[23] = 14'd8191;   test_exp[23] = 16'd362;    // 2^13 - 1

    // Edge cases: exact powers of 2 (even/odd k boundary)
    test_d[24] = 14'd8;      test_exp[24] = 16'd11585;  // 2^3,  k=3 odd
    test_d[25] = 14'd512;    test_exp[25] = 16'd1448;   // 2^9,  k=9 odd
    test_d[26] = 14'd1024;   test_exp[26] = 16'd1024;   // 2^10, k=10 even
    test_d[27] = 14'd4096;   test_exp[27] = 16'd512;    // 2^12, k=12 even
    test_d[28] = 14'd8192;   test_exp[28] = 16'd362;    // 2^13, k=13 odd

    // Edge case: near max input
    test_d[29] = 14'd16382;  test_exp[29] = 16'd256;    // max - 1

    errors = 0;
    in_valid = 1'b0;
    d = 14'd0;

    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_inv_sqrt.log", "w");

    // Wait for initialization
    #20;

    $display("=== inv_sqrt Testbench ===");
    $fwrite(fd, "=== inv_sqrt Testbench ===\n");

    for (ti = 0; ti < N_TESTS; ti = ti + 1) begin
      // Present input
      @(posedge clk);
      d = test_d[ti];
      in_valid = 1'b1;
      @(posedge clk);
      in_valid = 1'b0;

      // Wait 2 cycles for pipeline (stage 1 + stage 2)
      @(posedge clk);
      @(posedge clk);
      #1;

      if (result !== test_exp[ti]) begin
        $display("FAIL d=%0d: got %0d (0x%04x), expected %0d (0x%04x)",
                 test_d[ti], result, result, test_exp[ti], test_exp[ti]);
        $fwrite(fd, "FAIL d=%0d: got %0d (0x%04x), expected %0d (0x%04x)\n",
                 test_d[ti], result, result, test_exp[ti], test_exp[ti]);
        errors = errors + 1;
      end else begin
        $display("OK   d=%0d  result=%0d  (0x%04x)",
                 test_d[ti], result, result);
        $fwrite(fd, "OK   d=%0d  result=%0d  (0x%04x)\n",
                 test_d[ti], result, result);
      end
    end

    if (errors == 0) begin
      $display("=== All %0d tests passed ===", N_TESTS);
      $fwrite(fd, "=== All %0d tests passed ===\n", N_TESTS);
    end else begin
      $display("=== %0d errors out of %0d ===", errors, N_TESTS);
      $fwrite(fd, "=== %0d errors out of %0d ===\n", errors, N_TESTS);
    end

    $fclose(fd);
    $finish;
  end

endmodule