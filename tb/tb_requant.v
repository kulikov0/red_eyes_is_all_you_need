`timescale 1ns / 1ps

module tb_requant;

  integer fd, test_num;

  // DUT A: ACC_W=24, SCALE_W=16, SHIFT=22 (general / matvec case)
  reg  signed [23:0] acc_a;
  reg         [15:0] scale_a;
  wire signed [7:0]  q_a;

  requant #(
    .ACC_W  (24),
    .SCALE_W(16),
    .SHIFT  (22)
  ) dut_a (
    .acc_i  (acc_a),
    .scale_i(scale_a),
    .q_o    (q_a)
  );

  // DUT B: ACC_W=19, SCALE_W=16, SHIFT=19 (attention Q*K^T case)
  reg  signed [18:0] acc_b;
  reg         [15:0] scale_b;
  wire signed [7:0]  q_b;

  requant #(
    .ACC_W  (19),
    .SCALE_W(16),
    .SHIFT  (19)
  ) dut_b (
    .acc_i  (acc_b),
    .scale_i(scale_b),
    .q_o    (q_b)
  );

  task test_a;
    input integer tnum;
    input signed [23:0] a;
    input [15:0] s;
    begin
      acc_a   = a;
      scale_a = s;
      #10;
      $fwrite(fd, "T=%0d DUT=A ACC=%0d SCALE=%0d OUT=%02x\n",
              tnum, acc_a, scale_a, q_a[7:0]);
      $display("Test %0d (A): acc=%0d scale=%0d -> %0d", tnum, acc_a, scale_a, q_a);
    end
  endtask

  task test_b;
    input integer tnum;
    input signed [18:0] a;
    input [15:0] s;
    begin
      acc_b   = a;
      scale_b = s;
      #10;
      $fwrite(fd, "T=%0d DUT=B ACC=%0d SCALE=%0d OUT=%02x\n",
              tnum, acc_b, scale_b, q_b[7:0]);
      $display("Test %0d (B): acc=%0d scale=%0d -> %0d", tnum, acc_b, scale_b, q_b);
    end
  endtask

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_requant.log", "w");

    $display("=== Requant Testbench ===");

    // DUT A tests (SHIFT=22, shift-only uses scale=32768)
    // Test 0: zero
    test_a(0, 24'sd0, 16'd32768);
    // Test 1: small positive (1024 * 32768 >>> 22 = 1024 >>> 7 = 8)
    test_a(1, 24'sd1024, 16'd32768);
    // Test 2: small negative (-1024 * 32768 >>> 22 = -8)
    test_a(2, -24'sd1024, 16'd32768);
    // Test 3: large positive, saturates (100000 >>> 7 = 781 -> clamp 127)
    test_a(3, 24'sd100000, 16'd32768);
    // Test 4: large negative, saturates (-100000 >>> 7 = -782 -> clamp -128)
    test_a(4, -24'sd100000, 16'd32768);
    // Test 5: boundary -> exactly 127 (127 * 128 = 16256)
    test_a(5, 24'sd16256, 16'd32768);
    // Test 6: boundary -> exactly -128 (-128 * 128 = -16384)
    test_a(6, -24'sd16384, 16'd32768);
    // Test 7: odd value, truncation (-1 >>> 7 = -1 in arithmetic shift)
    test_a(7, -24'sd1, 16'd32768);
    // Test 8: scale=16384 (half), acc=2048 -> 2048*16384>>>22 = 2048>>>8 = 8
    test_a(8, 24'sd2048, 16'd16384);
    // Test 9: scale=49152 (1.5x), acc=-1024 -> -1024*49152>>>22 = -12
    test_a(9, -24'sd1024, 16'd49152);
    // Test 10: scale=1 (minimum), large acc -> near zero
    test_a(10, 24'sd1000000, 16'd1);

    // DUT B tests (SHIFT=19, shift-only uses scale=32768)
    // Test 11: small positive (256 * 32768 >>> 19 = 256 >>> 4 = 16)
    test_b(11, 19'sd256, 16'd32768);
    // Test 12: negative (-256 * 32768 >>> 19 = -16)
    test_b(12, -19'sd256, 16'd32768);
    // Test 13: saturates (200000 >>> 4 = 12500 -> 127)
    test_b(13, 19'sd200000, 16'd32768);
    // Test 14: scale multiply, scale=8192 (quarter)
    // 1024 * 8192 >>> 19 = 1024 >>> 6 = 16
    test_b(14, 19'sd1024, 16'd8192);

    $display("=== All tests done ===");
    $fclose(fd);
    $finish;
  end

endmodule