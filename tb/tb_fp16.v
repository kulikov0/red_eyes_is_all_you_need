`timescale 1ns / 1ps

// Unified FP16 testbench: add, mul, mac, from_int8, to_int8, to_q167, q115_to_fp16, rsqrt, matvec
module tb_fp16;

  reg clk;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer errors, fd;
  reg rst;
  integer ti, ei, ri, base;
  reg [15:0] mac_acc_reg;

  // fp16_add
  reg  [15:0] add_a, add_b;
  wire [15:0] add_sum;
  fp16_add u_add (.clk_i(clk), .a_i(add_a), .b_i(add_b), .sum_o(add_sum));

  localparam N_ADD = 50;
  reg [47:0] tv_add [0:N_ADD-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_add_vectors.hex", tv_add);

  // fp16_mul
  reg  [15:0] mul_a, mul_b;
  wire [15:0] mul_prod;
  fp16_mul u_mul (.clk_i(clk), .a_i(mul_a), .b_i(mul_b), .prod_o(mul_prod));

  localparam N_MUL = 48;
  reg [47:0] tv_mul [0:N_MUL-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_mul_vectors.hex", tv_mul);

  // fp16_mac
  reg  [15:0] mac_a, mac_b, mac_acc_in;
  wire [15:0] mac_acc_out;
  fp16_mac u_mac (.clk_i(clk), .a_i(mac_a), .b_i(mac_b), .acc_i(mac_acc_in), .acc_o(mac_acc_out));

  localparam N_MAC = 5;
  localparam MAC_LEN = 16;
  reg [31:0] tv_mac_pairs [0:N_MAC*MAC_LEN-1];
  reg [15:0] tv_mac_exp [0:N_MAC-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_mac_pairs.hex", tv_mac_pairs);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_mac_expected.hex", tv_mac_exp);
  end

  // fp16_from_int8
  reg  [7:0]  cvt_in;
  wire [15:0] cvt_fp16;
  fp16_from_int8 u_from (.val_i(cvt_in), .fp16_o(cvt_fp16));

  localparam N_FROM = 256;
  reg [23:0] tv_from [0:N_FROM-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_from_int8_vectors.hex", tv_from);

  // fp16_to_int8
  reg  [15:0] to_in;
  wire [7:0]  to_int8;
  fp16_to_int8 u_to (.val_i(to_in), .int8_o(to_int8));

  localparam N_TO = 51;
  reg [23:0] tv_to [0:N_TO-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_to_int8_vectors.hex", tv_to);

  // fp16_to_q167
  reg  [15:0] q167_in;
  wire [23:0] q167_out;
  fp16_to_q167 u_q167 (.val_i(q167_in), .q167_o(q167_out));

  localparam N_Q167 = 57;
  reg [39:0] tv_q167 [0:N_Q167-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_to_q167_vectors.hex", tv_q167);

  // q115_to_fp16
  reg  [15:0] q115_in;
  wire [15:0] q115_out;
  q115_to_fp16 u_q115 (.val_i(q115_in), .fp16_o(q115_out));

  localparam N_Q115 = 66;
  reg [31:0] tv_q115 [0:N_Q115-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/q115_to_fp16_vectors.hex", tv_q115);

  // fp16_rsqrt (2-cycle registered, needs LUT BRAM)
  reg         rsqrt_valid_in;
  reg  [15:0] rsqrt_in;
  wire [15:0] rsqrt_out;
  fp16_rsqrt u_rsqrt (
    .clk_i(clk), .valid_i(rsqrt_valid_in),
    .val_i(rsqrt_in), .valid_o(), .result_o(rsqrt_out)
  );

  localparam N_RSQRT = 62;
  reg [31:0] tv_rsqrt [0:N_RSQRT-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_rsqrt_vectors.hex", tv_rsqrt);

  // matvec_fp16 test 1: 4x4
  localparam T1_IN = 4, T1_OUT = 4;
  reg                              mv1_start;
  reg  [T1_IN*16-1:0]             mv1_in;
  reg  [15:0]                     mv1_scale;
  wire [$clog2(T1_OUT*T1_IN)-1:0] mv1_waddr;
  wire [T1_OUT*16-1:0]            mv1_out;
  wire                            mv1_done;
  reg signed [7:0] mv1_wmem [0:T1_OUT*T1_IN-1];
  reg signed [7:0] mv1_wdata;
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_4x4_weights.hex",
              mv1_wmem);
  end
  always @(posedge clk) mv1_wdata <= mv1_wmem[mv1_waddr];

  matvec_fp16 #(.IN_DIM(T1_IN), .OUT_DIM(T1_OUT)) u_mv1 (
    .clk_i(clk), .rst_i(rst), .start_i(mv1_start),
    .in_vec_i(mv1_in), .scale_i(mv1_scale),
    .weight_addr_o(mv1_waddr), .weight_data_i(mv1_wdata),
    .out_vec_o(mv1_out), .done_o(mv1_done)
  );

  reg [15:0] mv1_iv [0:T1_IN-1];
  reg [15:0] mv1_exp [0:T1_OUT-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_4x4_input.hex", mv1_iv);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_4x4_expected.hex", mv1_exp);
  end

  // matvec_fp16 test 2: 8x4
  localparam T2_IN = 4, T2_OUT = 8;
  reg                              mv2_start;
  reg  [T2_IN*16-1:0]             mv2_in;
  reg  [15:0]                     mv2_scale;
  wire [$clog2(T2_OUT*T2_IN)-1:0] mv2_waddr;
  wire [T2_OUT*16-1:0]            mv2_out;
  wire                            mv2_done;
  reg signed [7:0] mv2_wmem [0:T2_OUT*T2_IN-1];
  reg signed [7:0] mv2_wdata;
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_8x4_weights.hex",
              mv2_wmem);
  end
  always @(posedge clk) mv2_wdata <= mv2_wmem[mv2_waddr];

  matvec_fp16 #(.IN_DIM(T2_IN), .OUT_DIM(T2_OUT)) u_mv2 (
    .clk_i(clk), .rst_i(rst), .start_i(mv2_start),
    .in_vec_i(mv2_in), .scale_i(mv2_scale),
    .weight_addr_o(mv2_waddr), .weight_data_i(mv2_wdata),
    .out_vec_o(mv2_out), .done_o(mv2_done)
  );

  reg [15:0] mv2_iv [0:T2_IN-1];
  reg [15:0] mv2_exp [0:T2_OUT-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_8x4_input.hex", mv2_iv);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_8x4_expected.hex", mv2_exp);
  end

  initial begin
    rst = 1'b1;
    add_a = 16'd0; add_b = 16'd0;
    mul_a = 16'd0; mul_b = 16'd0;
    mac_a = 16'd0; mac_b = 16'd0; mac_acc_in = 16'd0;
    cvt_in = 8'd0; to_in = 16'd0; q167_in = 16'd0; q115_in = 16'd0;
    rsqrt_valid_in = 1'b0; rsqrt_in = 16'd0;
    mv1_start = 1'b0; mv2_start = 1'b0;
    mv1_in = 0; mv2_in = 0;
    mv1_scale = 16'h2c00; mv2_scale = 16'h2c00;
    errors = 0;

    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_fp16.log", "w");

    #30;
    @(posedge clk);
    rst = 1'b0;
    #10;

    // fp16_add
    $display("=== fp16_add (%0d tests) ===", N_ADD);
    $fwrite(fd, "=== fp16_add (%0d tests) ===\n", N_ADD);
    for (ti = 0; ti < N_ADD; ti = ti + 1) begin
      @(posedge clk);
      add_a = tv_add[ti][47:32];
      add_b = tv_add[ti][31:16];
      @(posedge clk); #1;
      begin : add_chk
        reg [15:0] expected;
        reg nan_e, nan_g;
        expected = tv_add[ti][15:0];
        nan_e = (expected[14:10] == 5'd31) && (expected[9:0] != 10'd0);
        nan_g = (add_sum[14:10] == 5'd31) && (add_sum[9:0] != 10'd0);
        if (nan_e && nan_g) begin
          $fwrite(fd, "ADD [%0d] a=%04x b=%04x got=%04x exp=%04x OK\n",
                  ti, add_a, add_b, add_sum, expected);
        end else if (add_sum !== expected) begin
          $fwrite(fd, "ADD [%0d] a=%04x b=%04x got=%04x exp=%04x FAIL\n",
                  ti, add_a, add_b, add_sum, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "ADD [%0d] a=%04x b=%04x got=%04x exp=%04x OK\n",
                  ti, add_a, add_b, add_sum, expected);
        end
      end
    end

    // fp16_mul
    $display("=== fp16_mul (%0d tests) ===", N_MUL);
    $fwrite(fd, "=== fp16_mul (%0d tests) ===\n", N_MUL);
    for (ti = 0; ti < N_MUL; ti = ti + 1) begin
      @(posedge clk);
      mul_a = tv_mul[ti][47:32];
      mul_b = tv_mul[ti][31:16];
      @(posedge clk); #1;
      begin : mul_chk
        reg [15:0] expected;
        reg nan_e, nan_g;
        expected = tv_mul[ti][15:0];
        nan_e = (expected[14:10] == 5'd31) && (expected[9:0] != 10'd0);
        nan_g = (mul_prod[14:10] == 5'd31) && (mul_prod[9:0] != 10'd0);
        if (nan_e && nan_g) begin
          $fwrite(fd, "MUL [%0d] a=%04x b=%04x got=%04x exp=%04x OK\n",
                  ti, mul_a, mul_b, mul_prod, expected);
        end else if (mul_prod !== expected) begin
          $fwrite(fd, "MUL [%0d] a=%04x b=%04x got=%04x exp=%04x FAIL\n",
                  ti, mul_a, mul_b, mul_prod, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "MUL [%0d] a=%04x b=%04x got=%04x exp=%04x OK\n",
                  ti, mul_a, mul_b, mul_prod, expected);
        end
      end
    end

    // fp16_mac
    $display("=== fp16_mac (%0d tests) ===", N_MAC);
    $fwrite(fd, "=== fp16_mac (%0d tests) ===\n", N_MAC);
    for (ti = 0; ti < N_MAC; ti = ti + 1) begin
      base = ti * MAC_LEN;
      mac_acc_reg = 16'd0;
      for (ei = 0; ei < MAC_LEN; ei = ei + 1) begin
        @(posedge clk);
        mac_a = tv_mac_pairs[base + ei][31:16];
        mac_b = tv_mac_pairs[base + ei][15:0];
        mac_acc_in = mac_acc_reg;
        @(posedge clk); #1;
        mac_acc_reg = mac_acc_out;
      end
      begin : mac_chk
        reg [15:0] expected;
        reg nan_e, nan_g;
        expected = tv_mac_exp[ti];
        nan_e = (expected[14:10] == 5'd31) && (expected[9:0] != 10'd0);
        nan_g = (mac_acc_reg[14:10] == 5'd31) && (mac_acc_reg[9:0] != 10'd0);
        if (nan_e && nan_g) begin
          $fwrite(fd, "MAC [%0d] got=%04x exp=%04x OK\n", ti, mac_acc_reg, expected);
        end else if (mac_acc_reg !== expected) begin
          $fwrite(fd, "MAC [%0d] got=%04x exp=%04x FAIL\n", ti, mac_acc_reg, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "MAC [%0d] got=%04x exp=%04x OK\n", ti, mac_acc_reg, expected);
        end
      end
    end

    // fp16_from_int8
    $display("=== fp16_from_int8 (%0d tests) ===", N_FROM);
    $fwrite(fd, "=== fp16_from_int8 (%0d tests) ===\n", N_FROM);
    for (ti = 0; ti < N_FROM; ti = ti + 1) begin
      cvt_in = tv_from[ti][23:16];
      #10;
      begin : from_chk
        reg [15:0] expected;
        expected = tv_from[ti][15:0];
        if (cvt_fp16 !== expected) begin
          $fwrite(fd, "FROM [%0d] in=%02x got=%04x exp=%04x FAIL\n",
                  ti, cvt_in, cvt_fp16, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "FROM [%0d] in=%02x got=%04x exp=%04x OK\n", ti, cvt_in, cvt_fp16, expected);
        end
      end
    end

    // fp16_to_int8
    $display("=== fp16_to_int8 (%0d tests) ===", N_TO);
    $fwrite(fd, "=== fp16_to_int8 (%0d tests) ===\n", N_TO);
    for (ti = 0; ti < N_TO; ti = ti + 1) begin
      to_in = tv_to[ti][23:8];
      #10;
      begin : to_chk
        reg [7:0] expected;
        expected = tv_to[ti][7:0];
        if (to_int8 !== expected) begin
          $fwrite(fd, "TO [%0d] in=%04x got=%02x exp=%02x FAIL\n", ti, to_in, to_int8, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "TO [%0d] in=%04x got=%02x exp=%02x OK\n", ti, to_in, to_int8, expected);
        end
      end
    end

    // fp16_to_q167
    $display("=== fp16_to_q167 (%0d tests) ===", N_Q167);
    $fwrite(fd, "=== fp16_to_q167 (%0d tests) ===\n", N_Q167);
    for (ti = 0; ti < N_Q167; ti = ti + 1) begin
      q167_in = tv_q167[ti][39:24];
      #10;
      begin : q167_chk
        reg [23:0] expected;
        expected = tv_q167[ti][23:0];
        if (q167_out !== expected) begin
          $fwrite(fd, "Q167 [%0d] in=%04x got=%06x exp=%06x FAIL\n",
                  ti, q167_in, q167_out, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "Q167 [%0d] in=%04x got=%06x exp=%06x OK\n", ti, q167_in, q167_out, expected);
        end
      end
    end

    // q115_to_fp16
    $display("=== q115_to_fp16 (%0d tests) ===", N_Q115);
    $fwrite(fd, "=== q115_to_fp16 (%0d tests) ===\n", N_Q115);
    for (ti = 0; ti < N_Q115; ti = ti + 1) begin
      q115_in = tv_q115[ti][31:16];
      #10;
      begin : q115_chk
        reg [15:0] expected;
        expected = tv_q115[ti][15:0];
        if (q115_out !== expected) begin
          $fwrite(fd, "Q115 [%0d] in=%04x got=%04x exp=%04x FAIL\n",
                  ti, q115_in, q115_out, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "Q115 [%0d] in=%04x got=%04x exp=%04x OK\n", ti, q115_in, q115_out, expected);
        end
      end
    end

    // fp16_rsqrt (2-cycle pipeline: present input, wait 2 clocks, read output)
    $display("=== fp16_rsqrt (%0d tests) ===", N_RSQRT);
    $fwrite(fd, "=== fp16_rsqrt (%0d tests) ===\n", N_RSQRT);
    for (ti = 0; ti < N_RSQRT; ti = ti + 1) begin
      @(posedge clk);
      rsqrt_in = tv_rsqrt[ti][31:16];
      rsqrt_valid_in = 1'b1;
      @(posedge clk);
      rsqrt_valid_in = 1'b0;
      @(posedge clk);
      @(posedge clk); #1;
      begin : rsqrt_chk
        reg [15:0] expected;
        expected = tv_rsqrt[ti][15:0];
        if (rsqrt_out !== expected) begin
          $fwrite(fd, "RSQRT [%0d] in=%04x got=%04x exp=%04x FAIL\n",
                  ti, tv_rsqrt[ti][31:16], rsqrt_out, expected);
          errors = errors + 1;
        end else begin
          $fwrite(fd, "RSQRT [%0d] in=%04x got=%04x exp=%04x OK\n",
                  ti, tv_rsqrt[ti][31:16], rsqrt_out, expected);
        end
      end
    end

    // matvec_fp16 test 1: 4x4
    $display("=== matvec_fp16 4x4 ===");
    $fwrite(fd, "=== matvec_fp16 4x4 ===\n");
    for (ri = 0; ri < T1_IN; ri = ri + 1)
      mv1_in[ri*16 +: 16] = mv1_iv[ri];
    @(posedge clk);
    mv1_start = 1'b1;
    @(posedge clk);
    mv1_start = 1'b0;
    wait(mv1_done);
    @(posedge clk); #1;
    for (ri = 0; ri < T1_OUT; ri = ri + 1) begin : mv1_chk
      reg [15:0] got, expected;
      got = mv1_out[ri*16 +: 16];
      expected = mv1_exp[ri];
      if (got !== expected) begin
        $fwrite(fd, "MV1 [%0d] got=%04x exp=%04x FAIL\n", ri, got, expected);
        errors = errors + 1;
      end else begin
        $fwrite(fd, "MV1 [%0d] got=%04x exp=%04x OK\n", ri, got, expected);
      end
    end

    // matvec_fp16 test 2: 8x4
    #20;
    $display("=== matvec_fp16 8x4 ===");
    $fwrite(fd, "=== matvec_fp16 8x4 ===\n");
    for (ri = 0; ri < T2_IN; ri = ri + 1)
      mv2_in[ri*16 +: 16] = mv2_iv[ri];
    @(posedge clk);
    mv2_start = 1'b1;
    @(posedge clk);
    mv2_start = 1'b0;
    wait(mv2_done);
    @(posedge clk); #1;
    for (ri = 0; ri < T2_OUT; ri = ri + 1) begin : mv2_chk
      reg [15:0] got, expected;
      got = mv2_out[ri*16 +: 16];
      expected = mv2_exp[ri];
      if (got !== expected) begin
        $fwrite(fd, "MV2 [%0d] got=%04x exp=%04x FAIL\n", ri, got, expected);
        errors = errors + 1;
      end else begin
        $fwrite(fd, "MV2 [%0d] got=%04x exp=%04x OK\n", ri, got, expected);
      end
    end

    // Summary
    if (errors == 0) begin
      $display("=== All tests passed ===");
      $fwrite(fd, "=== All tests passed ===\n");
    end else begin
      $display("=== %0d FAILURES ===", errors);
      $fwrite(fd, "=== %0d FAILURES ===\n", errors);
    end

    $fclose(fd);
    $finish;
  end

  initial begin
    #200000;
    $display("TIMEOUT");
    $finish;
  end

endmodule