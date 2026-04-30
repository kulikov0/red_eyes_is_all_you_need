`timescale 1ns / 1ps

// Unified FP16 testbench: add, mul, reduce_k4, from_int8, to_int8, to_q167, q115_to_fp16, rsqrt, matvec
module tb_fp16;

  reg clk;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer errors, fd;
  reg rst;
  integer ti, ri;

  // fp16_add: 3-cycle pipelined
  reg         add_valid_in;
  reg  [15:0] add_a, add_b;
  wire        add_valid_out;
  wire [15:0] add_sum;
  fp16_add u_add (
    .clk_i(clk), .valid_i(add_valid_in),
    .a_i(add_a), .b_i(add_b),
    .valid_o(add_valid_out), .sum_o(add_sum)
  );

  localparam N_ADD = 50;
  reg [47:0] tv_add [0:N_ADD-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_add_vectors.hex", tv_add);

  // fp16_mul: 2-cycle pipelined
  reg         mul_valid_in;
  reg  [15:0] mul_a, mul_b;
  wire        mul_valid_out;
  wire [15:0] mul_prod;
  fp16_mul u_mul (
    .clk_i(clk), .valid_i(mul_valid_in),
    .a_i(mul_a), .b_i(mul_b),
    .valid_o(mul_valid_out), .prod_o(mul_prod)
  );

  localparam N_MUL = 48;
  reg [47:0] tv_mul [0:N_MUL-1];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_mul_vectors.hex", tv_mul);

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

  // fp16_reduce_k4: streams 16 fp16 values per test, computes K=4 tree reduce
  reg         red_clear;
  reg         red_valid;
  reg         red_flush;
  reg  [15:0] red_data;
  wire        red_done;
  wire [15:0] red_sum;
  fp16_reduce_k4 u_reduce (
    .clk_i  (clk), .rst_i(rst),
    .clear_i(red_clear), .valid_i(red_valid), .data_i(red_data),
    .flush_i(red_flush),
    .done_o (red_done), .sum_o(red_sum)
  );

  localparam N_REDUCE_TESTS = 10;
  localparam N_REDUCE_VALS  = 16;
  reg [15:0] tv_reduce_in  [0:N_REDUCE_TESTS*N_REDUCE_VALS-1];
  reg [15:0] tv_reduce_exp [0:N_REDUCE_TESTS-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_reduce_k4_inputs.hex", tv_reduce_in);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/fp16_reduce_k4_expected.hex", tv_reduce_exp);
  end

  // matvec_fp16 test 1: 4x4
  localparam T1_IN = 4, T1_OUT = 4;
  reg                              mv1_start;
  reg  [15:0]                     mv1_scale;
  wire [$clog2(T1_OUT*T1_IN)-1:0] mv1_waddr;
  wire                            mv1_done;
  reg signed [7:0] mv1_wmem [0:T1_OUT*T1_IN-1];
  reg signed [7:0] mv1_wdata;
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_4x4_weights.hex",
              mv1_wmem);
  end
  // Mimic weight_store: addr boundary reg followed by sync BRAM read
  reg [$clog2(T1_OUT*T1_IN)-1:0] mv1_waddr_r;
  always @(posedge clk) begin
    mv1_waddr_r <= mv1_waddr;
    mv1_wdata   <= mv1_wmem[mv1_waddr_r];
  end

  reg  [15:0] mv1_act [0:T1_IN-1];
  reg  [15:0] mv1_res [0:T1_OUT-1];
  wire [$clog2(T1_IN)-1:0]  mv1_raddr;
  wire                      mv1_rwe;
  wire [$clog2(T1_OUT)-1:0] mv1_rwaddr;
  wire [15:0]               mv1_rwdata;

  matvec_fp16 #(.IN_DIM(T1_IN), .OUT_DIM(T1_OUT)) u_mv1 (
    .clk_i(clk), .rst_i(rst), .start_i(mv1_start),
    .scale_i(mv1_scale),
    .weight_addr_o(mv1_waddr), .weight_data_i(mv1_wdata),
    .act_raddr_o(mv1_raddr), .act_rdata_i(mv1_act[mv1_raddr]),
    .res_we_o(mv1_rwe), .res_waddr_o(mv1_rwaddr), .res_wdata_o(mv1_rwdata),
    .done_o(mv1_done)
  );
  always @(posedge clk) if (mv1_rwe) mv1_res[mv1_rwaddr] <= mv1_rwdata;

  reg [15:0] mv1_iv [0:T1_IN-1];
  reg [15:0] mv1_exp [0:T1_OUT-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_4x4_input.hex", mv1_iv);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_4x4_expected.hex", mv1_exp);
  end

  // matvec_fp16 test 2: 8x4
  localparam T2_IN = 4, T2_OUT = 8;
  reg                              mv2_start;
  reg  [15:0]                     mv2_scale;
  wire [$clog2(T2_OUT*T2_IN)-1:0] mv2_waddr;
  wire                            mv2_done;
  reg signed [7:0] mv2_wmem [0:T2_OUT*T2_IN-1];
  reg signed [7:0] mv2_wdata;
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_8x4_weights.hex",
              mv2_wmem);
  end
  reg [$clog2(T2_OUT*T2_IN)-1:0] mv2_waddr_r;
  always @(posedge clk) begin
    mv2_waddr_r <= mv2_waddr;
    mv2_wdata   <= mv2_wmem[mv2_waddr_r];
  end

  reg  [15:0] mv2_act [0:T2_IN-1];
  reg  [15:0] mv2_res [0:T2_OUT-1];
  wire [$clog2(T2_IN)-1:0]  mv2_raddr;
  wire                      mv2_rwe;
  wire [$clog2(T2_OUT)-1:0] mv2_rwaddr;
  wire [15:0]               mv2_rwdata;

  matvec_fp16 #(.IN_DIM(T2_IN), .OUT_DIM(T2_OUT)) u_mv2 (
    .clk_i(clk), .rst_i(rst), .start_i(mv2_start),
    .scale_i(mv2_scale),
    .weight_addr_o(mv2_waddr), .weight_data_i(mv2_wdata),
    .act_raddr_o(mv2_raddr), .act_rdata_i(mv2_act[mv2_raddr]),
    .res_we_o(mv2_rwe), .res_waddr_o(mv2_rwaddr), .res_wdata_o(mv2_rwdata),
    .done_o(mv2_done)
  );
  always @(posedge clk) if (mv2_rwe) mv2_res[mv2_rwaddr] <= mv2_rwdata;

  reg [15:0] mv2_iv [0:T2_IN-1];
  reg [15:0] mv2_exp [0:T2_OUT-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_8x4_input.hex", mv2_iv);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_8x4_expected.hex", mv2_exp);
  end

  // matvec_fp16_w8 test 3: 32x4 packed
  localparam T3_IN = 4, T3_OUT = 32;
  localparam T3_WORDS = (T3_OUT / 8) * T3_IN;
  reg                            mv3_start;
  reg  [15:0]                    mv3_scale;
  wire [$clog2(T3_WORDS)-1:0]    mv3_waddr;
  wire                           mv3_done;
  reg  [63:0] mv3_wmem [0:T3_WORDS-1];
  reg  [63:0] mv3_wdata;
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_w8_32x4_weights.hex",
              mv3_wmem);
  end
  reg [$clog2(T3_WORDS)-1:0] mv3_waddr_r;
  always @(posedge clk) begin
    mv3_waddr_r <= mv3_waddr;
    mv3_wdata   <= mv3_wmem[mv3_waddr_r];
  end

  reg  [15:0] mv3_act [0:T3_IN-1];
  reg  [15:0] mv3_res [0:T3_OUT-1];
  wire [$clog2(T3_IN)-1:0]  mv3_raddr;
  wire                      mv3_rwe;
  wire [$clog2(T3_OUT)-1:0] mv3_rwaddr;
  wire [15:0]               mv3_rwdata;

  matvec_fp16_w8 #(.IN_DIM(T3_IN), .OUT_DIM(T3_OUT)) u_mv3 (
    .clk_i(clk), .rst_i(rst), .start_i(mv3_start),
    .scale_i(mv3_scale),
    .weight_addr_o(mv3_waddr), .weight_data_i(mv3_wdata),
    .act_raddr_o(mv3_raddr), .act_rdata_i(mv3_act[mv3_raddr]),
    .res_we_o(mv3_rwe), .res_waddr_o(mv3_rwaddr), .res_wdata_o(mv3_rwdata),
    .done_o(mv3_done)
  );
  always @(posedge clk) if (mv3_rwe) mv3_res[mv3_rwaddr] <= mv3_rwdata;

  reg [15:0] mv3_iv [0:T3_IN-1];
  reg [15:0] mv3_exp [0:T3_OUT-1];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_w8_32x4_input.hex", mv3_iv);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/matvec_fp16_w8_32x4_expected.hex", mv3_exp);
  end

  initial begin
    rst = 1'b1;
    add_valid_in = 1'b0; add_a = 16'd0; add_b = 16'd0;
    mul_valid_in = 1'b0; mul_a = 16'd0; mul_b = 16'd0;
    cvt_in = 8'd0; to_in = 16'd0; q167_in = 16'd0; q115_in = 16'd0;
    rsqrt_valid_in = 1'b0; rsqrt_in = 16'd0;
    red_clear = 1'b0; red_valid = 1'b0; red_flush = 1'b0; red_data = 16'd0;
    mv1_start = 1'b0; mv2_start = 1'b0; mv3_start = 1'b0;
    mv1_scale = 16'h2c00; mv2_scale = 16'h2c00; mv3_scale = 16'h2c00;
    errors = 0;

    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_fp16.log", "w");

    #30;
    @(posedge clk);
    rst = 1'b0;
    #10;

    // fp16_add: 3-cycle pipelined, drive valid high, wait 3 clocks for output
    $display("=== fp16_add (%0d tests) ===", N_ADD);
    $fwrite(fd, "=== fp16_add (%0d tests) ===\n", N_ADD);
    add_valid_in = 1'b1;
    for (ti = 0; ti < N_ADD; ti = ti + 1) begin
      @(posedge clk);
      add_a = tv_add[ti][47:32];
      add_b = tv_add[ti][31:16];
      @(posedge clk);
      @(posedge clk);
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
    add_valid_in = 1'b0;

    // fp16_mul: 2-cycle pipelined, drive valid high, wait 2 clocks for output
    $display("=== fp16_mul (%0d tests) ===", N_MUL);
    $fwrite(fd, "=== fp16_mul (%0d tests) ===\n", N_MUL);
    mul_valid_in = 1'b1;
    for (ti = 0; ti < N_MUL; ti = ti + 1) begin
      @(posedge clk);
      mul_a = tv_mul[ti][47:32];
      mul_b = tv_mul[ti][31:16];
      @(posedge clk);
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
    mul_valid_in = 1'b0;

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

    // fp16_reduce_k4: stream 16 inputs per test, check final sum
    $display("=== fp16_reduce_k4 (%0d tests) ===", N_REDUCE_TESTS);
    $fwrite(fd, "=== fp16_reduce_k4 (%0d tests) ===\n", N_REDUCE_TESTS);
    for (ti = 0; ti < N_REDUCE_TESTS; ti = ti + 1) begin : red_loop
      reg [15:0] expected;
      integer k;
      @(posedge clk); #1;
      red_clear = 1'b1;
      @(posedge clk); #1;
      red_clear = 1'b0;
      for (k = 0; k < N_REDUCE_VALS; k = k + 1) begin
        @(posedge clk); #1;
        red_valid = 1'b1;
        red_data  = tv_reduce_in[ti*N_REDUCE_VALS + k];
        red_flush = (k == N_REDUCE_VALS - 1) ? 1'b1 : 1'b0;
      end
      @(posedge clk); #1;
      red_valid = 1'b0;
      red_flush = 1'b0;
      wait(red_done);
      @(posedge clk); #1;
      expected = tv_reduce_exp[ti];
      if (red_sum !== expected) begin
        $fwrite(fd, "REDUCE [%0d] got=%04x exp=%04x FAIL\n", ti, red_sum, expected);
        errors = errors + 1;
      end else begin
        $fwrite(fd, "REDUCE [%0d] got=%04x exp=%04x OK\n", ti, red_sum, expected);
      end
    end

    // matvec_fp16 test 1: 4x4
    $display("=== matvec_fp16 4x4 ===");
    $fwrite(fd, "=== matvec_fp16 4x4 ===\n");
    for (ri = 0; ri < T1_IN; ri = ri + 1)
      mv1_act[ri] = mv1_iv[ri];
    @(posedge clk);
    mv1_start = 1'b1;
    @(posedge clk);
    mv1_start = 1'b0;
    wait(mv1_done);
    @(posedge clk); #1;
    for (ri = 0; ri < T1_OUT; ri = ri + 1) begin : mv1_chk
      reg [15:0] got, expected;
      got = mv1_res[ri];
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
      mv2_act[ri] = mv2_iv[ri];
    @(posedge clk);
    mv2_start = 1'b1;
    @(posedge clk);
    mv2_start = 1'b0;
    wait(mv2_done);
    @(posedge clk); #1;
    for (ri = 0; ri < T2_OUT; ri = ri + 1) begin : mv2_chk
      reg [15:0] got, expected;
      got = mv2_res[ri];
      expected = mv2_exp[ri];
      if (got !== expected) begin
        $fwrite(fd, "MV2 [%0d] got=%04x exp=%04x FAIL\n", ri, got, expected);
        errors = errors + 1;
      end else begin
        $fwrite(fd, "MV2 [%0d] got=%04x exp=%04x OK\n", ri, got, expected);
      end
    end

    // matvec_fp16_w8 test 3: 32x4 packed
    #20;
    $display("=== matvec_fp16_w8 32x4 ===");
    $fwrite(fd, "=== matvec_fp16_w8 32x4 ===\n");
    for (ri = 0; ri < T3_IN; ri = ri + 1)
      mv3_act[ri] = mv3_iv[ri];
    @(posedge clk);
    mv3_start = 1'b1;
    @(posedge clk);
    mv3_start = 1'b0;
    wait(mv3_done);
    @(posedge clk); #1;
    for (ri = 0; ri < T3_OUT; ri = ri + 1) begin : mv3_chk
      reg [15:0] got, expected;
      got = mv3_res[ri];
      expected = mv3_exp[ri];
      if (got !== expected) begin
        $fwrite(fd, "MV3 [%0d] got=%04x exp=%04x FAIL\n", ri, got, expected);
        errors = errors + 1;
      end else begin
        $fwrite(fd, "MV3 [%0d] got=%04x exp=%04x OK\n", ri, got, expected);
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