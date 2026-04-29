`timescale 1ns / 1ps

module tb_attention;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst = 1'b0;
  reg start = 1'b0;
  reg [1:0] layer;
  reg [7:0] pos;
  reg  [15:0] act_reg [0:127];
  reg  [15:0] res_reg [0:127];
  wire [6:0]  act_raddr;
  wire        res_we;
  wire [6:0]  res_waddr;
  wire [15:0] res_wdata;

  always @(posedge clk) begin
    if (res_we)
      res_reg[res_waddr] <= res_wdata;
  end

  wire [5:0]  w_sel;
  wire [15:0] w_addr;
  wire [7:0]  w_data;

  wire [1:0]  k_layer, v_layer;
  wire [2:0]  k_head, v_head;
  wire [7:0]  k_pos, v_pos;
  wire [3:0]  k_dim, v_dim;
  wire        k_we;
  wire [15:0] k_wdata;
  wire [15:0] k_rdata;
  wire        v_we;
  wire [15:0] v_wdata;
  wire [15:0] v_rdata;

  // KV boundary register: mirrors transformer_top's reg between DUT and cache
  reg [1:0]  k_layer_r, v_layer_r;
  reg [2:0]  k_head_r,  v_head_r;
  reg [7:0]  k_pos_r,   v_pos_r;
  reg [3:0]  k_dim_r,   v_dim_r;
  reg        k_we_r;
  reg [15:0] k_wdata_r;
  reg        v_we_r;
  reg [15:0] v_wdata_r;
  always @(posedge clk) begin
    k_layer_r <= k_layer;
    k_head_r  <= k_head;
    k_pos_r   <= k_pos;
    k_dim_r   <= k_dim;
    k_we_r    <= k_we;
    k_wdata_r <= k_wdata;
    v_layer_r <= v_layer;
    v_head_r  <= v_head;
    v_pos_r   <= v_pos;
    v_dim_r   <= v_dim;
    v_we_r    <= v_we;
    v_wdata_r <= v_wdata;
  end

  wire done;

  wire [15:0] w_scale;

  // Weight store
  weight_store u_ws (
    .clk_i       (clk),
    .tensor_sel_i(w_sel),
    .addr_i      (w_addr),
    .data_o      (w_data),
    .scale_o     (w_scale)
  );

  // K cache (fp16, DATA_W=16)
  kv_cache #(.DATA_W(16)) u_k_cache (
    .clk_i  (clk),
    .layer_i(k_layer_r),
    .head_i (k_head_r),
    .pos_i  (k_pos_r),
    .dim_i  (k_dim_r),
    .we_i   (k_we_r),
    .wdata_i(k_wdata_r),
    .rdata_o(k_rdata)
  );

  // V cache (fp16, DATA_W=16)
  kv_cache #(.DATA_W(16)) u_v_cache (
    .clk_i  (clk),
    .layer_i(v_layer_r),
    .head_i (v_head_r),
    .pos_i  (v_pos_r),
    .dim_i  (v_dim_r),
    .we_i   (v_we_r),
    .wdata_i(v_wdata_r),
    .rdata_o(v_rdata)
  );

  // DUT
  attention dut (
    .clk_i     (clk),
    .rst_i     (rst),
    .start_i    (start),
    .layer_i    (layer),
    .pos_i      (pos),
    .act_raddr_o(act_raddr),
    .act_rdata_i(act_reg[act_raddr]),
    .res_we_o   (res_we),
    .res_waddr_o(res_waddr),
    .res_wdata_o(res_wdata),
    .w_sel_o    (w_sel),
    .w_addr_o   (w_addr),
    .w_data_i   (w_data),
    .w_scale_i  (w_scale),
    .k_we_o     (k_we),
    .k_wdata_o  (k_wdata),
    .k_layer_o  (k_layer),
    .k_head_o   (k_head),
    .k_pos_o    (k_pos),
    .k_dim_o    (k_dim),
    .k_rdata_i  (k_rdata),
    .v_we_o     (v_we),
    .v_wdata_o  (v_wdata),
    .v_layer_o  (v_layer),
    .v_head_o   (v_head),
    .v_pos_o    (v_pos),
    .v_dim_o    (v_dim),
    .v_rdata_i  (v_rdata),
    .done_o     (done)
  );

  integer fd, i;

  // int8 -> fp16 conversion (matches fp16_from_int8.v)
  function [15:0] int8_to_fp16;
    input [7:0] val;
    reg [7:0] abs_v;
    reg is_neg;
    reg [3:0] lod;
    reg [4:0] exp_v;
    reg [9:0] mant_v;
    integer ii;
    begin
      is_neg = val[7];
      abs_v = is_neg ? (~val + 8'd1) : val;
      if (abs_v == 8'd0) begin
        int8_to_fp16 = 16'd0;
      end else begin
        lod = 4'd0;
        for (ii = 0; ii < 8; ii = ii + 1)
          if (abs_v[ii]) lod = ii[3:0];
        exp_v = {1'b0, lod} + 5'd15;
        mant_v = ({2'b0, abs_v} << (4'd10 - lod));
        mant_v = mant_v & 10'h3FF;
        int8_to_fp16 = {is_neg, exp_v, mant_v};
      end
    end
  endfunction

  // Build fp16 input vector from seed (seed+k as int8 -> fp16)
  task build_input;
    input integer seed;
    integer k;
    reg [7:0] val;
    begin
      for (k = 0; k < 128; k = k + 1) begin
        val = (seed + k) & 8'hFF;
        act_reg[k] = int8_to_fp16(val);
      end
    end
  endtask

  // Run one attention pass and log fp16 output
  task run_test;
    input integer test_num;
    input [1:0]  t_layer;
    input [7:0]  t_pos;
    input integer seed;
    begin
      build_input(seed);
      layer = t_layer;
      pos   = t_pos;

      @(posedge clk);
      start = 1'b1;
      @(posedge clk);
      start = 1'b0;

      while (!done) @(posedge clk);

      $fwrite(fd, "TEST %0d LAYER=%0d POS=%0d SEED=%0d\n",
              test_num, t_layer, t_pos, seed);
      for (i = 0; i < 128; i = i + 1) begin
        $fwrite(fd, "OUT[%0d]=%04x\n", i, res_reg[i]);
      end

      $display("Test %0d done: layer=%0d pos=%0d seed=%0d",
               test_num, t_layer, t_pos, seed);
      repeat(5) @(posedge clk);
    end
  endtask

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_attention.log", "w");

    layer = 2'd0;
    pos   = 8'd0;
    for (i = 0; i < 128; i = i + 1) begin
      act_reg[i] = 16'd0;
    end

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    $display("=== Attention FP16 Testbench ===");

    // Test 0: pos=0, single token (trivial self-attention)
    run_test(0, 2'd0, 8'd0, 192);

    // Test 1: pos=1, two tokens
    run_test(1, 2'd0, 8'd1, 64);

    // Test 2: pos=2, three tokens
    run_test(2, 2'd0, 8'd2, 128);

    $display("=== All 3 tests done ===");

    $fclose(fd);
    $finish;
  end

  // Timeout
  initial begin
    #150_000_000;
    $display("TIMEOUT");
    $finish;
  end

endmodule