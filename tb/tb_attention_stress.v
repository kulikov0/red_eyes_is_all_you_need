`timescale 1ns / 1ps

module tb_attention_stress;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst = 1'b0;
  reg start = 1'b0;
  reg [1:0] layer;
  reg [7:0] pos;
  reg [2047:0] x_in;

  // Attention <-> weight store
  wire [5:0]  w_sel;
  wire [15:0] w_addr;
  wire [7:0]  w_data;

  // Attention <-> KV caches (both fp16)
  wire [1:0]  kv_layer;
  wire [2:0]  kv_head;
  wire [7:0]  kv_pos;
  wire [3:0]  kv_dim;
  wire        k_we;
  wire [15:0] k_wdata;
  wire [15:0] k_rdata;
  wire        v_we;
  wire [15:0] v_wdata;
  wire [15:0] v_rdata;

  wire [2047:0] out_vec;
  wire done;

  weight_store u_ws (
    .clk_i       (clk),
    .tensor_sel_i(w_sel),
    .addr_i      (w_addr),
    .data_o      (w_data),
    .scale_o     ()
  );

  // K cache (fp16, DATA_W=16)
  kv_cache #(.DATA_W(16)) u_k_cache (
    .clk_i  (clk),
    .layer_i(kv_layer),
    .head_i (kv_head),
    .pos_i  (kv_pos),
    .dim_i  (kv_dim),
    .we_i   (k_we),
    .wdata_i(k_wdata),
    .rdata_o(k_rdata)
  );

  // V cache (fp16, DATA_W=16)
  kv_cache #(.DATA_W(16)) u_v_cache (
    .clk_i  (clk),
    .layer_i(kv_layer),
    .head_i (kv_head),
    .pos_i  (kv_pos),
    .dim_i  (kv_dim),
    .we_i   (v_we),
    .wdata_i(v_wdata),
    .rdata_o(v_rdata)
  );

  attention dut (
    .clk_i     (clk),
    .rst_i     (rst),
    .start_i   (start),
    .layer_i   (layer),
    .pos_i     (pos),
    .x_i       (x_in),
    .w_sel_o   (w_sel),
    .w_addr_o  (w_addr),
    .w_data_i  (w_data),
    .k_we_o    (k_we),
    .k_wdata_o (k_wdata),
    .k_rdata_i (k_rdata),
    .v_we_o    (v_we),
    .v_wdata_o (v_wdata),
    .v_rdata_i (v_rdata),
    .kv_layer_o(kv_layer),
    .kv_head_o (kv_head),
    .kv_pos_o  (kv_pos),
    .kv_dim_o  (kv_dim),
    .out_vec_o (out_vec),
    .done_o    (done)
  );

  integer fd, i, t;

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
        x_in[k*16 +: 16] = int8_to_fp16(val);
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
        $fwrite(fd, "OUT[%0d]=%04x\n", i, out_vec[i*16 +: 16]);
      end

      if (test_num % 50 == 0)
        $display("Test %0d done: pos=%0d seed=%0d", test_num, t_pos, seed);

      repeat(2) @(posedge clk);
    end
  endtask

  integer seed;

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_attention_stress.log", "w");

    layer = 2'd0;
    pos   = 8'd0;
    x_in  = {2048{1'b0}};

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    $display("=== Attention Stress Test: 300 positions ===");

    // Run 300 sequential positions on layer 0
    for (t = 0; t < 300; t = t + 1) begin
      seed = ((t * 73 + 17) & 8'hFF);
      run_test(t, 2'd0, t[7:0], seed);
    end

    $display("=== All 300 tests done ===");

    $fclose(fd);
    $finish;
  end

  // Timeout: 2 billion ns (~200M cycles)
  initial begin
    #2_000_000_000;
    $display("TIMEOUT");
    $finish;
  end

endmodule