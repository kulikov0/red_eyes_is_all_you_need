`timescale 1ns / 1ps

module tb_attention_stress;

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

  wire [3:0]  w8_sel;
  wire [15:0] w8_addr;
  wire [63:0] w8_data;

  wire [1:0]  k_layer, v_layer;
  wire [2:0]  k_head,  v_head;
  wire [7:0]  k_pos,   v_pos;
  wire [3:0]  k_dim,   v_dim;
  wire        k_we;
  wire [15:0] k_wdata;
  wire [15:0] k_rdata;
  wire        v_we;
  wire [15:0] v_wdata;
  wire [15:0] v_rdata;

  wire done;

  wire [15:0] w8_scale;

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

  weight_store_w8 u_ws_w8 (
    .clk_i          (clk),
    .w8_sel_i       (w8_sel),
    .w8_addr_i      (w8_addr),
    .data_o         (w8_data),
    .scale_o        (w8_scale),
    .tok_emb_addr_i (12'd0),
    .tok_emb_data_o (),
    .tok_emb_scale_o()
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
    .w8_sel_o   (w8_sel),
    .w8_addr_o  (w8_addr),
    .w8_data_i  (w8_data),
    .w8_scale_i (w8_scale),
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
    for (i = 0; i < 128; i = i + 1) begin
      act_reg[i] = 16'd0;
    end

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