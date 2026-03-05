`timescale 1ns / 1ps

module tb_kv_cache;

  reg clk = 1'b0;
  always #5 clk = ~clk;

  // Shared address
  reg [1:0] layer;
  reg [2:0] head;
  reg [7:0] pos;
  reg [3:0] dim;

  // K cache (fp16, 16-bit)
  reg         k_we;
  reg  [15:0] k_wdata;
  wire [15:0] k_rdata;

  // V cache (fp16, 16-bit)
  reg         v_we;
  reg  [15:0] v_wdata;
  wire [15:0] v_rdata;

  kv_cache #(.DATA_W(16)) dut_k (
    .clk_i  (clk),
    .layer_i(layer),
    .head_i (head),
    .pos_i  (pos),
    .dim_i  (dim),
    .we_i   (k_we),
    .wdata_i(k_wdata),
    .rdata_o(k_rdata)
  );

  kv_cache #(.DATA_W(16)) dut_v (
    .clk_i  (clk),
    .layer_i(layer),
    .head_i (head),
    .pos_i  (pos),
    .dim_i  (dim),
    .we_i   (v_we),
    .wdata_i(v_wdata),
    .rdata_o(v_rdata)
  );

  integer fd, i;

  // Write one K entry (fp16)
  task write_k;
    input [1:0]  l;
    input [2:0]  h;
    input [7:0]  p;
    input [3:0]  d;
    input [15:0] val;
    begin
      @(posedge clk);
      layer   = l;
      head    = h;
      pos     = p;
      dim     = d;
      k_we    = 1'b1;
      k_wdata = val;
      @(posedge clk);
      k_we = 1'b0;
    end
  endtask

  // Write one V entry (fp16)
  task write_v;
    input [1:0]  l;
    input [2:0]  h;
    input [7:0]  p;
    input [3:0]  d;
    input [15:0] val;
    begin
      @(posedge clk);
      layer   = l;
      head    = h;
      pos     = p;
      dim     = d;
      v_we    = 1'b1;
      v_wdata = val;
      @(posedge clk);
      v_we = 1'b0;
    end
  endtask

  // Read one K entry (2-cycle latency), log result
  task read_k;
    input integer tnum;
    input [1:0] l;
    input [2:0] h;
    input [7:0] p;
    input [3:0] d;
    begin
      @(posedge clk);
      layer = l;
      head  = h;
      pos   = p;
      dim   = d;
      k_we  = 1'b0;
      @(posedge clk);
      @(posedge clk);
      $fwrite(fd, "T=%0d C=K L=%0d H=%0d P=%0d D=%0d OUT=%04x\n",
              tnum, l, h, p, d, k_rdata);
    end
  endtask

  // Read one V entry (2-cycle latency), log result
  task read_v;
    input integer tnum;
    input [1:0] l;
    input [2:0] h;
    input [7:0] p;
    input [3:0] d;
    begin
      @(posedge clk);
      layer = l;
      head  = h;
      pos   = p;
      dim   = d;
      v_we  = 1'b0;
      @(posedge clk);
      @(posedge clk);
      $fwrite(fd, "T=%0d C=V L=%0d H=%0d P=%0d D=%0d OUT=%04x\n",
              tnum, l, h, p, d, v_rdata);
    end
  endtask

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_kv_cache.log", "w");

    layer   = 2'd0;
    head    = 3'd0;
    pos     = 8'd0;
    dim     = 4'd0;
    k_we    = 1'b0;
    k_wdata = 16'd0;
    v_we    = 1'b0;
    v_wdata = 16'd0;

    repeat(3) @(posedge clk);

    $display("=== KV Cache Testbench ===");

    // Test 0: write+read single V fp16 (layer=0, head=0, pos=0, dim=0)
    write_v(2'd0, 3'd0, 8'd0, 4'd0, 16'h3C00);  // 1.0 in fp16
    read_v(0, 2'd0, 3'd0, 8'd0, 4'd0);
    $display("Test 0 done: single V fp16");

    // Test 1: write+read full position V (16 dims, layer=1, head=3, pos=42)
    for (i = 0; i < 16; i = i + 1)
      write_v(2'd1, 3'd3, 8'd42, i[3:0], 16'h3C00 + i[15:0]);
    for (i = 0; i < 16; i = i + 1)
      read_v(1, 2'd1, 3'd3, 8'd42, i[3:0]);
    $display("Test 1 done: full position V");

    // Test 2: cross-head isolation in K cache (fp16)
    write_k(2'd0, 3'd0, 8'd5, 4'd0, 16'h4011);
    write_k(2'd0, 3'd1, 8'd5, 4'd0, 16'h4022);
    read_k(2, 2'd0, 3'd0, 8'd5, 4'd0);  // expect 0x4011
    read_k(2, 2'd0, 3'd1, 8'd5, 4'd0);  // expect 0x4022
    $display("Test 2 done: cross-head K isolation");

    // Test 3: K/V cache isolation (write to both at same address)
    write_k(2'd2, 3'd5, 8'd100, 4'd7, 16'hCAFE);
    write_v(2'd2, 3'd5, 8'd100, 4'd7, 16'h5500);
    read_k(3, 2'd2, 3'd5, 8'd100, 4'd7);  // K: expect 0xCAFE
    read_v(3, 2'd2, 3'd5, 8'd100, 4'd7);  // V: expect 0x5500
    $display("Test 3 done: K/V cache isolation");

    // Test 4: cross-layer isolation in K cache
    write_k(2'd0, 3'd7, 8'd200, 4'd15, 16'hFFFF);
    write_k(2'd3, 3'd7, 8'd200, 4'd15, 16'h0001);
    read_k(4, 2'd0, 3'd7, 8'd200, 4'd15);  // layer 0: expect 0xFFFF
    read_k(4, 2'd3, 3'd7, 8'd200, 4'd15);  // layer 3: expect 0x0001
    $display("Test 4 done: cross-layer K isolation");

    $display("=== All 5 tests done ===");
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