`timescale 1ns / 1ps

module tb_kv_cache;

  reg clk = 1'b0;
  always #5 clk = ~clk;

  reg [1:0] layer;
  reg       kv_sel;
  reg [2:0] head;
  reg [7:0] pos;
  reg [3:0] dim;
  reg       we;
  reg [7:0] wdata;
  wire [7:0] rdata;

  kv_cache dut (
    .clk_i   (clk),
    .layer_i (layer),
    .kv_sel_i(kv_sel),
    .head_i  (head),
    .pos_i   (pos),
    .dim_i   (dim),
    .we_i    (we),
    .wdata_i (wdata),
    .rdata_o (rdata)
  );

  integer fd, test_num;

  // Write one byte
  task write_byte;
    input [1:0] l;
    input       kv;
    input [2:0] h;
    input [7:0] p;
    input [3:0] d;
    input [7:0] val;
    begin
      @(posedge clk);
      layer  = l;
      kv_sel = kv;
      head   = h;
      pos    = p;
      dim    = d;
      we     = 1'b1;
      wdata  = val;
      @(posedge clk);
      we = 1'b0;
    end
  endtask

  // Read one byte (set addr, wait 2 cycles for sel_r + BRAM latency, log result)
  task read_byte;
    input integer tnum;
    input [1:0] l;
    input       kv;
    input [2:0] h;
    input [7:0] p;
    input [3:0] d;
    begin
      @(posedge clk);
      layer  = l;
      kv_sel = kv;
      head   = h;
      pos    = p;
      dim    = d;
      we     = 1'b0;
      @(posedge clk);  // DUT latches addr + sel
      @(posedge clk);  // rdata_o valid
      $fwrite(fd, "T=%0d L=%0d KV=%0d H=%0d P=%0d D=%0d OUT=%02x\n",
              tnum, l, kv, h, p, d, rdata);
    end
  endtask

  integer i;

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_kv_cache.log", "w");

    layer  = 2'd0;
    kv_sel = 1'b0;
    head   = 3'd0;
    pos    = 8'd0;
    dim    = 4'd0;
    we     = 1'b0;
    wdata  = 8'd0;

    repeat(3) @(posedge clk);

    $display("=== KV Cache Testbench ===");

    // Test 0: write+read single byte (layer=0, K, head=0, pos=0, dim=0)
    write_byte(2'd0, 1'b0, 3'd0, 8'd0, 4'd0, 8'hA5);
    read_byte(0, 2'd0, 1'b0, 3'd0, 8'd0, 4'd0);
    $display("Test 0 done: single byte");

    // Test 1: write+read full position (16 dims, layer=1, V, head=3, pos=42)
    for (i = 0; i < 16; i = i + 1)
      write_byte(2'd1, 1'b1, 3'd3, 8'd42, i[3:0], i[7:0] + 8'd10);
    for (i = 0; i < 16; i = i + 1)
      read_byte(1, 2'd1, 1'b1, 3'd3, 8'd42, i[3:0]);
    $display("Test 1 done: full position");

    // Test 2: cross-head isolation
    // Write different values to head 0 and head 1 at same (layer, pos, dim)
    write_byte(2'd0, 1'b0, 3'd0, 8'd5, 4'd0, 8'h11);
    write_byte(2'd0, 1'b0, 3'd1, 8'd5, 4'd0, 8'h22);
    read_byte(2, 2'd0, 1'b0, 3'd0, 8'd5, 4'd0);  // expect 0x11
    read_byte(2, 2'd0, 1'b0, 3'd1, 8'd5, 4'd0);  // expect 0x22
    $display("Test 2 done: cross-head isolation");

    // Test 3: K/V isolation
    // Write different values to K and V at same (layer, head, pos, dim)
    write_byte(2'd2, 1'b0, 3'd5, 8'd100, 4'd7, 8'hAA);
    write_byte(2'd2, 1'b1, 3'd5, 8'd100, 4'd7, 8'h55);
    read_byte(3, 2'd2, 1'b0, 3'd5, 8'd100, 4'd7);  // K: expect 0xAA
    read_byte(3, 2'd2, 1'b1, 3'd5, 8'd100, 4'd7);  // V: expect 0x55
    $display("Test 3 done: K/V isolation");

    // Test 4: cross-layer isolation
    // Write different values to same (head, pos, dim) in layer 0 and layer 3
    write_byte(2'd0, 1'b0, 3'd7, 8'd200, 4'd15, 8'hFF);
    write_byte(2'd3, 1'b0, 3'd7, 8'd200, 4'd15, 8'h01);
    read_byte(4, 2'd0, 1'b0, 3'd7, 8'd200, 4'd15);  // layer 0: expect 0xFF
    read_byte(4, 2'd3, 1'b0, 3'd7, 8'd200, 4'd15);  // layer 3: expect 0x01
    $display("Test 4 done: cross-layer isolation");

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