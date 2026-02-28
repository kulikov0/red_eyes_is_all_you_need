`timescale 1ns / 1ps

module tb_layernorm;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst = 1'b0;
  reg start = 1'b0;
  reg signed [7:0] x_data;
  reg [5:0] gamma_sel;

  wire [5:0]  w_sel;
  wire [6:0]  w_addr;
  wire [7:0]  w_data;
  wire signed [7:0] y_data;
  wire y_valid;
  wire done;
  wire busy;

  // Simulate weight_store LN BRAM: 2304 bytes
  reg [7:0] ln_mem [0:2303];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/ln_params.hex", ln_mem);

  // Offset LUT (matches weight_store.v)
  reg [11:0] ln_offset;
  always @(*) begin
    case (w_sel)
      6'd2:  ln_offset = 12'd0;
      6'd3:  ln_offset = 12'd128;
      6'd6:  ln_offset = 12'd256;
      6'd7:  ln_offset = 12'd384;
      6'd10: ln_offset = 12'd512;
      6'd11: ln_offset = 12'd640;
      6'd14: ln_offset = 12'd768;
      6'd15: ln_offset = 12'd896;
      6'd18: ln_offset = 12'd1024;
      6'd19: ln_offset = 12'd1152;
      6'd22: ln_offset = 12'd1280;
      6'd23: ln_offset = 12'd1408;
      6'd26: ln_offset = 12'd1536;
      6'd27: ln_offset = 12'd1664;
      6'd30: ln_offset = 12'd1792;
      6'd31: ln_offset = 12'd1920;
      6'd34: ln_offset = 12'd2048;
      6'd35: ln_offset = 12'd2176;
      default: ln_offset = 12'd0;
    endcase
  end

  // Registered BRAM read (1-cycle latency)
  wire [11:0] ln_addr = ln_offset + {5'd0, w_addr};
  reg [7:0] w_data_r;
  always @(posedge clk) w_data_r <= ln_mem[ln_addr];
  assign w_data = w_data_r;

  // DUT
  layernorm #(
    .DIM(128)
  ) dut (
    .clk_i       (clk),
    .rst_i       (rst),
    .start_i     (start),
    .x_data_i    (x_data),
    .w_sel_o     (w_sel),
    .w_addr_o    (w_addr),
    .w_data_i    (w_data),
    .gamma_sel_i (gamma_sel),
    .y_data_o    (y_data),
    .y_valid_o   (y_valid),
    .done_o      (done),
    .busy_o      (busy)
  );

  // Test input storage
  reg signed [7:0] test_input [0:127];
  integer fd, t, i, out_count;

  // Run one test: load test_input, start layernorm, capture outputs
  task run_test;
    input integer test_num;
    begin
      // Start
      @(posedge clk);
      start = 1'b1;
      @(posedge clk);
      start = 1'b0;

      // Feed input data for 128 cycles
      for (i = 0; i < 128; i = i + 1) begin
        x_data = test_input[i];
        @(posedge clk);
      end
      x_data = 8'sd0;

      // Log after MEAN_ACC -> VAR_ACC transition
      @(posedge clk);
      $fwrite(fd, "T=%0d MEAN_ACC=%0d MEAN=%0d\n",
              test_num, dut.mean_acc, dut.mean);
      $fwrite(fd, "T=%0d XBUF[0]=%0d XBUF[63]=%0d XBUF[127]=%0d\n",
              test_num, dut.x_buf[0], dut.x_buf[63], dut.x_buf[127]);

      // Wait for VAR_ACC to finish
      while (dut.state == 3'd2) @(posedge clk);
      $fwrite(fd, "T=%0d VAR_ACC=%0d ISQRT_IN=%0d\n",
              test_num, dut.var_acc, dut.var_acc[23:7]);
      $fwrite(fd, "T=%0d XBUF_C[0]=%0d XBUF_C[63]=%0d XBUF_C[127]=%0d\n",
              test_num, dut.x_buf[0], dut.x_buf[63], dut.x_buf[127]);

      // Wait for INV_SQRT to finish
      while (dut.state == 3'd3) @(posedge clk);
      $fwrite(fd, "T=%0d INV_STD=%0d (0x%04h)\n",
              test_num, dut.inv_std, dut.inv_std);

      // Wait for LOAD_GAMMA to finish
      while (dut.state == 3'd4) @(posedge clk);
      $fwrite(fd, "T=%0d GAMMA[0]=%0d GAMMA[63]=%0d GAMMA[127]=%0d\n",
              test_num, dut.gamma_buf[0], dut.gamma_buf[63], dut.gamma_buf[127]);

      // Wait for LOAD_BETA to finish
      while (dut.state == 3'd5) @(posedge clk);
      $fwrite(fd, "T=%0d BETA[0]=%0d BETA[63]=%0d BETA[127]=%0d\n",
              test_num, dut.beta_buf[0], dut.beta_buf[63], dut.beta_buf[127]);

      // Capture NORM outputs
      out_count = 0;
      while (!done) begin
        if (y_valid) begin
          if (out_count < 3) begin
            $fwrite(fd, "T=%0d NORM[%0d] diff=%0d full_prod=%0d bias=%0d\n",
                    test_num, out_count,
                    dut.norm_blk.diff,
                    dut.norm_blk.full_prod,
                    dut.norm_blk.biased);
          end
          $fwrite(fd, "T=%0d OUT[%0d]=%0d\n", test_num, out_count, y_data);
          out_count = out_count + 1;
        end
        @(posedge clk);
      end
      // Capture any remaining valid on the done cycle
      if (y_valid) begin
        $fwrite(fd, "T=%0d OUT[%0d]=%0d\n", test_num, out_count, y_data);
        out_count = out_count + 1;
      end

      $display("Test %0d done: %0d outputs", test_num, out_count);
      @(posedge clk);
    end
  endtask

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_layernorm.log", "w");

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    // Test 0: ramp input (0..127) with block0_ln1 (gamma=sel 2, beta=sel 3)
    $display("=== Test 0: ramp ===");
    $fwrite(fd, "=== Test 0: ramp ===\n");
    gamma_sel = 6'd2;
    for (i = 0; i < 128; i = i + 1) test_input[i] = i[7:0];
    run_test(0);

    // Test 1: constant input (all 42) - zero variance case
    $display("=== Test 1: constant ===");
    $fwrite(fd, "=== Test 1: constant ===\n");
    gamma_sel = 6'd2;
    for (i = 0; i < 128; i = i + 1) test_input[i] = 8'sd42;
    run_test(1);

    // Test 2: signed ramp (-64..63) with block0_ln2 (gamma=sel 6, beta=sel 7)
    $display("=== Test 2: signed ramp ===");
    $fwrite(fd, "=== Test 2: signed ramp ===\n");
    gamma_sel = 6'd6;
    for (i = 0; i < 128; i = i + 1) test_input[i] = (i - 64);
    run_test(2);

    // Test 3: alternating +/- with block1_ln1 (gamma=sel 10, beta=sel 11)
    $display("=== Test 3: alternating ===");
    $fwrite(fd, "=== Test 3: alternating ===\n");
    gamma_sel = 6'd10;
    for (i = 0; i < 128; i = i + 1) begin
      if (i % 2 == 0) test_input[i] = 8'sd50;
      else test_input[i] = -8'sd50;
    end
    run_test(3);

    // Test 4: ramp with ln_f (gamma=sel 34, beta=sel 35)
    $display("=== Test 4: ramp with ln_f ===");
    $fwrite(fd, "=== Test 4: ramp with ln_f ===\n");
    gamma_sel = 6'd34;
    for (i = 0; i < 128; i = i + 1) test_input[i] = i[7:0];
    run_test(4);

    $display("=== All tests done ===");
    $fwrite(fd, "=== All tests done ===\n");

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
