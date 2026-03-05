`timescale 1ns / 1ps

module tb_layernorm;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst = 1'b0;
  reg start = 1'b0;
  reg [2047:0] x_data;
  reg [5:0] gamma_sel;
  reg [15:0] gamma_scale;
  reg [15:0] beta_scale;

  wire [5:0]  w_sel;
  wire [6:0]  w_addr;
  wire [7:0]  w_data;
  wire [2047:0] y_data;
  wire done;

  // Test input vectors: 8 tests x 128 fp16 = 1024 entries
  reg [15:0] test_mem [0:1023];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/ln_test_inputs.hex", test_mem);

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

  // FP16 dequant scales from weight_scales.vh
  `include "weight_scales.vh"

  // DUT
  layernorm #(
    .DIM(128)
  ) dut (
    .clk_i         (clk),
    .rst_i         (rst),
    .start_i       (start),
    .x_i           (x_data),
    .w_sel_o       (w_sel),
    .w_addr_o      (w_addr),
    .w_data_i      (w_data),
    .gamma_sel_i   (gamma_sel),
    .gamma_scale_i (gamma_scale),
    .beta_scale_i  (beta_scale),
    .y_o           (y_data),
    .done_o        (done),
    .busy_o        ()
  );

  integer fd, i;

  task load_test_input;
    input integer test_num;
    integer base;
    begin
      base = test_num * 128;
      for (i = 0; i < 128; i = i + 1)
        x_data[i*16 +: 16] = test_mem[base + i];
    end
  endtask

  task run_test;
    input integer test_num;
    begin
      load_test_input(test_num);
      @(posedge clk);
      start = 1'b1;
      @(posedge clk);
      start = 1'b0;

      // Wait for done
      while (!done) @(posedge clk);

      // Log intermediate values
      $fwrite(fd, "T=%0d NEG_MEAN=%04x VAR_ACC=%04x INV_STD=%04x\n",
              test_num, dut.neg_mean, dut.var_acc, dut.inv_std);

      // Log outputs
      for (i = 0; i < 128; i = i + 1) begin
        $fwrite(fd, "T=%0d I=%0d OUT=%04x\n",
                test_num, i, y_data[i*16 +: 16]);
      end

      $display("Test %0d done", test_num);
      @(posedge clk);
    end
  endtask

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_layernorm.log", "w");

    x_data = {2048{1'b0}};

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    $display("=== LayerNorm FP16 Testbench ===");

    // Test 0: embed(0,0) block0_ln1 - typical small values
    $display("Test 0: embed(0,0) block0_ln1");
    $fwrite(fd, "Test 0: embed(0,0) block0_ln1\n");
    gamma_sel = 6'd2;
    gamma_scale = SCALE_BLOCK0_LN1_WEIGHT;
    beta_scale = SCALE_BLOCK0_LN1_BIAS;
    run_test(0);

    // Test 1: embed(42,10) block0_ln2 - different embedding
    $display("Test 1: embed(42,10) block0_ln2");
    $fwrite(fd, "Test 1: embed(42,10) block0_ln2\n");
    gamma_sel = 6'd6;
    gamma_scale = SCALE_BLOCK0_LN2_WEIGHT;
    beta_scale = SCALE_BLOCK0_LN2_BIAS;
    run_test(1);

    // Test 2: embed(200,100) block1_ln1 - high token/pos
    $display("Test 2: embed(200,100) block1_ln1");
    $fwrite(fd, "Test 2: embed(200,100) block1_ln1\n");
    gamma_sel = 6'd10;
    gamma_scale = SCALE_BLOCK1_LN1_WEIGHT;
    beta_scale = SCALE_BLOCK1_LN1_BIAS;
    run_test(2);

    // Test 3: large magnitude [-50,+50] block2_ln1
    $display("Test 3: large values block2_ln1");
    $fwrite(fd, "Test 3: large values block2_ln1\n");
    gamma_sel = 6'd18;
    gamma_scale = SCALE_BLOCK2_LN1_WEIGHT;
    beta_scale = SCALE_BLOCK2_LN1_BIAS;
    run_test(3);

    // Test 4: near-constant (tiny variance) block2_ln2
    $display("Test 4: near-constant block2_ln2");
    $fwrite(fd, "Test 4: near-constant block2_ln2\n");
    gamma_sel = 6'd22;
    gamma_scale = SCALE_BLOCK2_LN2_WEIGHT;
    beta_scale = SCALE_BLOCK2_LN2_BIAS;
    run_test(4);

    // Test 5: mixed outliers block3_ln1
    $display("Test 5: mixed outliers block3_ln1");
    $fwrite(fd, "Test 5: mixed outliers block3_ln1\n");
    gamma_sel = 6'd26;
    gamma_scale = SCALE_BLOCK3_LN1_WEIGHT;
    beta_scale = SCALE_BLOCK3_LN1_BIAS;
    run_test(5);

    // Test 6: all negative block3_ln2
    $display("Test 6: all negative block3_ln2");
    $fwrite(fd, "Test 6: all negative block3_ln2\n");
    gamma_sel = 6'd30;
    gamma_scale = SCALE_BLOCK3_LN2_WEIGHT;
    beta_scale = SCALE_BLOCK3_LN2_BIAS;
    run_test(6);

    // Test 7: sparse with zeros ln_f
    $display("Test 7: sparse ln_f");
    $fwrite(fd, "Test 7: sparse ln_f\n");
    gamma_sel = 6'd34;
    gamma_scale = SCALE_LN_F_WEIGHT;
    beta_scale = SCALE_LN_F_BIAS;
    run_test(7);

    $display("=== All 8 tests done ===");
    $fwrite(fd, "=== All 8 tests done ===\n");

    $fclose(fd);
    $finish;
  end

  // Timeout
  initial begin
    #10_000_000;
    $display("TIMEOUT");
    $finish;
  end

endmodule