`timescale 1ns / 1ps

module tb_matvec_int8;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  // DUT signals
  reg rst = 1'b0;
  reg start = 1'b0;
  wire [1023:0] out_vec;
  wire done;

  // Input vector: all ones
  reg [1023:0] in_vec;
  integer i;

  // External BRAM - testbench owns this
  reg signed [7:0] w_q [0:16383];
  initial $readmemh("/home/user/red_eyes_is_all_you_need/mem/block0_attn_proj_weight.hex", w_q);

  // BRAM read - registered (1-cycle latency, matches weight_rom.v)
  wire [13:0] weight_addr;
  reg signed [7:0] weight_data;
  always @(posedge clk) weight_data <= w_q[weight_addr];

  // DUT instantiation
  matvec_int8 #(
    .IN_DIM(128), .OUT_DIM(128)
  ) dut (
    .clk_i        (clk),
    .rst_i        (rst),
    .start_i      (start),
    .in_vec_i     (in_vec),
    .weight_addr_o(weight_addr),
    .weight_data_i(weight_data),
    .out_vec_o    (out_vec),
    .done_o       (done)
  );

  // Test
  integer errors;
  integer fd;
  initial begin
    $dumpfile("tb_matvec.vcd");
    $dumpvars(0, tb_matvec_int8);

    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_matvec_int8.log", "w");

    // Fill input with all ones
    for (i = 0; i < 128; i = i + 1) begin
      in_vec[i*8 +: 8] = 8'sd1;
    end

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    // Start
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    // Wait for done
    wait(done);
    @(posedge clk);

    // Print and check outputs
    errors = 0;
    for (i = 0; i < 128; i = i + 1) begin
      if (out_vec[i*8 +: 8] === 8'bx || out_vec[i*8 +: 8] === 8'bz) begin
        $display("out[%0d] = X  <- UNINITIALIZED", i);
        $fwrite(fd, "out[%0d] = X  <- UNINITIALIZED\n", i);
        errors = errors + 1;
      end else begin
        $display("out[%0d] = %0d", i, $signed(out_vec[i*8 +: 8]));
        $fwrite(fd, "out[%0d] = %0d\n", i, $signed(out_vec[i*8 +: 8]));
      end
    end

    if (errors == 0) begin
      $display("PASS");
      $fwrite(fd, "PASS\n");
    end else begin
      $display("FAIL: %0d errors", errors);
      $fwrite(fd, "FAIL: %0d errors\n", errors);
    end

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
