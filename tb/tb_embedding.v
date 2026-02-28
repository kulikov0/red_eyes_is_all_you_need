`timescale 1ns / 1ps

module tb_embedding;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst = 1'b0;
  reg start = 1'b0;
  reg [7:0] token_id;
  reg [7:0] position;

  wire [5:0]  w_sel;
  wire [15:0] w_addr;
  wire [7:0]  w_data;
  wire [1023:0] embed;
  wire done;
  wire busy;

  // Simulate weight_store: tok_emb (sel 0) and pos_emb (sel 1)
  reg [7:0] tok_mem [0:32767];
  reg [7:0] pos_mem [0:32767];
  initial begin
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/tok_emb_weight.hex", tok_mem);
    $readmemh("/home/user/red_eyes_is_all_you_need/mem/pos_emb_weight.hex", pos_mem);
  end

  // Registered BRAM read with tensor_sel mux (1-cycle latency)
  reg [5:0] sel_r;
  reg [7:0] w_data_r;
  always @(posedge clk) begin
    sel_r <= w_sel;
    case (w_sel)
      6'd0: w_data_r <= tok_mem[w_addr[14:0]];
      6'd1: w_data_r <= pos_mem[w_addr[14:0]];
      default: w_data_r <= 8'd0;
    endcase
  end
  assign w_data = w_data_r;

  // DUT
  embedding #(
    .DIM(128)
  ) dut (
    .clk_i       (clk),
    .rst_i       (rst),
    .start_i     (start),
    .token_id_i  (token_id),
    .position_i  (position),
    .w_sel_o     (w_sel),
    .w_addr_o    (w_addr),
    .w_data_i    (w_data),
    .embed_o     (embed),
    .done_o      (done),
    .busy_o      (busy)
  );

  integer fd, t, i;

  task run_test;
    input integer test_num;
    input [7:0] tok;
    input [7:0] pos;
    begin
      @(posedge clk);
      token_id = tok;
      position = pos;
      @(posedge clk);
      start = 1'b1;
      @(posedge clk);
      start = 1'b0;

      // Wait for done
      while (!done) @(posedge clk);

      for (i = 0; i < 128; i = i + 1) begin
        $fwrite(fd, "T=%0d TOK=%0d POS=%0d I=%0d OUT=%02x\n",
                test_num, tok, pos, i, embed[i*8 +: 8]);
      end

      $display("Test %0d done: tok=%0d pos=%0d", test_num, tok, pos);
      @(posedge clk);
    end
  endtask

  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_embedding.log", "w");

    token_id = 8'd0;
    position = 8'd0;

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    $display("=== Embedding Testbench ===");

    run_test(0, 8'd0,   8'd0);
    run_test(1, 8'd1,   8'd1);
    run_test(2, 8'd42,  8'd10);
    run_test(3, 8'd255, 8'd255);

    $display("=== All tests done ===");

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