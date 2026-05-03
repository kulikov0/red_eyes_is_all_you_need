`timescale 1ns / 1ps

module tb_sampler;

  localparam N        = 256;
  localparam N_TESTS  = 7;
  localparam MASK_W   = 16;
  localparam HEADER_W = 4 + MASK_W;
  localparam REC_W    = HEADER_W + N;

  reg          clk;
  reg          rst;
  reg          start;
  reg  [15:0]  inv_temp;
  reg  [7:0]   top_k;
  reg  [15:0]  inv_penalty;
  reg          mark_seen;
  reg  [7:0]   mark_token;
  reg          seed_load;
  reg  [15:0]  seed;
  wire [7:0]   logit_raddr;
  reg  [15:0]  logit_rdata;
  wire [7:0]   token;
  wire         done;

  reg [15:0] act_ram [0:N-1];
  reg [15:0] all_vec [0:N_TESTS*REC_W-1];

  always @(*) logit_rdata = act_ram[logit_raddr];

  sampler u_samp (
    .clk_i        (clk),
    .rst_i        (rst),
    .start_i      (start),
    .inv_temp_i   (inv_temp),
    .top_k_i      (top_k),
    .inv_penalty_i(inv_penalty),
    .mark_seen_i  (mark_seen),
    .mark_token_i (mark_token),
    .seed_load_i  (seed_load),
    .seed_i       (seed),
    .logit_raddr_o(logit_raddr),
    .logit_rdata_i(logit_rdata),
    .token_o      (token),
    .done_o       (done)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer fd, t, k;
  reg [7:0] tokens [0:N_TESTS-1];

  task run_one;
    input integer ti;
    integer i;
    begin
      inv_temp    = all_vec[ti*REC_W + 0];
      top_k       = all_vec[ti*REC_W + 1][7:0];
      seed        = all_vec[ti*REC_W + 2];
      inv_penalty = all_vec[ti*REC_W + 3];
      for (i = 0; i < N; i = i + 1)
        act_ram[i] = all_vec[ti*REC_W + HEADER_W + i];

      @(posedge clk);
      seed_load <= 1'b1;
      @(posedge clk);
      seed_load <= 1'b0;

      for (i = 0; i < N; i = i + 1) begin
        if (all_vec[ti*REC_W + 4 + (i >> 4)][i & 4'hF]) begin
          @(posedge clk);
          mark_seen  <= 1'b1;
          mark_token <= i[7:0];
          @(posedge clk);
          mark_seen  <= 1'b0;
        end
      end

      @(posedge clk);
      start <= 1'b1;
      @(posedge clk);
      start <= 1'b0;

      while (!done) @(posedge clk);
      tokens[ti] = token;
      @(posedge clk);
    end
  endtask

  initial begin
    rst         = 1'b1;
    start       = 1'b0;
    seed_load   = 1'b0;
    mark_seen   = 1'b0;
    mark_token  = 8'd0;
    inv_temp    = 16'h3C00;
    top_k       = 8'd1;
    inv_penalty = 16'h3C00;
    seed        = 16'hACE1;

    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_sampler.log", "w");
    $fwrite(fd, "=== Sampler Testbench ===\n");
    $display("=== Sampler Testbench ===");

    $readmemh("/home/user/red_eyes_is_all_you_need/mem/sampler_test_vectors.hex", all_vec);

    #30;
    @(posedge clk);
    rst = 1'b0;
    #20;

    for (t = 0; t < N_TESTS; t = t + 1) begin
      $fwrite(fd, "\nTest %0d:\n", t);
      $display("Test %0d:", t);
      run_one(t);
      $fwrite(fd, "TOKEN=%0d INV_TEMP=%04x TOPK=%0d SEED=%04x PEN=%04x\n",
              tokens[t], all_vec[t*REC_W+0],
              all_vec[t*REC_W+1][7:0], all_vec[t*REC_W+2],
              all_vec[t*REC_W+3]);
      $display("  token=%0d inv_temp=%04x topk=%0d seed=%04x pen=%04x",
               tokens[t], all_vec[t*REC_W+0],
               all_vec[t*REC_W+1][7:0], all_vec[t*REC_W+2],
               all_vec[t*REC_W+3]);
    end

    $fclose(fd);
    $finish;
  end

endmodule