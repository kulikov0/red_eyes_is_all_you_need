`timescale 1ns / 1ps

module tb_transformer_top;

  reg clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst = 1'b0;
  reg [7:0] token;
  reg start = 1'b0;
  reg gen_mode = 1'b0;

  // DUT <-> weight store
  wire [5:0]  w_sel;
  wire [15:0] w_addr;
  wire [7:0]  w_data;
  wire [3:0]  w8_sel;
  wire [15:0] w8_addr;
  wire [63:0] w8_data;
  wire [11:0] tok_emb_addr;
  wire [63:0] tok_emb_data;
  wire [15:0] tok_emb_scale;
  // DUT <-> KV caches, independent address ports
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

  // DUT outputs
  wire [7:0]  out_token;
  wire        token_valid;
  wire        done;

  wire [15:0] w_scale;
  wire [15:0] w8_scale;

  weight_store u_ws (
    .clk_i       (clk),
    .tensor_sel_i(w_sel),
    .addr_i      (w_addr),
    .data_o      (w_data),
    .scale_o     (w_scale)
  );

  weight_store_w8 u_ws_w8 (
    .clk_i           (clk),
    .w8_sel_i        (w8_sel),
    .w8_addr_i       (w8_addr),
    .data_o          (w8_data),
    .scale_o         (w8_scale),
    .tok_emb_addr_i  (tok_emb_addr),
    .tok_emb_data_o  (tok_emb_data),
    .tok_emb_scale_o (tok_emb_scale)
  );

  // K cache (fp16, DATA_W=16)
  kv_cache #(.DATA_W(16)) u_k_cache (
    .clk_i  (clk),
    .layer_i(k_layer),
    .head_i (k_head),
    .pos_i  (k_pos),
    .dim_i  (k_dim),
    .we_i   (k_we),
    .wdata_i(k_wdata),
    .rdata_o(k_rdata)
  );

  // V cache (fp16, DATA_W=16)
  kv_cache #(.DATA_W(16)) u_v_cache (
    .clk_i  (clk),
    .layer_i(v_layer),
    .head_i (v_head),
    .pos_i  (v_pos),
    .dim_i  (v_dim),
    .we_i   (v_we),
    .wdata_i(v_wdata),
    .rdata_o(v_rdata)
  );

  // DUT
  transformer_top dut (
    .clk_i       (clk),
    .rst_i       (rst),
    .token_i     (token),
    .start_i     (start),
    .generate_i  (gen_mode),
    .w_sel_o     (w_sel),
    .w_addr_o    (w_addr),
    .w_data_i    (w_data),
    .w_scale_i   (w_scale),
    .w8_sel_o        (w8_sel),
    .w8_addr_o       (w8_addr),
    .w8_data_i       (w8_data),
    .w8_scale_i      (w8_scale),
    .tok_emb_addr_o  (tok_emb_addr),
    .tok_emb_data_i  (tok_emb_data),
    .tok_emb_scale_i (tok_emb_scale),
    .k_we_o      (k_we),
    .k_wdata_o   (k_wdata),
    .k_layer_o   (k_layer),
    .k_head_o    (k_head),
    .k_pos_o     (k_pos),
    .k_dim_o     (k_dim),
    .k_rdata_i   (k_rdata),
    .v_we_o      (v_we),
    .v_wdata_o   (v_wdata),
    .v_layer_o   (v_layer),
    .v_head_o    (v_head),
    .v_pos_o     (v_pos),
    .v_dim_o     (v_dim),
    .v_rdata_i   (v_rdata),
    .token_o     (out_token),
    .token_valid_o(token_valid),
    .busy_o      (),
    .done_o      (done)
  );

  integer fd, i;
  integer token_count;


  initial begin
    fd = $fopen("/home/user/red_eyes_is_all_you_need/logs/tb_transformer_top.log", "w");

    token    = 8'd0;
    start    = 1'b0;
    gen_mode = 1'b0;

    // Reset
    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    $display("=== Transformer Top Testbench ===");

    // Test 0: Single token generation (token=42, start+generate)
    $display("Test 0: single token generate, token=42");
    $fwrite(fd, "TEST 0 TOKEN=42 POS=0 GENERATE\n");
    token    = 8'd42;
    gen_mode = 1'b1;
    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;
    gen_mode = 1'b0;

    // Wait for first token_valid
    while (!token_valid) @(posedge clk);

    // Log logits
    for (i = 0; i < 256; i = i + 1) begin
      $fwrite(fd, "LOGITS[%0d]=%04x\n", i, dut.act_ram[i]);
    end
    $fwrite(fd, "OUT_TOKEN=%02x\n", out_token);
    $display("  Test 0: output token=%0d (0x%02x)", out_token, out_token);

    // Stop autoregressive by resetting
    @(posedge clk);
    rst = 1'b1;
    repeat(3) @(posedge clk);
    rst = 1'b0;
    repeat(5) @(posedge clk);

    // Test 1: Two prompt tokens + generate
    // First prompt token (no generate)
    $display("Test 1: prompt token=10, then token=20+generate");
    $fwrite(fd, "TEST 1 TOKEN=10 POS=0 PROMPT\n");
    token    = 8'd10;
    gen_mode = 1'b0;
    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    // Wait for done (prompt mode)
    while (!done) @(posedge clk);
    $fwrite(fd, "PROMPT_DONE POS=0\n");
    $display("  Prompt token=10 done");
    repeat(5) @(posedge clk);

    // Second token with generate
    $fwrite(fd, "TEST 1 TOKEN=20 POS=1 GENERATE\n");
    token    = 8'd20;
    gen_mode = 1'b1;
    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;
    gen_mode = 1'b0;

    // Wait for first token_valid
    while (!token_valid) @(posedge clk);
    for (i = 0; i < 256; i = i + 1) begin
      $fwrite(fd, "LOGITS[%0d]=%04x\n", i, dut.act_ram[i]);
    end
    $fwrite(fd, "OUT_TOKEN=%02x\n", out_token);
    $display("  Test 1: output token=%0d (0x%02x)", out_token, out_token);

    // Stop autoregressive
    @(posedge clk);
    rst = 1'b1;
    repeat(3) @(posedge clk);
    rst = 1'b0;
    repeat(5) @(posedge clk);

    // Test 2: Short autoregressive sequence (5 tokens from token=42)
    $display("Test 2: autoregressive 5 tokens from token=42");
    $fwrite(fd, "TEST 2 TOKEN=42 POS=0 GENERATE\n");
    token    = 8'd42;
    gen_mode = 1'b1;
    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;
    gen_mode = 1'b0;

    token_count = 0;
    while (token_count < 5) begin
      @(posedge clk);
      if (token_valid) begin
        $fwrite(fd, "GEN_TOKEN[%0d]=%02x\n", token_count, out_token);
        $display("  Token %0d: %0d (0x%02x)", token_count, out_token, out_token);
        token_count = token_count + 1;
        // Log logits only for first generated token
        if (token_count == 1) begin
          for (i = 0; i < 256; i = i + 1) begin
            $fwrite(fd, "LOGITS[%0d]=%04x\n", i, dut.act_ram[i]);
          end
        end
      end
    end

    // Stop autoregressive
    @(posedge clk);
    rst = 1'b1;
    repeat(3) @(posedge clk);
    rst = 1'b0;
    repeat(5) @(posedge clk);

    $display("=== All 3 tests done ===");
    $fclose(fd);
    $finish;
  end

  // Timeout: 10B ns (split to avoid 32-bit overflow)
  initial begin
    #2_000_000_000;
    #2_000_000_000;
    #2_000_000_000;
    #2_000_000_000;
    #2_000_000_000;
    $display("TIMEOUT");
    $fclose(fd);
    $finish;
  end

endmodule