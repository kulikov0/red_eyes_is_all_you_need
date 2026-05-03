`timescale 1ns / 1ps

module tb_profile;

  reg clk = 1'b0;
  always #5 clk = ~clk;

  reg rst = 1'b0;
  reg [7:0] token;
  reg start = 1'b0;
  reg gen_mode = 1'b0;

  wire [5:0]  w_sel;
  wire [15:0] w_addr;
  wire [7:0]  w_data;
  wire [1:0]   layer_idx;
  wire [11:0]  qkv_addr,    ff_up_addr,    ff_down_addr;
  wire [9:0]   proj_addr;
  wire [127:0] qkv_data,    proj_data,     ff_up_data,   ff_down_data;
  wire [15:0]  qkv_scale,   proj_scale,    ff_up_scale,  ff_down_scale;
  wire [10:0]  tok_emb_addr;
  wire [127:0] tok_emb_data;
  wire [15:0] tok_emb_scale;
  wire [1:0]  k_layer, v_layer;
  wire [2:0]  k_head,  v_head;
  wire [7:0]  k_pos,   v_pos;
  wire [3:0]  k_dim,   v_dim;
  wire        k_we;
  wire [15:0] k_wdata;
  wire [31:0] k_rdata;
  wire        v_we;
  wire [15:0] v_wdata;
  wire [63:0] v_rdata;

  wire [7:0]  out_token;
  wire        token_valid;
  wire        done;
  wire [15:0] w_scale;

  weight_store         u_ws         (.clk_i(clk), .tensor_sel_i(w_sel), .addr_i(w_addr), .data_o(w_data), .scale_o(w_scale));
  weight_store_qkv     u_ws_qkv     (.clk_i(clk), .layer_i(layer_idx), .addr_i(qkv_addr), .data_o(qkv_data), .scale_o(qkv_scale));
  weight_store_proj    u_ws_proj    (.clk_i(clk), .layer_i(layer_idx), .addr_i(proj_addr), .data_o(proj_data), .scale_o(proj_scale));
  weight_store_ff_up   u_ws_ff_up   (.clk_i(clk), .layer_i(layer_idx), .addr_i(ff_up_addr), .data_o(ff_up_data), .scale_o(ff_up_scale));
  weight_store_ff_down u_ws_ff_down (.clk_i(clk), .layer_i(layer_idx), .addr_i(ff_down_addr), .data_o(ff_down_data), .scale_o(ff_down_scale));
  weight_store_tok_emb u_ws_tok_emb (.clk_i(clk), .addr_i(tok_emb_addr), .data_o(tok_emb_data), .scale_o(tok_emb_scale));

  kv_cache u_k_cache (.clk_i(clk), .layer_i(k_layer), .head_i(k_head), .pos_i(k_pos), .dim_i(k_dim), .we_i(k_we), .wdata_i(k_wdata), .rdata_o(k_rdata));
  v_cache_w4 u_v_cache (.clk_i(clk), .layer_i(v_layer), .head_i(v_head), .pos_i(v_pos), .dim_i(v_dim), .we_i(v_we), .wdata_i(v_wdata), .rdata_o(v_rdata));

  transformer_top dut (
    .clk_i(clk), .rst_i(rst), .token_i(token), .start_i(start), .generate_i(gen_mode),
    .w_sel_o(w_sel), .w_addr_o(w_addr), .w_data_i(w_data), .w_scale_i(w_scale),
    .layer_idx_o(layer_idx),
    .qkv_addr_o(qkv_addr), .qkv_data_i(qkv_data), .qkv_scale_i(qkv_scale),
    .proj_addr_o(proj_addr), .proj_data_i(proj_data), .proj_scale_i(proj_scale),
    .ff_up_addr_o(ff_up_addr), .ff_up_data_i(ff_up_data), .ff_up_scale_i(ff_up_scale),
    .ff_down_addr_o(ff_down_addr), .ff_down_data_i(ff_down_data), .ff_down_scale_i(ff_down_scale),
    .tok_emb_addr_o(tok_emb_addr), .tok_emb_data_i(tok_emb_data), .tok_emb_scale_i(tok_emb_scale),
    .k_we_o(k_we), .k_wdata_o(k_wdata), .k_layer_o(k_layer), .k_head_o(k_head), .k_pos_o(k_pos), .k_dim_o(k_dim), .k_rdata_i(k_rdata),
    .v_we_o(v_we), .v_wdata_o(v_wdata), .v_layer_o(v_layer), .v_head_o(v_head), .v_pos_o(v_pos), .v_dim_o(v_dim), .v_rdata_i(v_rdata),
    .inv_temp_i(16'h3C00), .top_k_i(8'd1), .seed_load_i(1'b0), .seed_i(16'hACE1),
    .token_o(out_token), .token_valid_o(token_valid), .busy_o(), .done_o(done)
  );

  integer top_cnt [0:15];
  integer tl_cnt [0:15];
  integer attn_cnt [0:7];
  integer sc_cnt [0:7];
  integer av_cnt [0:7];

  integer total_cycles;
  integer profiling;
  integer i;

  initial begin
    for (i = 0; i < 16; i = i + 1) begin top_cnt[i] = 0; tl_cnt[i] = 0; end
    for (i = 0; i < 8;  i = i + 1) begin attn_cnt[i] = 0; sc_cnt[i] = 0; av_cnt[i] = 0; end
    total_cycles = 0;
    profiling    = 0;
  end

  always @(posedge clk) begin
    if (profiling) begin
      total_cycles            <= total_cycles + 1;
      top_cnt[dut.state]      <= top_cnt[dut.state] + 1;
      tl_cnt[dut.u_tl.state]  <= tl_cnt[dut.u_tl.state] + 1;
      attn_cnt[dut.u_tl.u_attn.state]       <= attn_cnt[dut.u_tl.u_attn.state] + 1;
      sc_cnt[dut.u_tl.u_attn.score_state]   <= sc_cnt[dut.u_tl.u_attn.score_state] + 1;
      av_cnt[dut.u_tl.u_attn.av_state]      <= av_cnt[dut.u_tl.u_attn.av_state] + 1;
    end
  end

  integer cycles_at_token [0:255];
  integer token_count;

  initial begin
    token       = 8'd0;
    start       = 1'b0;
    gen_mode    = 1'b0;
    token_count = 0;

    rst = 1'b1;
    repeat(5) @(posedge clk);
    rst = 1'b0;
    repeat(2) @(posedge clk);

    token    = 8'd42;
    gen_mode = 1'b1;
    @(posedge clk);
    start     = 1'b1;
    profiling = 1;
    @(posedge clk);
    start     = 1'b0;
    gen_mode  = 1'b0;

    while (token_count < 50) begin
      @(posedge clk);
      if (token_valid) begin
        cycles_at_token[token_count] = total_cycles;
        token_count = token_count + 1;
      end
    end
    profiling = 0;

    $display("");
    for (i = 0; i < 50; i = i + 1) begin
      if (i == 0) $display("  token %0d (pos=0): %0d cycles", i, cycles_at_token[i]);
      else        $display("  token %0d (pos=%0d): delta=%0d",
                           i, i, cycles_at_token[i] - cycles_at_token[i-1]);
    end
    $display("");
    $display("avg per-token (50 tokens): %0d cycles", cycles_at_token[49] / 50);
    $display("=== TOTAL CYCLES: %0d ===", total_cycles);
    $display("S_HEADS=%0d  AV_RUN=%0d  AV_WAIT=%0d  SC_SCORE=%0d  SC_PAD=%0d",
             attn_cnt[3], av_cnt[1], av_cnt[3], sc_cnt[1], sc_cnt[2]);

    $finish;
  end

  initial begin
    #2_000_000_000;
    $display("TIMEOUT");
    $finish;
  end

endmodule
