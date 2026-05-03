// Full sampler: temperature, softmax, top-k, LFSR multinomial draw

module sampler #(
  parameter N     = 256,
  parameter K_MAX = 16
) (
  input  wire         clk_i,
  input  wire         rst_i,
  input  wire         start_i,
  input  wire [15:0]  inv_temp_i,
  input  wire [7:0]   top_k_i,
  input  wire [15:0]  inv_penalty_i,
  input  wire         mark_seen_i,
  input  wire [7:0]   mark_token_i,
  input  wire         seed_load_i,
  input  wire [15:0]  seed_i,
  output wire [7:0]   logit_raddr_o,
  input  wire [15:0]  logit_rdata_i,
  output reg  [7:0]   token_o,
  output reg          done_o
);

  localparam ADDR_W  = 8;
  localparam K_IDX_W = 4;
  localparam K_CNT_W = 5;

  localparam [3:0] S_IDLE        = 4'd0,
                   S_PREP        = 4'd1,
                   S_TEMP        = 4'd2,
                   S_TOPK_FETCH  = 4'd3,
                   S_TOPK_WAIT   = 4'd4,
                   S_TOPK_UPDATE = 4'd5,
                   S_FINALIZE    = 4'd6,
                   S_DRAW        = 4'd7,
                   S_SCAN        = 4'd8,
                   S_DONE        = 4'd9;

  reg [3:0] state;

  // Repetition penalty: tokens already in the context are demoted by
  // multiplying their logit by inv_penalty (matching float reference
  // logit /= penalty). seen_mask is loaded externally via mark_seen_i pulses
  reg [N-1:0] seen_mask;
  always @(posedge clk_i) begin
    if (rst_i || seed_load_i) begin
      seen_mask <= {N{1'b0}};
    end else if (mark_seen_i) begin
      seen_mask[mark_token_i] <= 1'b1;
    end
  end

  // Latch config at S_IDLE so per-cycle muxes use stable operands
  reg [15:0] inv_temp_r;
  reg [15:0] inv_penalty_r;

  reg [K_CNT_W-1:0] k_eff_r;
  reg               full_vocab_r;

  // Polynomial x^16 + x^14 + x^13 + x^11 + 1
  // Advance one cycle before S_DRAW so r_target uses the post-advance value
  reg [15:0] lfsr_r;
  wire fin_done_pending;
  always @(posedge clk_i) begin
    if (rst_i) begin
      lfsr_r <= 16'hACE1;
    end else if (seed_load_i) begin
      lfsr_r <= seed_i;
    end else if (fin_done_pending) begin
      lfsr_r <= {lfsr_r[14:0],
                 lfsr_r[15] ^ lfsr_r[13] ^ lfsr_r[12] ^ lfsr_r[10]};
    end
  end

  // Precompute inv_temp_pen = inv_temp * inv_penalty once per sample call
  reg         pre_fire;
  wire [15:0] inv_temp_pen_w;
  wire        inv_temp_pen_v;
  fp16_mul u_pre (
    .clk_i  (clk_i),
    .valid_i(pre_fire),
    .a_i    (inv_temp_r),
    .b_i    (inv_penalty_r),
    .valid_o(inv_temp_pen_v),
    .prod_o (inv_temp_pen_w)
  );
  reg [15:0] inv_temp_pen_r;
  always @(posedge clk_i) begin
    if (inv_temp_pen_v) inv_temp_pen_r <= inv_temp_pen_w;
  end

  reg [ADDR_W:0] feed_cnt;
  wire feed_active = (state == S_TEMP) && (feed_cnt < N);
  assign logit_raddr_o = feed_cnt[ADDR_W-1:0];

  wire token_seen = seen_mask[feed_cnt[ADDR_W-1:0]];
  wire [15:0] mul_b = token_seen ? inv_temp_pen_r : inv_temp_r;

  wire [15:0] scaled_fp16;
  wire        mul_valid_o;
  fp16_mul u_mul (
    .clk_i  (clk_i),
    .valid_i(feed_active),
    .a_i    (logit_rdata_i),
    .b_i    (mul_b),
    .valid_o(mul_valid_o),
    .prod_o (scaled_fp16)
  );

  wire [23:0] sm_in_q167;
  fp16_to_q167 u_q167 (
    .val_i (scaled_fp16),
    .q167_o(sm_in_q167)
  );

  reg         sm_start;
  wire        sm_in_ready;
  wire        sm_out_valid;
  wire [15:0] sm_out_data;
  wire        sm_done;
  softmax #(
    .N    (N),
    .IN_W (24),
    .FRAC_W(7),
    .OUT_W(16)
  ) u_sm (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .start_i    (sm_start),
    .in_valid_i (mul_valid_o),
    .in_data_i  (sm_in_q167),
    .in_ready_o (sm_in_ready),
    .out_valid_o(sm_out_valid),
    .out_data_o (sm_out_data),
    .done_o     (sm_done)
  );

  // Distributed RAM so the top-k and cumsum scans can read async
  (* ram_style = "distributed" *) reg [15:0] prob_buf [0:N-1];
  reg [ADDR_W:0] drain_cnt;
  always @(posedge clk_i) begin
    if (rst_i) begin
      drain_cnt <= 0;
    end else if (state == S_IDLE && start_i) begin
      drain_cnt <= 0;
    end else if (sm_out_valid && drain_cnt < N) begin
      prob_buf[drain_cnt[ADDR_W-1:0]] <= sm_out_data;
      drain_cnt <= drain_cnt + 1'b1;
    end
  end

  reg [15:0]        topk_val [0:K_MAX-1];
  reg [ADDR_W-1:0]  topk_idx [0:K_MAX-1];
  reg [K_CNT_W-1:0] n_filled;

  // Pre-computed at S_IDLE so the per-slot compare g < k_eff_r does not
  // fan out into the min tree on every cycle
  reg [K_MAX-1:0] active_mask_r;

  // Balanced compare tree: behavioral for-loop synthesizes as a serpentine
  // chain that blows the timing budget, so the tree is written out explicitly
  // Slots beyond k_eff_r are forced to max so they never win min
  wire [15:0] cand_val [0:K_MAX-1];
  genvar g;
  generate
    for (g = 0; g < K_MAX; g = g + 1) begin : g_cand
      assign cand_val[g] = active_mask_r[g] ? topk_val[g] : 16'hFFFF;
    end
  endgenerate

  wire [15:0]        s1_val [0:K_MAX/2-1];
  wire [K_IDX_W-1:0] s1_idx [0:K_MAX/2-1];
  generate
    for (g = 0; g < K_MAX/2; g = g + 1) begin : g_s1
      wire [K_IDX_W-1:0] idx_a = 2*g;
      wire [K_IDX_W-1:0] idx_b = 2*g + 1;
      wire pick_b = cand_val[2*g+1] < cand_val[2*g];
      assign s1_val[g] = pick_b ? cand_val[2*g+1] : cand_val[2*g];
      assign s1_idx[g] = pick_b ? idx_b : idx_a;
    end
  endgenerate

  // Register s1 to break the 4-level tree into a 2-cycle pipeline.
  // Combined with min_*_r this yields 2 cycles of latency from a slot
  // write to its effect on min_*_r, matched by S_TOPK_FETCH -> WAIT -> UPDATE
  reg [15:0]        s1_val_r [0:K_MAX/2-1];
  reg [K_IDX_W-1:0] s1_idx_r [0:K_MAX/2-1];

  wire [15:0]        s2_val [0:K_MAX/4-1];
  wire [K_IDX_W-1:0] s2_idx [0:K_MAX/4-1];
  generate
    for (g = 0; g < K_MAX/4; g = g + 1) begin : g_s2
      wire pick_b = s1_val_r[2*g+1] < s1_val_r[2*g];
      assign s2_val[g] = pick_b ? s1_val_r[2*g+1] : s1_val_r[2*g];
      assign s2_idx[g] = pick_b ? s1_idx_r[2*g+1] : s1_idx_r[2*g];
    end
  endgenerate

  wire [15:0]        s3_val [0:K_MAX/8-1];
  wire [K_IDX_W-1:0] s3_idx [0:K_MAX/8-1];
  generate
    for (g = 0; g < K_MAX/8; g = g + 1) begin : g_s3
      wire pick_b = s2_val[2*g+1] < s2_val[2*g];
      assign s3_val[g] = pick_b ? s2_val[2*g+1] : s2_val[2*g];
      assign s3_idx[g] = pick_b ? s2_idx[2*g+1] : s2_idx[2*g];
    end
  endgenerate

  wire pick_b_final  = s3_val[1] < s3_val[0];
  wire [15:0]        min_val_c  = pick_b_final ? s3_val[1] : s3_val[0];
  wire [K_IDX_W-1:0] min_slot_c = pick_b_final ? s3_idx[1] : s3_idx[0];

  reg [K_IDX_W-1:0] min_slot_r;
  reg [15:0]        min_val_r;
  reg [15:0]        prob_r;
  reg [ADDR_W-1:0]  prob_idx_r;

  integer si;
  always @(posedge clk_i) begin
    for (si = 0; si < K_MAX/2; si = si + 1) begin
      s1_val_r[si] <= s1_val[si];
      s1_idx_r[si] <= s1_idx[si];
    end
    min_val_r  <= min_val_c;
    min_slot_r <= min_slot_c;
  end

  reg [ADDR_W:0] scan_cnt;
  wire [15:0] cur_prob = prob_buf[scan_cnt[ADDR_W-1:0]];

  reg [N-1:0] keep_mask;

  reg [23:0]        sum_topk;
  reg [K_CNT_W-1:0] fin_cnt;
  reg [ADDR_W:0]    fin_full_cnt;

  assign fin_done_pending = (state == S_FINALIZE) &&
    (full_vocab_r ? (fin_full_cnt == N) : (fin_cnt == k_eff_r));

  // r_target = floor(lfsr * sum_topk / 65536), keeping the upper 24 bits
  wire [39:0] r_full = lfsr_r * sum_topk;
  reg [23:0] r_target;

  reg [23:0]       cum_acc;
  reg              picked;
  reg [ADDR_W-1:0] last_kept;

  integer j;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state          <= S_IDLE;
      done_o         <= 1'b0;
      sm_start       <= 1'b0;
      pre_fire       <= 1'b0;
      feed_cnt       <= 0;
      keep_mask      <= {N{1'b0}};
      n_filled       <= 0;
      sum_topk       <= 24'd0;
      fin_cnt        <= 0;
      fin_full_cnt   <= 0;
      r_target       <= 24'd0;
      cum_acc        <= 24'd0;
      picked         <= 1'b0;
      last_kept      <= {ADDR_W{1'b0}};
      token_o        <= 8'd0;
      k_eff_r        <= K_MAX[K_CNT_W-1:0];
      full_vocab_r   <= 1'b0;
      active_mask_r  <= {K_MAX{1'b0}};
      inv_temp_r     <= 16'h3C00;
      inv_penalty_r  <= 16'h3C00;
    end else begin
      done_o   <= 1'b0;
      sm_start <= 1'b0;
      pre_fire <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            inv_temp_r    <= inv_temp_i;
            inv_penalty_r <= inv_penalty_i;
            pre_fire      <= 1'b1;
            full_vocab_r  <= (top_k_i == 8'd0);
            if (top_k_i == 8'd0 || top_k_i > K_MAX[7:0]) begin
              k_eff_r       <= K_MAX[K_CNT_W-1:0];
              active_mask_r <= {K_MAX{1'b1}};
            end else begin
              k_eff_r       <= top_k_i[K_CNT_W-1:0];
              active_mask_r <= ({K_MAX{1'b1}} >> (K_MAX - top_k_i[K_CNT_W-1:0]));
            end
            for (j = 0; j < K_MAX; j = j + 1) begin
              topk_val[j] <= 16'd0;
              topk_idx[j] <= {ADDR_W{1'b0}};
            end
            n_filled     <= 0;
            keep_mask    <= {N{1'b0}};
            feed_cnt     <= 0;
            sum_topk     <= 24'd0;
            fin_cnt      <= 0;
            fin_full_cnt <= 0;
            picked       <= 1'b0;
            cum_acc      <= 24'd0;
            last_kept    <= {ADDR_W{1'b0}};
            state        <= S_PREP;
          end
        end

        // Wait for the inv_temp * inv_penalty precompute to land in
        // inv_temp_pen_r before the per-token feed mux uses it
        S_PREP: begin
          if (inv_temp_pen_v) begin
            sm_start <= 1'b1;
            state    <= S_TEMP;
          end
        end

        S_TEMP: begin
          if (feed_cnt < N) begin
            feed_cnt <= feed_cnt + 1'b1;
          end
          if (sm_done) begin
            state    <= full_vocab_r ? S_FINALIZE : S_TOPK_FETCH;
            scan_cnt <= 0;
          end
        end

        // Capture prob; min_*_r is updated continuously by the always block below
        S_TOPK_FETCH: begin
          if (scan_cnt < N) begin
            prob_r     <= cur_prob;
            prob_idx_r <= scan_cnt[ADDR_W-1:0];
            state      <= S_TOPK_WAIT;
          end else begin
            state <= S_FINALIZE;
          end
        end

        // One-cycle bubble for the s1 register stage to drain into min_*_r
        S_TOPK_WAIT: begin
          state <= S_TOPK_UPDATE;
        end

        // Strict-greater replace; equal probs do not enter so earlier idx wins
        S_TOPK_UPDATE: begin
          if (n_filled < k_eff_r) begin
            topk_val[n_filled[K_IDX_W-1:0]] <= prob_r;
            topk_idx[n_filled[K_IDX_W-1:0]] <= prob_idx_r;
            n_filled <= n_filled + 1'b1;
          end else if (prob_r > min_val_r) begin
            topk_val[min_slot_r] <= prob_r;
            topk_idx[min_slot_r] <= prob_idx_r;
          end
          scan_cnt <= scan_cnt + 1'b1;
          state    <= S_TOPK_FETCH;
        end

        S_FINALIZE: begin
          if (full_vocab_r) begin
            if (fin_full_cnt < N) begin
              keep_mask[fin_full_cnt[ADDR_W-1:0]] <= 1'b1;
              sum_topk     <= sum_topk + {8'd0, prob_buf[fin_full_cnt[ADDR_W-1:0]]};
              fin_full_cnt <= fin_full_cnt + 1'b1;
            end else begin
              state <= S_DRAW;
            end
          end else begin
            if (fin_cnt < k_eff_r) begin
              keep_mask[topk_idx[fin_cnt[K_IDX_W-1:0]]] <= 1'b1;
              sum_topk <= sum_topk + {8'd0, topk_val[fin_cnt[K_IDX_W-1:0]]};
              fin_cnt  <= fin_cnt + 1'b1;
            end else begin
              state <= S_DRAW;
            end
          end
        end

        S_DRAW: begin
          r_target <= r_full[39:16];
          scan_cnt <= 0;
          state    <= S_SCAN;
        end

        S_SCAN: begin
          if (sum_topk == 24'd0) begin
            if (scan_cnt < N) begin
              if (keep_mask[scan_cnt[ADDR_W-1:0]] && !picked) begin
                token_o <= scan_cnt[ADDR_W-1:0];
                picked  <= 1'b1;
                state   <= S_DONE;
              end else begin
                scan_cnt <= scan_cnt + 1'b1;
              end
            end else begin
              token_o <= 8'd0;
              state   <= S_DONE;
            end
          end else if (scan_cnt < N) begin
            if (keep_mask[scan_cnt[ADDR_W-1:0]]) begin
              cum_acc   <= cum_acc + {8'd0, cur_prob};
              last_kept <= scan_cnt[ADDR_W-1:0];
              if (cum_acc + {8'd0, cur_prob} > r_target) begin
                token_o <= scan_cnt[ADDR_W-1:0];
                picked  <= 1'b1;
                state   <= S_DONE;
              end
            end
            scan_cnt <= scan_cnt + 1'b1;
          end else begin
            // Tail safety in case cum did not exceed r_target
            token_o <= picked ? token_o : last_kept;
            state   <= S_DONE;
          end
        end

        S_DONE: begin
          done_o <= 1'b1;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule