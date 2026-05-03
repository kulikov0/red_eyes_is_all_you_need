// Multi-head self-attention for single-token autoregressive inference, W8A16.
//
// Flow: QKV matvec_fp16_w16 -> KV store -> 8 heads of score, softmax, AV -> proj.
//
// Precision:
//   Weights: int8 in per-tensor banks (qkv/proj), dequanted to fp16 via scale.
//   K/V cache: fp16.
//   Score: fp16 dot(Q,K) * 1/sqrt(16), converted to Q16.7 for softmax.
//   Softmax: Q16.7 input IN_W=24, Q1.15 output via bipartite LUT.
//   AV: Q1.15 -> fp16, then fp16 accumulation.
//
// w16_sel encoding for the per-layer weight bus: {layer_i, type}
//   type 00: qkv, type 01: proj

module attention (
  input  wire          clk_i,
  input  wire          rst_i,
  input  wire          start_i,
  input  wire [1:0]    layer_i,
  input  wire [7:0]    pos_i,
  output wire [6:0]    act_raddr_o,
  input  wire [15:0]   act_rdata_i,

  output reg           res_we_o,
  output reg  [6:0]    res_waddr_o,
  output reg  [15:0]   res_wdata_o,

  // Per-tensor weight buses: qkv and proj have their own dedicated banks
  // co-located with this matvec instance, no shared addr/data mux
  output wire [11:0]   qkv_addr_o,
  input  wire [127:0]  qkv_data_i,
  input  wire [15:0]   qkv_scale_i,
  output wire [9:0]    proj_addr_o,
  input  wire [127:0]  proj_data_i,
  input  wire [15:0]   proj_scale_i,

  // K cache addr is separate from V so score and AV can run in parallel.
  // KV cache packs 2 consecutive positions per 32-bit word; rdata returns
  // {odd_pos, even_pos} and the consumer half-selects via pos[0]
  output reg           k_we_o,
  output reg  [15:0]   k_wdata_o,
  output reg  [1:0]    k_layer_o,
  output reg  [2:0]    k_head_o,
  output reg  [7:0]    k_pos_o,
  output reg  [3:0]    k_dim_o,
  input  wire [31:0]   k_rdata_i,

  output reg           v_we_o,
  output reg  [15:0]   v_wdata_o,
  output reg  [1:0]    v_layer_o,
  output reg  [2:0]    v_head_o,
  output reg  [7:0]    v_pos_o,
  output reg  [3:0]    v_dim_o,
  input  wire [31:0]   v_rdata_i,

  output reg           done_o
);

  // 1/sqrt(HEAD_DIM) = 1/sqrt(16) = 0.25
  localparam [15:0] INV_SQRT_DK = 16'h3400;

  // Top-level FSM states. S_HEADS runs the score and AV sub-FSMs in parallel
  localparam [2:0] S_IDLE     = 3'd0,
                   S_QKV      = 3'd1,
                   S_KV_STORE = 3'd2,
                   S_HEADS    = 3'd3,
                   S_PROJ     = 3'd4,
                   S_DONE     = 3'd5;
  reg [2:0] state;

  // Score sub-FSM: handles score, pad, softmax wait for one head
  localparam [2:0] SC_IDLE    = 3'd0,
                   SC_SCORE   = 3'd1,
                   SC_PAD     = 3'd2,
                   SC_SM_WAIT = 3'd3,
                   SC_WAIT    = 3'd4,
                   SC_DONE    = 3'd5;
  reg [2:0] score_state;
  reg [2:0] score_head_idx;

  // AV sub-FSM: handles AV accumulate and store for one head
  localparam [2:0] AV_IDLE  = 3'd0,
                   AV_RUN   = 3'd1,
                   AV_STORE = 3'd2,
                   AV_WAIT  = 3'd3,
                   AV_DONE  = 3'd4;
  reg [2:0] av_state;
  reg [2:0] av_head_idx;

  // Pipeline depth tracking. score writes attn_buf when scored_count
  // increments; AV reads when av_done_count is behind. Difference must stay
  // <= 2 so that score head N+2 doesn't overwrite buf still being read by
  // AV head N
  reg [3:0] scored_count;
  reg [3:0] av_done_count;

  // Cap sub-FSM runs in parallel with score sub-FSM SC_SCORE issue side
  localparam [1:0] C_IDLE = 2'd0,
                   C_WAIT = 2'd1,
                   C_MUL  = 2'd2,
                   C_FEED = 2'd3;
  reg [1:0] cap_state;
  reg [7:0] sc_pos_cap;
  reg       issue_done;
  reg       cap_done;

  // Latched inputs
  reg [1:0] layer_r;
  reg [7:0] pos_r;

  // Internal distributed RAMs for QKV output and head output
  reg [15:0] qkv_ram [0:383];
  reg [15:0] head_ram [0:127];

  // QKV matvec: reads from shared RAM, writes to qkv_ram
  reg         qkv_start;
  wire [6:0]  qkv_act_raddr;
  wire        qkv_res_we;
  wire [8:0]  qkv_res_waddr;
  wire [15:0] qkv_res_wdata;
  wire        qkv_done;

  matvec_fp16_w16 #(.IN_DIM(128), .OUT_DIM(384)) u_qkv (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (qkv_start),
    .scale_i      (qkv_scale_i),
    .weight_addr_o(qkv_addr_o),
    .weight_data_i(qkv_data_i),
    .act_raddr_o  (qkv_act_raddr),
    .act_rdata_i  (act_rdata_i),
    .res_we_o     (qkv_res_we),
    .res_waddr_o  (qkv_res_waddr),
    .res_wdata_o  (qkv_res_wdata),
    .done_o       (qkv_done)
  );

  // QKV result -> qkv_ram
  always @(posedge clk_i) begin
    if (qkv_res_we)
      qkv_ram[qkv_res_waddr] <= qkv_res_wdata;
  end

  // Proj matvec: reads from head_ram, writes to shared RAM
  reg          proj_start;
  wire [6:0]   proj_act_raddr;
  wire         proj_res_we;
  wire [6:0]   proj_res_waddr;
  wire [15:0]  proj_res_wdata;
  wire         proj_done;

  matvec_fp16_w16 #(.IN_DIM(128), .OUT_DIM(128)) u_proj (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (proj_start),
    .scale_i      (proj_scale_i),
    .weight_addr_o(proj_addr_o),
    .weight_data_i(proj_data_i),
    .act_raddr_o  (proj_act_raddr),
    .act_rdata_i  (head_ram[proj_act_raddr]),
    .res_we_o     (proj_res_we),
    .res_waddr_o  (proj_res_waddr),
    .res_wdata_o  (proj_res_wdata),
    .done_o       (proj_done)
  );

  // Proj result -> shared RAM passthrough
  always @(*) begin
    res_we_o    = proj_res_we;
    res_waddr_o = proj_res_waddr;
    res_wdata_o = proj_res_wdata;
  end

  // Two softmax instances allow head N+1 to start scoring while head N is still
  // running softmax. Even heads route to sm_a, odd heads to sm_b
  reg         sm_start;
  reg         sm_in_valid;
  reg  [23:0] sm_in_data;

  wire sm_a_active = !score_head_idx[0];
  wire sm_b_active =  score_head_idx[0];

  wire        sm_a_out_valid, sm_b_out_valid;
  wire [15:0] sm_a_out_data,  sm_b_out_data;
  wire        sm_a_done,      sm_b_done;

  softmax #(.N(256), .IN_W(24)) u_sm_a (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .start_i    (sm_start && sm_a_active),
    .in_valid_i (sm_in_valid && sm_a_active),
    .in_data_i  (sm_in_data),
    .in_ready_o (),
    .out_valid_o(sm_a_out_valid),
    .out_data_o (sm_a_out_data),
    .done_o     (sm_a_done)
  );

  softmax #(.N(256), .IN_W(24)) u_sm_b (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .start_i    (sm_start && sm_b_active),
    .in_valid_i (sm_in_valid && sm_b_active),
    .in_data_i  (sm_in_data),
    .in_ready_o (),
    .out_valid_o(sm_b_out_valid),
    .out_data_o (sm_b_out_data),
    .done_o     (sm_b_done)
  );

  // Per-instance busy flags + capture counters. Capture runs as a free pair
  // of always-block writers so head N+1 can score into sm_b while sm_a is
  // still streaming head N's outputs to attn_buf_a
  reg sm_a_busy, sm_b_busy;
  reg [7:0] sm_a_out_cnt, sm_b_out_cnt;

  // QKV act_raddr passthrough to shared RAM
  assign act_raddr_o = qkv_act_raddr;

  // Attention buffer: 256 x 16-bit softmax outputs (Q1.15), ping-ponged
  // between two banks so head N+1's softmax writes one bank while head N's
  // AV reads the other. Bank chosen by head index LSB
  reg [15:0] attn_buf_a [0:255];
  reg [15:0] attn_buf_b [0:255];

  // AV accumulators: 16 x fp16
  reg [15:0] av_acc [0:15];

  // KV store counter (0..255: first 128 = K, next 128 = V)
  reg [8:0] kv_cnt;
  // 3-cycle write FSM lets the long route to KV BRAMs use multicycle-2 in xdc
  reg [1:0] kv_phase;

  // Score computation pipeline
  reg [4:0]  sc_cnt;
  reg [7:0]  sc_pos;
  reg [3:0]  sc_dim_pipe [0:3];
  reg [3:0]  sc_bram_v;
  reg [4:0]  sc_red_in_cnt;

  // AV computation pipeline
  reg [4:0]  av_cnt;
  reg [7:0]  av_pos;
  reg [2:0]  av_bram_v;
  reg [3:0]  av_dim_pipe [0:12];
  reg        av_issue_done;

  reg [8:0] pad_cnt;


  // Q_head extraction: read from qkv_ram
  wire [15:0] q_head [0:15];
  genvar gi;
  generate
    for (gi = 0; gi < 16; gi = gi + 1) begin : gen_q
      assign q_head[gi] = qkv_ram[score_head_idx * 16 + gi];
    end
  endgenerate

  // KV store: read K and V from qkv_ram
  wire [6:0] kv_idx = kv_cnt[6:0];
  wire [15:0] kv_fp16_k = qkv_ram[128 + kv_idx];
  wire [15:0] kv_fp16_v = qkv_ram[256 + kv_idx];

  // Score Q*K mul, products feed the reducers below.
  // Each cycle reads 2 K values (lower=pos 2*pp, upper=pos 2*pp+1) from the
  // 32-bit packed kv_cache and feeds two fp16_mul instances in parallel
  wire sc_issue = (score_state == SC_SCORE) && (sc_cnt < 5'd16);

  // Register k_rdata before the score multipliers to break the K cache BRAM
  // clk-to-out -> DSP B chain
  reg [31:0] sc_k_rdata_r;
  always @(posedge clk_i) sc_k_rdata_r <= k_rdata_i;

  // Pair partial flag pipelined to align with mul valid_i. Set when the
  // current pair's upper position is past pos_r (only when pos_r is even)
  reg sc_partial_pair;
  always @(*) sc_partial_pair = (sc_pos == pos_r) && !pos_r[0];

  // sc_partial_pipe tracks sc_partial_pair through the 4-stage bram_v pipeline
  // so the mul valid for upper half can be gated when the upper pos is invalid
  reg [3:0] sc_partial_pipe;
  always @(posedge clk_i) begin
    if (rst_i) sc_partial_pipe <= 4'b0000;
    else       sc_partial_pipe <= {sc_partial_pipe[2:0], sc_partial_pair && sc_issue};
  end

  wire        sc_mul_a_v_out, sc_mul_b_v_out;
  wire [15:0] sc_mac_prod_a, sc_mac_prod_b;
  fp16_mul u_sc_mul_a (
    .clk_i  (clk_i),
    .valid_i(sc_bram_v[3]),
    .a_i    (q_head[sc_dim_pipe[3]]),
    .b_i    (sc_k_rdata_r[15:0]),
    .valid_o(sc_mul_a_v_out),
    .prod_o (sc_mac_prod_a)
  );

  fp16_mul u_sc_mul_b (
    .clk_i  (clk_i),
    .valid_i(sc_bram_v[3] && !sc_partial_pipe[3]),
    .a_i    (q_head[sc_dim_pipe[3]]),
    .b_i    (sc_k_rdata_r[31:16]),
    .valid_o(sc_mul_b_v_out),
    .prod_o (sc_mac_prod_b)
  );

  // Eight reducers in 4-way ping-pong: red_a/b/c/d each handle every 4th
  // score pair, lo/hi per pair for the K cache packed word halves.
  // red_issue_sel cycles 0..3 across pairs, mul_pos_sel is its pipelined
  // form that routes mul outputs to the matching reducer set
  reg [1:0] red_issue_sel;
  reg [2:0] red_cap_sel;
  reg [1:0] mul_pos_sel;

  wire sc_clear_pulse  = (score_state == SC_SCORE) && (sc_cnt < 5'd5);
  wire sc_clear_a_pair = sc_clear_pulse && (red_issue_sel == 2'd0);
  wire sc_clear_b_pair = sc_clear_pulse && (red_issue_sel == 2'd1);
  wire sc_clear_c_pair = sc_clear_pulse && (red_issue_sel == 2'd2);
  wire sc_clear_d_pair = sc_clear_pulse && (red_issue_sel == 2'd3);
  wire sc_clear_a_lo   = sc_clear_a_pair;
  wire sc_clear_a_hi   = sc_clear_a_pair && !sc_partial_pair;
  wire sc_clear_b_lo   = sc_clear_b_pair;
  wire sc_clear_b_hi   = sc_clear_b_pair && !sc_partial_pair;
  wire sc_clear_c_lo   = sc_clear_c_pair;
  wire sc_clear_c_hi   = sc_clear_c_pair && !sc_partial_pair;
  wire sc_clear_d_lo   = sc_clear_d_pair;
  wire sc_clear_d_hi   = sc_clear_d_pair && !sc_partial_pair;

  wire sc_red_a_lo_valid_i = sc_mul_a_v_out && (mul_pos_sel == 2'd0);
  wire sc_red_a_hi_valid_i = sc_mul_b_v_out && (mul_pos_sel == 2'd0);
  wire sc_red_b_lo_valid_i = sc_mul_a_v_out && (mul_pos_sel == 2'd1);
  wire sc_red_b_hi_valid_i = sc_mul_b_v_out && (mul_pos_sel == 2'd1);
  wire sc_red_c_lo_valid_i = sc_mul_a_v_out && (mul_pos_sel == 2'd2);
  wire sc_red_c_hi_valid_i = sc_mul_b_v_out && (mul_pos_sel == 2'd2);
  wire sc_red_d_lo_valid_i = sc_mul_a_v_out && (mul_pos_sel == 2'd3);
  wire sc_red_d_hi_valid_i = sc_mul_b_v_out && (mul_pos_sel == 2'd3);

  wire sc_red_a_lo_flush_i = sc_red_a_lo_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_a_hi_flush_i = sc_red_a_hi_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_b_lo_flush_i = sc_red_b_lo_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_b_hi_flush_i = sc_red_b_hi_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_c_lo_flush_i = sc_red_c_lo_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_c_hi_flush_i = sc_red_c_hi_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_d_lo_flush_i = sc_red_d_lo_valid_i && (sc_red_in_cnt == 5'd15);
  wire sc_red_d_hi_flush_i = sc_red_d_hi_valid_i && (sc_red_in_cnt == 5'd15);

  wire        sc_red_a_lo_done, sc_red_a_hi_done;
  wire        sc_red_b_lo_done, sc_red_b_hi_done;
  wire        sc_red_c_lo_done, sc_red_c_hi_done;
  wire        sc_red_d_lo_done, sc_red_d_hi_done;
  wire [15:0] sc_red_a_lo_sum,  sc_red_a_hi_sum;
  wire [15:0] sc_red_b_lo_sum,  sc_red_b_hi_sum;
  wire [15:0] sc_red_c_lo_sum,  sc_red_c_hi_sum;
  wire [15:0] sc_red_d_lo_sum,  sc_red_d_hi_sum;

  fp16_reduce_k8 u_sc_red_a_lo (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_a_lo), .valid_i(sc_red_a_lo_valid_i),
    .data_i (sc_mac_prod_a), .flush_i(sc_red_a_lo_flush_i),
    .done_o (sc_red_a_lo_done), .sum_o(sc_red_a_lo_sum)
  );
  fp16_reduce_k8 u_sc_red_a_hi (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_a_hi), .valid_i(sc_red_a_hi_valid_i),
    .data_i (sc_mac_prod_b), .flush_i(sc_red_a_hi_flush_i),
    .done_o (sc_red_a_hi_done), .sum_o(sc_red_a_hi_sum)
  );
  fp16_reduce_k8 u_sc_red_b_lo (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_b_lo), .valid_i(sc_red_b_lo_valid_i),
    .data_i (sc_mac_prod_a), .flush_i(sc_red_b_lo_flush_i),
    .done_o (sc_red_b_lo_done), .sum_o(sc_red_b_lo_sum)
  );
  fp16_reduce_k8 u_sc_red_b_hi (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_b_hi), .valid_i(sc_red_b_hi_valid_i),
    .data_i (sc_mac_prod_b), .flush_i(sc_red_b_hi_flush_i),
    .done_o (sc_red_b_hi_done), .sum_o(sc_red_b_hi_sum)
  );
  fp16_reduce_k8 u_sc_red_c_lo (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_c_lo), .valid_i(sc_red_c_lo_valid_i),
    .data_i (sc_mac_prod_a), .flush_i(sc_red_c_lo_flush_i),
    .done_o (sc_red_c_lo_done), .sum_o(sc_red_c_lo_sum)
  );
  fp16_reduce_k8 u_sc_red_c_hi (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_c_hi), .valid_i(sc_red_c_hi_valid_i),
    .data_i (sc_mac_prod_b), .flush_i(sc_red_c_hi_flush_i),
    .done_o (sc_red_c_hi_done), .sum_o(sc_red_c_hi_sum)
  );
  fp16_reduce_k8 u_sc_red_d_lo (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_d_lo), .valid_i(sc_red_d_lo_valid_i),
    .data_i (sc_mac_prod_a), .flush_i(sc_red_d_lo_flush_i),
    .done_o (sc_red_d_lo_done), .sum_o(sc_red_d_lo_sum)
  );
  fp16_reduce_k8 u_sc_red_d_hi (
    .clk_i(clk_i), .rst_i(rst_i),
    .clear_i(sc_clear_d_hi), .valid_i(sc_red_d_hi_valid_i),
    .data_i (sc_mac_prod_b), .flush_i(sc_red_d_hi_flush_i),
    .done_o (sc_red_d_hi_done), .sum_o(sc_red_d_hi_sum)
  );

  // Pend latches: lo and hi of the same pair pulse done_o on the same cycle,
  // so the serial cap_state cannot catch both. Set on done, clear when
  // cap_state captures that reducer. red_cap_sel walks 0..7 through a_lo,
  // a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi in pos order
  reg sc_red_a_lo_pend, sc_red_a_hi_pend;
  reg sc_red_b_lo_pend, sc_red_b_hi_pend;
  reg sc_red_c_lo_pend, sc_red_c_hi_pend;
  reg sc_red_d_lo_pend, sc_red_d_hi_pend;

  reg        sc_red_done;
  reg [15:0] sc_red_sum;
  always @(*) begin
    case (red_cap_sel)
      3'd0: begin sc_red_done = sc_red_a_lo_pend; sc_red_sum = sc_red_a_lo_sum; end
      3'd1: begin sc_red_done = sc_red_a_hi_pend; sc_red_sum = sc_red_a_hi_sum; end
      3'd2: begin sc_red_done = sc_red_b_lo_pend; sc_red_sum = sc_red_b_lo_sum; end
      3'd3: begin sc_red_done = sc_red_b_hi_pend; sc_red_sum = sc_red_b_hi_sum; end
      3'd4: begin sc_red_done = sc_red_c_lo_pend; sc_red_sum = sc_red_c_lo_sum; end
      3'd5: begin sc_red_done = sc_red_c_hi_pend; sc_red_sum = sc_red_c_hi_sum; end
      3'd6: begin sc_red_done = sc_red_d_lo_pend; sc_red_sum = sc_red_d_lo_sum; end
      3'd7: begin sc_red_done = sc_red_d_hi_pend; sc_red_sum = sc_red_d_hi_sum; end
    endcase
  end

  // Final dot product scaled by 1/sqrt(d_k) and converted to Q16.7. Pipeline
  // register sc_scaled_r splits the long fp16_mul_comb -> fp16_to_q167 chain
  reg [15:0] sc_dot_r;
  reg [15:0] sc_scaled_r;

  wire [15:0] sc_scaled;
  fp16_mul_comb u_sc_scale (
    .a_i(sc_dot_r),
    .b_i(INV_SQRT_DK),
    .prod_o(sc_scaled)
  );

  always @(posedge clk_i) sc_scaled_r <= sc_scaled;

  // Convert fp16 score to Q16.7 for softmax
  wire [23:0] sc_q167;
  fp16_to_q167 u_sc_cvt (
    .val_i(sc_scaled_r),
    .q167_o(sc_q167)
  );

  // AV: Q1.15 -> fp16 for both attn_buf entries of the current pair, then
  // (attn_a*V_lo + attn_b*V_hi) summed via fp16_add, accumulated into av_acc[d].
  // For partial pair (pos_r even, last pair) attn_b is forced to 0 so the
  // upper position contributes nothing
  reg [7:0]  av_pos_r1;
  reg        av_partial_r1;
  reg [15:0] attn_val_a_r, attn_val_b_r;
  wire av_partial = (av_pos == pos_r) && !pos_r[0];
  always @(posedge clk_i) begin
    av_pos_r1     <= av_pos;
    av_partial_r1 <= av_partial;
    attn_val_a_r  <= av_head_idx[0] ? attn_buf_b[av_pos_r1]
                                    : attn_buf_a[av_pos_r1];
    attn_val_b_r  <= av_partial_r1 ? 16'd0
                                   : (av_head_idx[0] ? attn_buf_b[av_pos_r1 + 8'd1]
                                                     : attn_buf_a[av_pos_r1 + 8'd1]);
  end

  wire [15:0] av_attn_a_fp16, av_attn_b_fp16;
  q115_to_fp16 u_av_cvt_a (.val_i(attn_val_a_r), .fp16_o(av_attn_a_fp16));
  q115_to_fp16 u_av_cvt_b (.val_i(attn_val_b_r), .fp16_o(av_attn_b_fp16));

  reg [15:0] av_attn_a_fp16_r, av_attn_b_fp16_r;
  always @(posedge clk_i) begin
    av_attn_a_fp16_r <= av_attn_a_fp16;
    av_attn_b_fp16_r <= av_attn_b_fp16;
  end

  wire av_issue = (av_state == AV_RUN) && (av_cnt < 5'd16);
  wire av_mul_v_in = av_bram_v[2];

  wire        av_mul_a_v_out, av_mul_b_v_out;
  wire [15:0] av_mac_prod_a,  av_mac_prod_b;
  fp16_mul u_av_mul_a (
    .clk_i(clk_i),
    .valid_i(av_mul_v_in),
    .a_i(av_attn_a_fp16_r),
    .b_i(v_rdata_i[15:0]),
    .valid_o(av_mul_a_v_out),
    .prod_o(av_mac_prod_a)
  );
  fp16_mul u_av_mul_b (
    .clk_i(clk_i),
    .valid_i(av_mul_v_in),
    .a_i(av_attn_b_fp16_r),
    .b_i(v_rdata_i[31:16]),
    .valid_o(av_mul_b_v_out),
    .prod_o(av_mac_prod_b)
  );

  // Sum the two products into the running accumulator value across two adders.
  // sum_add: prod_a + prod_b -> temp. acc_add: av_acc[d] + temp -> av_acc[d]
  wire        av_sum_v_out;
  wire [15:0] av_sum_temp;
  fp16_add u_av_sum_add (
    .clk_i(clk_i),
    .valid_i(av_mul_a_v_out),
    .a_i(av_mac_prod_a),
    .b_i(av_mac_prod_b),
    .valid_o(av_sum_v_out),
    .sum_o(av_sum_temp)
  );

  wire [3:0] av_dim_at_add_in  = av_dim_pipe[8];
  wire [3:0] av_dim_at_add_out = av_dim_pipe[12];

  wire        av_add_v_out;
  wire [15:0] av_mac_sum;
  fp16_add u_av_acc_add (
    .clk_i(clk_i),
    .valid_i(av_sum_v_out),
    .a_i(av_acc[av_dim_at_add_in]),
    .b_i(av_sum_temp),
    .valid_o(av_add_v_out),
    .sum_o(av_mac_sum)
  );

  integer j;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state          <= S_IDLE;
      score_state    <= SC_IDLE;
      av_state       <= AV_IDLE;
      done_o         <= 1'b0;
      qkv_start      <= 1'b0;
      proj_start     <= 1'b0;
      sm_start       <= 1'b0;
      sm_in_valid    <= 1'b0;
      k_we_o         <= 1'b0;
      v_we_o         <= 1'b0;
      av_bram_v      <= 3'b000;
      sc_bram_v      <= 4'b0000;
      av_issue_done  <= 1'b0;
      score_head_idx <= 3'd0;
      av_head_idx    <= 3'd0;
      scored_count   <= 4'd0;
      av_done_count  <= 4'd0;
      sc_red_a_lo_pend <= 1'b0;
      sc_red_a_hi_pend <= 1'b0;
      sc_red_b_lo_pend <= 1'b0;
      sc_red_b_hi_pend <= 1'b0;
      sc_red_c_lo_pend <= 1'b0;
      sc_red_c_hi_pend <= 1'b0;
      sc_red_d_lo_pend <= 1'b0;
      sc_red_d_hi_pend <= 1'b0;
      sm_a_busy      <= 1'b0;
      sm_b_busy      <= 1'b0;
      sm_a_out_cnt   <= 8'd0;
      sm_b_out_cnt   <= 8'd0;
    end else begin
      done_o      <= 1'b0;
      qkv_start   <= 1'b0;
      proj_start  <= 1'b0;
      sm_start    <= 1'b0;
      sm_in_valid <= 1'b0;
      k_we_o      <= 1'b0;
      v_we_o      <= 1'b0;

      // Per-instance busy track + output capture, runs concurrently with score
      if (sm_start && sm_a_active) begin sm_a_busy <= 1'b1; sm_a_out_cnt <= 8'd0; end
      if (sm_start && sm_b_active) begin sm_b_busy <= 1'b1; sm_b_out_cnt <= 8'd0; end
      if (sm_a_done) sm_a_busy <= 1'b0;
      if (sm_b_done) sm_b_busy <= 1'b0;
      if (sm_a_out_valid) begin
        attn_buf_a[sm_a_out_cnt] <= sm_a_out_data;
        sm_a_out_cnt <= sm_a_out_cnt + 8'd1;
      end
      if (sm_b_out_valid) begin
        attn_buf_b[sm_b_out_cnt] <= sm_b_out_data;
        sm_b_out_cnt <= sm_b_out_cnt + 8'd1;
      end
      scored_count <= scored_count
                    + {3'd0, sm_a_done}
                    + {3'd0, sm_b_done};

      av_bram_v <= {av_bram_v[1:0], av_issue};
      av_dim_pipe[0] <= av_cnt[3:0];
      for (j = 1; j < 13; j = j + 1) av_dim_pipe[j] <= av_dim_pipe[j-1];
      if (av_add_v_out) begin
        av_acc[av_dim_at_add_out] <= av_mac_sum;
      end

      sc_bram_v <= {sc_bram_v[2:0], sc_issue};
      sc_dim_pipe[0] <= sc_cnt[3:0];
      sc_dim_pipe[1] <= sc_dim_pipe[0];
      sc_dim_pipe[2] <= sc_dim_pipe[1];
      sc_dim_pipe[3] <= sc_dim_pipe[2];

      // sc_red_in_cnt wraps every 16 mul outputs (one pair worth of dims),
      // advancing mul_pos_sel so the next pair routes to the next reducer
      // in the 4-way ping-pong
      if (sc_mul_a_v_out) begin
        if (sc_red_in_cnt == 5'd15) begin
          sc_red_in_cnt <= 5'd0;
          mul_pos_sel   <= mul_pos_sel + 2'd1;
        end else begin
          sc_red_in_cnt <= sc_red_in_cnt + 5'd1;
        end
      end

      if (sc_red_a_lo_done) sc_red_a_lo_pend <= 1'b1;
      if (sc_red_a_hi_done) sc_red_a_hi_pend <= 1'b1;
      if (sc_red_b_lo_done) sc_red_b_lo_pend <= 1'b1;
      if (sc_red_b_hi_done) sc_red_b_hi_pend <= 1'b1;
      if (sc_red_c_lo_done) sc_red_c_lo_pend <= 1'b1;
      if (sc_red_c_hi_done) sc_red_c_hi_pend <= 1'b1;
      if (sc_red_d_lo_done) sc_red_d_lo_pend <= 1'b1;
      if (sc_red_d_hi_done) sc_red_d_hi_pend <= 1'b1;

      if (score_state == SC_SCORE) begin
        case (cap_state)
          C_WAIT: begin
            if (sc_red_done && !cap_done) begin
              sc_dot_r  <= sc_red_sum;
              cap_state <= C_MUL;
            end
          end
          C_MUL: cap_state <= C_FEED;
          C_FEED: begin
            sm_in_valid <= 1'b1;
            sm_in_data  <= sc_q167;
            red_cap_sel <= red_cap_sel + 3'd1;
            case (red_cap_sel)
              3'd0: sc_red_a_lo_pend <= 1'b0;
              3'd1: sc_red_a_hi_pend <= 1'b0;
              3'd2: sc_red_b_lo_pend <= 1'b0;
              3'd3: sc_red_b_hi_pend <= 1'b0;
              3'd4: sc_red_c_lo_pend <= 1'b0;
              3'd5: sc_red_c_hi_pend <= 1'b0;
              3'd6: sc_red_d_lo_pend <= 1'b0;
              3'd7: sc_red_d_hi_pend <= 1'b0;
            endcase
            if (sc_pos_cap == pos_r) begin
              cap_done  <= 1'b1;
              cap_state <= C_IDLE;
            end else begin
              sc_pos_cap <= sc_pos_cap + 8'd1;
              cap_state  <= C_WAIT;
            end
          end
          default: ;
        endcase
      end

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state     <= S_QKV;
            layer_r   <= layer_i;
            pos_r     <= pos_i;
            qkv_start <= 1'b1;
          end
        end

        S_QKV: begin
          if (qkv_done) begin
            state    <= S_KV_STORE;
            kv_cnt   <= 9'd0;
            kv_phase <= 2'd0;
          end
        end

        // Write K[pos] and V[pos] to caches (both fp16, no requant)
        // First 128 cycles: K, next 128 cycles: V
        S_KV_STORE: begin
          k_layer_o <= layer_r;
          k_pos_o   <= pos_r;
          k_head_o  <= kv_idx[6:4];
          k_dim_o   <= kv_idx[3:0];
          v_layer_o <= layer_r;
          v_pos_o   <= pos_r;
          v_head_o  <= kv_idx[6:4];
          v_dim_o   <= kv_idx[3:0];

          if (kv_cnt < 9'd128) begin
            k_wdata_o <= kv_fp16_k;
          end else begin
            v_wdata_o <= kv_fp16_v;
          end

          if (kv_phase == 2'd2) begin
            if (kv_cnt < 9'd128) k_we_o <= 1'b1;
            else                 v_we_o <= 1'b1;
            kv_cnt   <= kv_cnt + 9'd1;
            kv_phase <= 2'd0;
          end else begin
            kv_phase <= kv_phase + 2'd1;
          end

          if (kv_cnt == 9'd255 && kv_phase == 2'd2) begin
            state          <= S_HEADS;
            score_head_idx <= 3'd0;
            av_head_idx    <= 3'd0;
            scored_count   <= 4'd0;
            av_done_count  <= 4'd0;
            score_state    <= SC_SCORE;
            av_state       <= AV_IDLE;
            sm_start       <= 1'b1;
            sc_pos         <= 8'd0;
            sc_pos_cap     <= 8'd0;
            sc_cnt         <= 5'd0;
            sc_bram_v      <= 4'b0000;
            sc_red_in_cnt  <= 5'd0;
            red_issue_sel  <= 2'd0;
            red_cap_sel    <= 3'd0;
            mul_pos_sel    <= 2'd0;
            issue_done     <= 1'b0;
            cap_done       <= 1'b0;
            cap_state      <= C_WAIT;
            sc_red_a_lo_pend <= 1'b0;
            sc_red_a_hi_pend <= 1'b0;
            sc_red_b_lo_pend <= 1'b0;
            sc_red_b_hi_pend <= 1'b0;
            sc_red_c_lo_pend <= 1'b0;
            sc_red_c_hi_pend <= 1'b0;
            sc_red_d_lo_pend <= 1'b0;
            sc_red_d_hi_pend <= 1'b0;
          end
        end

        S_HEADS: begin
          // Score sub-FSM
          case (score_state)
            SC_SCORE: begin
              k_layer_o <= layer_r;
              k_head_o  <= score_head_idx;
              k_we_o    <= 1'b0;

              if (!issue_done && sc_cnt < 5'd16) begin
                k_pos_o <= sc_pos;
                k_dim_o <= sc_cnt[3:0];
                sc_cnt  <= sc_cnt + 5'd1;
              end

              if (!issue_done && sc_cnt == 5'd15) begin
                if (sc_pos[7:1] == pos_r[7:1]) begin
                  issue_done <= 1'b1;
                end else begin
                  sc_pos        <= sc_pos + 8'd2;
                  sc_cnt        <= 5'd0;
                  red_issue_sel <= red_issue_sel + 2'd1;
                end
              end

              if (issue_done && cap_done) begin
                if (pos_r == 8'd255) begin
                  // All scores fed; either advance to next head or finalize
                  if (score_head_idx == 3'd7) score_state <= SC_SM_WAIT;
                  else                        score_state <= SC_WAIT;
                end else begin
                  score_state <= SC_PAD;
                  pad_cnt     <= {1'b0, pos_r} + 9'd1;
                end
              end
            end

            SC_PAD: begin
              sm_in_valid <= 1'b1;
              sm_in_data  <= 24'sh800000;
              pad_cnt     <= pad_cnt + 9'd1;
              if (pad_cnt == 9'd255) begin
                if (score_head_idx == 3'd7) score_state <= SC_SM_WAIT;
                else                        score_state <= SC_WAIT;
              end
            end

            // Last head only: wait for both softmax instances to finish before
            // signalling SC_DONE. Capture is handled in the always block above
            SC_SM_WAIT: begin
              if (!sm_a_busy && !sm_b_busy) score_state <= SC_DONE;
            end

            // Hold off until AV has freed the attn_buf and the target softmax
            // instance is idle
            SC_WAIT: begin
              if ({1'b0, score_head_idx} <= av_done_count
                  && (score_head_idx[0] ? !sm_a_busy : !sm_b_busy)) begin
                score_head_idx <= score_head_idx + 3'd1;
                score_state    <= SC_SCORE;
                sm_start       <= 1'b1;
                sc_pos         <= 8'd0;
                sc_pos_cap     <= 8'd0;
                sc_cnt         <= 5'd0;
                sc_bram_v      <= 4'b0000;
                sc_red_in_cnt  <= 5'd0;
                red_issue_sel  <= 2'd0;
                red_cap_sel    <= 3'd0;
                mul_pos_sel    <= 2'd0;
                issue_done     <= 1'b0;
                cap_done       <= 1'b0;
                cap_state      <= C_WAIT;
                sc_red_a_lo_pend <= 1'b0;
                sc_red_a_hi_pend <= 1'b0;
                sc_red_b_lo_pend <= 1'b0;
                sc_red_b_hi_pend <= 1'b0;
                sc_red_c_lo_pend <= 1'b0;
                sc_red_c_hi_pend <= 1'b0;
                sc_red_d_lo_pend <= 1'b0;
                sc_red_d_hi_pend <= 1'b0;
              end
            end

            default: ;
          endcase

          // AV sub-FSM
          case (av_state)
            AV_IDLE: begin
              // Wait for first head's softmax to fill its attn_buf
              if (scored_count >= 4'd1) begin
                av_state      <= AV_RUN;
                av_pos        <= 8'd0;
                av_cnt        <= 5'd0;
                av_bram_v     <= 3'b000;
                av_issue_done <= 1'b0;
                for (j = 0; j < 16; j = j + 1) av_acc[j] <= 16'd0;
              end
            end

            // AV: av_acc[d] += attn_fp16[p] * V_fp16[p][d] for d=0..15, p=0..pos
            AV_RUN: begin
              v_layer_o <= layer_r;
              v_head_o  <= av_head_idx;
              v_we_o    <= 1'b0;

              if (!av_issue_done) begin
                if (av_cnt < 5'd16) begin
                  v_pos_o <= av_pos;
                  v_dim_o <= av_cnt[3:0];
                end

                if (av_cnt == 5'd15) begin
                  if (av_pos[7:1] == pos_r[7:1]) begin
                    av_issue_done <= 1'b1;
                    av_cnt        <= av_cnt + 5'd1;
                  end else begin
                    av_pos <= av_pos + 8'd2;
                    av_cnt <= 5'd0;
                  end
                end else begin
                  av_cnt <= av_cnt + 5'd1;
                end
              end

              if (av_issue_done && av_add_v_out && av_dim_at_add_out == 4'd15) begin
                av_state <= AV_STORE;
              end
            end

            AV_STORE: begin
              for (j = 0; j < 16; j = j + 1) begin
                head_ram[av_head_idx * 16 + j] <= av_acc[j];
              end
              av_done_count <= av_done_count + 4'd1;
              if (av_head_idx == 3'd7) av_state <= AV_DONE;
              else                     av_state <= AV_WAIT;
            end

            // Wait for next head's softmax done
            AV_WAIT: begin
              if (scored_count >= {1'b0, av_head_idx} + 4'd2) begin
                av_head_idx   <= av_head_idx + 3'd1;
                av_state      <= AV_RUN;
                av_pos        <= 8'd0;
                av_cnt        <= 5'd0;
                av_bram_v     <= 3'b000;
                av_issue_done <= 1'b0;
                for (j = 0; j < 16; j = j + 1) av_acc[j] <= 16'd0;
              end
            end

            default: ;
          endcase

          // Exit when last head's AV+store is done
          if (av_state == AV_DONE) begin
            state      <= S_PROJ;
            proj_start <= 1'b1;
          end
        end

        S_PROJ: begin
          // proj writes directly to shared RAM via proj_res_we/waddr/wdata
          if (proj_done) begin
            state <= S_DONE;
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