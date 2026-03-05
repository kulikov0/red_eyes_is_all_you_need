// Multi-head self-attention for single-token autoregressive inference (W8A16)
//
// Flow: QKV matvec_fp16 -> KV store -> 8x(score, softmax, AV) -> proj matvec_fp16
// Submodules: 2x matvec_fp16, 1x softmax (reused per head),
//             fp16_mul_comb, fp16_add_comb, fp16_to_q167, q115_to_fp16
//
// Precision:
//   Weights: int8 (from weight_store BRAM), dequanted to fp16 via scale
//   QKV: fp16 output (128 Q + 128 K + 128 V)
//   K/V cache: fp16 (16-bit), stored directly (no requant)
//   Score: fp16 dot(Q,K) * 1/sqrt(16), converted to Q16.7 for softmax
//   Softmax: Q16.7 input (IN_W=24), Q1.15 output (bipartite LUT)
//   AV: Q1.15 -> fp16, then fp16 accumulation
//   Proj: matvec_fp16, fp16 output
//
// Weight store tensor_sel:
//   QKV = {layer_i, 3'b000} + 4   (4, 12, 20, 28)
//   Proj = {layer_i, 3'b000} + 5  (5, 13, 21, 29)
//
// KV cache 2-cycle read latency pipeline:
//   Cycle T:   set addr -> cache latches at posedge T+1
//   Cycle T+1: BRAM reads internally, sel_r latches
//   Cycle T+2: rdata_o valid
//
// Latency: 49154 + 256 + 8*(36*(P+1) + 517) + 16386 + 10 cycles
//   P = position (0..255). At P=255: ~143,678 cycles

module attention (
  input  wire          clk_i,
  input  wire          rst_i,
  input  wire          start_i,
  input  wire [1:0]    layer_i,
  input  wire [7:0]    pos_i,
  input  wire [2047:0] x_i,

  // Weight store (active during S_QKV and S_PROJ)
  output wire [5:0]    w_sel_o,
  output wire [15:0]   w_addr_o,
  input  wire [7:0]    w_data_i,

  // K cache (fp16)
  output reg           k_we_o,
  output reg  [15:0]   k_wdata_o,
  input  wire [15:0]   k_rdata_i,

  // V cache (fp16)
  output reg           v_we_o,
  output reg  [15:0]   v_wdata_o,
  input  wire [15:0]   v_rdata_i,

  // Shared KV address (K and V never accessed simultaneously)
  output reg  [1:0]    kv_layer_o,
  output reg  [2:0]    kv_head_o,
  output reg  [7:0]    kv_pos_o,
  output reg  [3:0]    kv_dim_o,

  // Output
  output reg  [2047:0] out_vec_o,
  output reg           done_o
);

  `include "weight_scales.vh"

  // 1/sqrt(HEAD_DIM) = 1/sqrt(16) = 0.25
  localparam [15:0] INV_SQRT_DK = 16'h3400;

  // FSM states
  localparam [3:0] S_IDLE      = 4'd0,
                   S_QKV       = 4'd1,
                   S_KV_STORE  = 4'd2,
                   S_SCORE     = 4'd3,
                   S_SCORE_PAD = 4'd4,
                   S_SM_WAIT   = 4'd5,
                   S_AV        = 4'd6,
                   S_AV_STORE  = 4'd7,
                   S_NEXT_HEAD = 4'd8,
                   S_PROJ      = 4'd9,
                   S_DONE      = 4'd10;

  reg [3:0] state;

  // Latched inputs
  reg [1:0] layer_r;
  reg [7:0] pos_r;

  // Per-layer scale selection
  reg [15:0] qkv_scale;
  reg [15:0] proj_scale;
  always @(*) begin
    case (layer_r)
      2'd0: begin
        qkv_scale  = SCALE_BLOCK0_ATTN_QKV_WEIGHT;
        proj_scale = SCALE_BLOCK0_ATTN_PROJ_WEIGHT;
      end
      2'd1: begin
        qkv_scale  = SCALE_BLOCK1_ATTN_QKV_WEIGHT;
        proj_scale = SCALE_BLOCK1_ATTN_PROJ_WEIGHT;
      end
      2'd2: begin
        qkv_scale  = SCALE_BLOCK2_ATTN_QKV_WEIGHT;
        proj_scale = SCALE_BLOCK2_ATTN_PROJ_WEIGHT;
      end
      2'd3: begin
        qkv_scale  = SCALE_BLOCK3_ATTN_QKV_WEIGHT;
        proj_scale = SCALE_BLOCK3_ATTN_PROJ_WEIGHT;
      end
    endcase
  end

  // Head iteration
  reg [2:0] head_idx;

  // QKV matvec (fp16 output)
  reg         qkv_start;
  wire [15:0] qkv_addr;
  wire [384*16-1:0] qkv_out;
  wire        qkv_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(384)) u_qkv (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (qkv_start),
    .in_vec_i     (x_i),
    .scale_i      (qkv_scale),
    .weight_addr_o(qkv_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (qkv_out),
    .done_o       (qkv_done)
  );

  // Proj matvec (fp16 output)
  reg          proj_start;
  wire [13:0]  proj_addr;
  reg  [2047:0] head_out_buf;
  wire [2047:0] proj_out;
  wire         proj_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(128)) u_proj (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (proj_start),
    .in_vec_i     (head_out_buf),
    .scale_i      (proj_scale),
    .weight_addr_o(proj_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (proj_out),
    .done_o       (proj_done)
  );

  // Softmax (Q16.7 input, Q1.15 output)
  reg         sm_start;
  reg         sm_in_valid;
  reg  [23:0] sm_in_data;
  wire        sm_out_valid;
  wire [15:0] sm_out_data;
  wire        sm_done;

  softmax #(.N(256), .IN_W(24)) u_sm (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .start_i    (sm_start),
    .in_valid_i (sm_in_valid),
    .in_data_i  (sm_in_data),
    .in_ready_o (),
    .out_valid_o(sm_out_valid),
    .out_data_o (sm_out_data),
    .done_o     (sm_done)
  );

  // QKV buffer: 384 x fp16
  reg [384*16-1:0] qkv_buf;

  // Attention buffer: 256 x 16-bit softmax outputs (Q1.15)
  reg [15:0] attn_buf [0:255];

  // AV accumulators: 16 x fp16
  reg [15:0] av_acc [0:15];

  // KV store counter (0..255: first 128 = K, next 128 = V)
  reg [8:0] kv_cnt;

  // Score computation pipeline
  reg [4:0]  sc_cnt;
  reg [7:0]  sc_pos;
  reg [15:0] score_acc;  // fp16 accumulator
  reg [1:0]  sc_valid;
  reg [3:0]  sc_dim_d1;
  reg [3:0]  sc_dim_d2;

  // AV computation pipeline
  reg [4:0]  av_cnt;
  reg [7:0]  av_pos;
  reg [1:0]  av_valid;
  reg [3:0]  av_dim_d1;
  reg [3:0]  av_dim_d2;

  // Softmax output capture counter
  reg [8:0] sm_out_cnt;
  reg [8:0] pad_cnt;  // counter for S_SCORE_PAD


  // Q_head extraction: 16 fp16 values for current head
  wire [15:0] q_head [0:15];
  genvar gi;
  generate
    for (gi = 0; gi < 16; gi = gi + 1) begin : gen_q
      assign q_head[gi] = qkv_buf[(head_idx * 16 + gi) * 16 +: 16];
    end
  endgenerate

  // KV store: extract fp16 K and V for current kv_cnt index
  wire [6:0] kv_idx = kv_cnt[6:0];
  wire [15:0] kv_fp16_k = qkv_buf[(9'd128 + {2'b0, kv_idx}) * 16 +: 16];
  wire [15:0] kv_fp16_v = qkv_buf[(9'd256 + {2'b0, kv_idx}) * 16 +: 16];

  // Score pipeline: fp16 Q*K dot product -> scale -> Q16.7 for softmax
  wire [15:0] sc_mac_prod;
  fp16_mul_comb u_sc_mul (
    .a_i(q_head[sc_dim_d2]),
    .b_i(k_rdata_i),
    .prod_o(sc_mac_prod)
  );

  wire [15:0] sc_mac_sum;
  fp16_add_comb u_sc_add (
    .a_i(score_acc),
    .b_i(sc_mac_prod),
    .sum_o(sc_mac_sum)
  );

  // Scale the final dot product by 1/sqrt(d_k)
  wire [15:0] sc_scaled;
  fp16_mul_comb u_sc_scale (
    .a_i(sc_mac_sum),
    .b_i(INV_SQRT_DK),
    .prod_o(sc_scaled)
  );

  // Convert fp16 score to Q16.7 for softmax (24-bit, no overflow)
  wire [23:0] sc_q167;
  fp16_to_q167 u_sc_cvt (
    .val_i(sc_scaled),
    .q167_o(sc_q167)
  );

  // AV pipeline: Q1.15 -> fp16, then fp16 * V, accumulated
  wire [15:0] av_attn_fp16;
  q115_to_fp16 u_av_cvt (
    .val_i(attn_buf[av_pos]),
    .fp16_o(av_attn_fp16)
  );

  wire [15:0] av_mac_prod;
  fp16_mul_comb u_av_mul (
    .a_i(av_attn_fp16),
    .b_i(v_rdata_i),
    .prod_o(av_mac_prod)
  );

  wire [15:0] av_mac_sum;
  fp16_add_comb u_av_add (
    .a_i(av_acc[av_dim_d2]),
    .b_i(av_mac_prod),
    .sum_o(av_mac_sum)
  );

  // Combinational weight store address mux
  reg [5:0]  w_sel_r;
  reg [15:0] w_addr_r;
  assign w_sel_o  = w_sel_r;
  assign w_addr_o = w_addr_r;

  always @(*) begin
    case (state)
      S_QKV: begin
        w_sel_r  = {layer_r, 3'b000} + 6'd4;
        w_addr_r = qkv_addr;
      end
      S_PROJ: begin
        w_sel_r  = {layer_r, 3'b000} + 6'd5;
        w_addr_r = {2'b00, proj_addr};
      end
      default: begin
        w_sel_r  = 6'd0;
        w_addr_r = 16'd0;
      end
    endcase
  end

  integer j;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      done_o      <= 1'b0;
      qkv_start   <= 1'b0;
      proj_start  <= 1'b0;
      sm_start    <= 1'b0;
      sm_in_valid <= 1'b0;
      k_we_o      <= 1'b0;
      v_we_o      <= 1'b0;
    end else begin
      done_o      <= 1'b0;
      qkv_start   <= 1'b0;
      proj_start  <= 1'b0;
      sm_start    <= 1'b0;
      sm_in_valid <= 1'b0;
      k_we_o      <= 1'b0;
      v_we_o      <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state     <= S_QKV;
            layer_r   <= layer_i;
            pos_r     <= pos_i;
            qkv_start <= 1'b1;
          end
        end

        // QKV matvec: fp16 output
        S_QKV: begin
          if (qkv_done) begin
            qkv_buf <= qkv_out;
            state   <= S_KV_STORE;
            kv_cnt  <= 9'd0;
          end
        end

        // Write K[pos] and V[pos] to caches (both fp16, no requant)
        // First 128 cycles: K, next 128 cycles: V
        S_KV_STORE: begin
          kv_layer_o <= layer_r;
          kv_pos_o   <= pos_r;
          kv_head_o  <= kv_idx[6:4];
          kv_dim_o   <= kv_idx[3:0];

          if (kv_cnt < 9'd128) begin
            k_we_o    <= 1'b1;
            k_wdata_o <= kv_fp16_k;
          end else begin
            v_we_o    <= 1'b1;
            v_wdata_o <= kv_fp16_v;
          end

          if (kv_cnt == 9'd255) begin
            state    <= S_SCORE;
            head_idx <= 3'd0;
            sm_start <= 1'b1;
            sc_pos   <= 8'd0;
            sc_cnt   <= 5'd0;
            score_acc <= 16'd0;
            sc_valid <= 2'b00;
          end
          kv_cnt <= kv_cnt + 9'd1;
        end

        // Score: fp16 Q . K[p] for p = 0..pos
        // Pipeline: issue addr at sc_cnt=0..15, data valid at sc_cnt=2..17
        S_SCORE: begin
          kv_layer_o <= layer_r;
          kv_head_o  <= head_idx;
          k_we_o     <= 1'b0;

          // Dim delay line
          sc_dim_d1 <= sc_cnt[3:0];
          sc_dim_d2 <= sc_dim_d1;

          // Valid pipeline
          sc_valid <= {sc_valid[0], (sc_cnt < 5'd16) ? 1'b1 : 1'b0};

          // Issue K cache read address for dims 0..15
          if (sc_cnt < 5'd16) begin
            kv_pos_o <= sc_pos;
            kv_dim_o <= sc_cnt[3:0];
          end

          sc_cnt <= sc_cnt + 5'd1;

          // MAC when data is valid (2 cycles after addr issue)
          if (sc_valid[1]) begin
            if (sc_dim_d2 == 4'd15) begin
              // Last dim: sc_mac_sum has final dot product
              // sc_scaled = sc_mac_sum * INV_SQRT_DK (combinational)
              // sc_q167 = fp16_to_q167(sc_scaled) (combinational)
              sm_in_valid <= 1'b1;
              sm_in_data  <= sc_q167;
              score_acc   <= 16'd0;

              if (sc_pos == pos_r) begin
                if (pos_r == 8'd255) begin
                  state      <= S_SM_WAIT;
                  sm_out_cnt <= 9'd0;
                end else begin
                  state   <= S_SCORE_PAD;
                  pad_cnt <= {1'b0, pos_r} + 9'd1;
                end
              end else begin
                sc_pos  <= sc_pos + 8'd1;
                sc_cnt  <= 5'd0;
                sc_valid <= 2'b00;
              end
            end else begin
              score_acc <= sc_mac_sum;
            end
          end
        end

        // Pad remaining slots with Q16.7 minimum for softmax
        S_SCORE_PAD: begin
          sm_in_valid <= 1'b1;
          sm_in_data  <= 24'sh800000;
          pad_cnt     <= pad_cnt + 9'd1;
          if (pad_cnt == 9'd255) begin
            state      <= S_SM_WAIT;
            sm_out_cnt <= 9'd0;
          end
        end

        // Capture softmax outputs (Q1.15)
        S_SM_WAIT: begin
          if (sm_out_valid) begin
            attn_buf[sm_out_cnt[7:0]] <= sm_out_data;
            sm_out_cnt <= sm_out_cnt + 9'd1;
          end
          if (sm_done) begin
            state   <= S_AV;
            av_pos  <= 8'd0;
            av_cnt  <= 5'd0;
            av_valid <= 2'b00;
            for (j = 0; j < 16; j = j + 1) begin
              av_acc[j] <= 16'd0;
            end
          end
        end

        // AV: av_acc[d] += attn_fp16[p] * V_fp16[p][d] for d=0..15, p=0..pos
        S_AV: begin
          kv_layer_o <= layer_r;
          kv_head_o  <= head_idx;
          v_we_o     <= 1'b0;

          // Dim delay line
          av_dim_d1 <= av_cnt[3:0];
          av_dim_d2 <= av_dim_d1;

          // Valid pipeline
          av_valid <= {av_valid[0], (av_cnt < 5'd16) ? 1'b1 : 1'b0};

          // Issue V cache read address for dims 0..15
          if (av_cnt < 5'd16) begin
            kv_pos_o <= av_pos;
            kv_dim_o <= av_cnt[3:0];
          end

          av_cnt <= av_cnt + 5'd1;

          // MAC when data valid
          if (av_valid[1]) begin
            av_acc[av_dim_d2] <= av_mac_sum;

            if (av_dim_d2 == 4'd15) begin
              if (av_pos == pos_r) begin
                state <= S_AV_STORE;
              end else begin
                av_pos  <= av_pos + 8'd1;
                av_cnt  <= 5'd0;
                av_valid <= 2'b00;
              end
            end
          end
        end

        // Copy 16 fp16 AV accumulators to head_out_buf (1 cycle)
        S_AV_STORE: begin
          for (j = 0; j < 16; j = j + 1) begin
            head_out_buf[(head_idx * 16 + j) * 16 +: 16] <= av_acc[j];
          end
          state <= S_NEXT_HEAD;
        end

        S_NEXT_HEAD: begin
          if (head_idx == 3'd7) begin
            state      <= S_PROJ;
            proj_start <= 1'b1;
          end else begin
            head_idx  <= head_idx + 3'd1;
            state     <= S_SCORE;
            sm_start  <= 1'b1;
            sc_pos    <= 8'd0;
            sc_cnt    <= 5'd0;
            score_acc <= 16'd0;
            sc_valid  <= 2'b00;
          end
        end

        // Proj matvec: fp16 output
        S_PROJ: begin
          if (proj_done) begin
            out_vec_o <= proj_out;
            state     <= S_DONE;
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