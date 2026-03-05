// Transformer top: full inference pipeline (W8A16)
//
// Embedding -> 4 transformer layers -> ln_f -> head projection -> sampler
// Autoregressive generation loop: output token feeds back as next input
//
// Protocol:
//   Prompt token: set token_i, pulse start_i (generate_i=0)
//     Runs embedding + 4 layers, populates KV cache. done_o pulses.
//   Generate: set token_i, pulse start_i + generate_i
//     Full pipeline through sampler. token_valid_o pulses with output token.
//     Loops autoregressive until pos=255 or external stop.
//
// Weight store: head.weight reuses tok_emb (tensor_sel=0), weight tying
//
// Submodules: embedding, transformer_layer, layernorm (ln_f), matvec_fp16
// (head proj, IN=128 OUT=256), sampler (fp16 argmax)
//
// Latency per token (generate mode): 258 + 4*layer + 656 + 32770 + 258 + 3 cycles
//   P=0: ~536K cycles, P=255: ~1.1M cycles (layer latency grows with position)

module transformer_top (
  input  wire        clk_i,
  input  wire        rst_i,

  // Token interface
  input  wire [7:0]  token_i,
  input  wire        start_i,
  input  wire        generate_i,

  // Weight store interface
  output reg  [5:0]  w_sel_o,
  output reg  [15:0] w_addr_o,
  input  wire [7:0]  w_data_i,

  // K cache (fp16)
  output wire        k_we_o,
  output wire [15:0] k_wdata_o,
  input  wire [15:0] k_rdata_i,

  // V cache (fp16)
  output wire        v_we_o,
  output wire [15:0] v_wdata_o,
  input  wire [15:0] v_rdata_i,

  // KV cache address (shared K/V)
  output wire [1:0]  kv_layer_o,
  output wire [2:0]  kv_head_o,
  output wire [7:0]  kv_pos_o,
  output wire [3:0]  kv_dim_o,

  // Output
  output reg  [7:0]  token_o,
  output reg         token_valid_o,
  output reg         busy_o,
  output reg         done_o
);

  `include "weight_scales.vh"

  // FSM states (no S_LN_F_FEED -- fp16 LN uses flat bus)
  localparam [3:0] S_IDLE         = 4'd0,
                   S_EMBED        = 4'd1,
                   S_LAYER_START  = 4'd2,
                   S_LAYER_WAIT   = 4'd3,
                   S_LN_F_START   = 4'd4,
                   S_LN_F_WAIT    = 4'd5,
                   S_HEAD_PROJ    = 4'd6,
                   S_SAMPLE       = 4'd7,
                   S_TOKEN_OUT    = 4'd8;

  reg [3:0] state;

  // Registers
  reg [2047:0] x_reg;        // current hidden state (128 x fp16)
  reg [7:0]    cur_token;    // token being processed
  reg [7:0]    pos_r;        // current position (auto-increments)
  reg [1:0]    layer_idx;    // which transformer layer (0-3)
  reg          generating;   // mode flag (0=prompt, 1=generate)

  // Embedding
  reg         emb_start;
  wire [5:0]  emb_w_sel;
  wire [15:0] emb_w_addr;
  wire [2047:0] emb_out;
  wire        emb_done;

  embedding u_emb (
    .clk_i     (clk_i),
    .rst_i     (rst_i),
    .start_i   (emb_start),
    .token_id_i(cur_token),
    .position_i(pos_r),
    .tok_scale_i(SCALE_TOK_EMB_WEIGHT),
    .pos_scale_i(SCALE_POS_EMB_WEIGHT),
    .w_sel_o   (emb_w_sel),
    .w_addr_o  (emb_w_addr),
    .w_data_i  (w_data_i),
    .embed_o   (emb_out),
    .done_o    (emb_done),
    .busy_o    ()
  );

  // Transformer layer (reused for all 4 layers)
  reg          tl_start;
  wire [5:0]   tl_w_sel;
  wire [15:0]  tl_w_addr;
  wire [2047:0] tl_out;
  wire         tl_done;

  wire         tl_k_we;
  wire [15:0]  tl_k_wdata;
  wire         tl_v_we;
  wire [15:0]  tl_v_wdata;
  wire [1:0]   tl_kv_layer;
  wire [2:0]   tl_kv_head;
  wire [7:0]   tl_kv_pos;
  wire [3:0]   tl_kv_dim;

  transformer_layer u_tl (
    .clk_i     (clk_i),
    .rst_i     (rst_i),
    .start_i   (tl_start),
    .layer_i   (layer_idx),
    .pos_i     (pos_r),
    .x_i       (x_reg),
    .w_sel_o   (tl_w_sel),
    .w_addr_o  (tl_w_addr),
    .w_data_i  (w_data_i),
    .k_we_o    (tl_k_we),
    .k_wdata_o (tl_k_wdata),
    .k_rdata_i (k_rdata_i),
    .v_we_o    (tl_v_we),
    .v_wdata_o (tl_v_wdata),
    .v_rdata_i (v_rdata_i),
    .kv_layer_o(tl_kv_layer),
    .kv_head_o (tl_kv_head),
    .kv_pos_o  (tl_kv_pos),
    .kv_dim_o  (tl_kv_dim),
    .out_vec_o (tl_out),
    .done_o    (tl_done)
  );

  // KV cache: pass-through from transformer_layer when active, else 0
  reg kv_active;
  assign k_we_o     = kv_active ? tl_k_we     : 1'b0;
  assign k_wdata_o  = kv_active ? tl_k_wdata  : 16'd0;
  assign v_we_o     = kv_active ? tl_v_we     : 1'b0;
  assign v_wdata_o  = kv_active ? tl_v_wdata  : 16'd0;
  assign kv_layer_o = kv_active ? tl_kv_layer : 2'd0;
  assign kv_head_o  = kv_active ? tl_kv_head  : 3'd0;
  assign kv_pos_o   = kv_active ? tl_kv_pos   : 8'd0;
  assign kv_dim_o   = kv_active ? tl_kv_dim   : 4'd0;

  // Final LayerNorm (ln_f, gamma_sel=34, flat bus fp16)
  reg         lnf_start;
  wire [5:0]  lnf_w_sel;
  wire [6:0]  lnf_w_addr;
  wire        lnf_done;
  wire [2047:0] lnf_y;

  layernorm u_ln_f (
    .clk_i       (clk_i),
    .rst_i       (rst_i),
    .start_i     (lnf_start),
    .x_i         (x_reg),
    .w_sel_o     (lnf_w_sel),
    .w_addr_o    (lnf_w_addr),
    .w_data_i    (w_data_i),
    .gamma_sel_i (6'd34),
    .gamma_scale_i(SCALE_LN_F_WEIGHT),
    .beta_scale_i (SCALE_LN_F_BIAS),
    .y_o         (lnf_y),
    .done_o      (lnf_done),
    .busy_o      ()
  );

  // Head projection: matvec_fp16 128->256 (weight-tied with tok_emb)
  reg          head_start;
  wire [14:0]  head_addr;
  wire [256*16-1:0] head_out;
  wire         head_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(256)) u_head_proj (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (head_start),
    .in_vec_i     (x_reg),
    .scale_i      (SCALE_TOK_EMB_WEIGHT),
    .weight_addr_o(head_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (head_out),
    .done_o       (head_done)
  );

  // Sampler (fp16 argmax)
  reg          samp_start;
  wire [7:0]   samp_token;
  wire         samp_done;

  sampler u_samp (
    .clk_i   (clk_i),
    .rst_i   (rst_i),
    .start_i (samp_start),
    .logits_i(head_out),
    .token_o (samp_token),
    .done_o  (samp_done)
  );

  // Weight store mux (combinational)
  always @(*) begin
    case (state)
      S_EMBED: begin
        w_sel_o  = emb_w_sel;
        w_addr_o = emb_w_addr;
      end
      S_LAYER_START, S_LAYER_WAIT: begin
        w_sel_o  = tl_w_sel;
        w_addr_o = tl_w_addr;
      end
      S_LN_F_START, S_LN_F_WAIT: begin
        w_sel_o  = lnf_w_sel;
        w_addr_o = {9'd0, lnf_w_addr};
      end
      S_HEAD_PROJ: begin
        w_sel_o  = 6'd0;
        w_addr_o = {1'b0, head_addr};
      end
      default: begin
        w_sel_o  = 6'd0;
        w_addr_o = 16'd0;
      end
    endcase
  end

  // KV active only during layer processing
  always @(*) begin
    kv_active = (state == S_LAYER_START) || (state == S_LAYER_WAIT);
  end

  // Main FSM
  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      done_o      <= 1'b0;
      token_valid_o <= 1'b0;
      busy_o      <= 1'b0;
      emb_start   <= 1'b0;
      tl_start    <= 1'b0;
      lnf_start   <= 1'b0;
      head_start  <= 1'b0;
      samp_start  <= 1'b0;
      generating  <= 1'b0;
      pos_r       <= 8'd0;
      layer_idx   <= 2'd0;

    end else begin
      done_o        <= 1'b0;
      token_valid_o <= 1'b0;
      emb_start     <= 1'b0;
      tl_start      <= 1'b0;
      lnf_start     <= 1'b0;
      head_start    <= 1'b0;
      samp_start    <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            cur_token  <= token_i;
            if (generate_i) begin
              generating <= 1'b1;
            end
            emb_start  <= 1'b1;
            busy_o     <= 1'b1;
            state      <= S_EMBED;
          end
        end

        // Wait for embedding to complete
        S_EMBED: begin
          if (emb_done) begin
            x_reg     <= emb_out;
            layer_idx <= 2'd0;
            tl_start  <= 1'b1;
            state     <= S_LAYER_WAIT;
          end
        end

        // Pulse tl_start for layers 1-3
        S_LAYER_START: begin
          tl_start <= 1'b1;
          state    <= S_LAYER_WAIT;
        end

        // Wait for transformer_layer to complete
        S_LAYER_WAIT: begin
          if (tl_done) begin
            x_reg     <= tl_out;
            layer_idx <= layer_idx + 2'd1;
            if (layer_idx == 2'd3) begin
              // All 4 layers done
              if (!generating) begin
                // Prompt mode: skip ln_f/head/sampler
                pos_r  <= pos_r + 8'd1;
                done_o <= 1'b1;
                busy_o <= 1'b0;
                state  <= S_IDLE;
              end else begin
                // Generate mode: continue to ln_f
                state <= S_LN_F_START;
              end
            end else begin
              state <= S_LAYER_START;
            end
          end
        end

        // Start final layernorm (flat bus, no streaming)
        S_LN_F_START: begin
          lnf_start <= 1'b1;
          state     <= S_LN_F_WAIT;
        end

        // Wait for ln_f, capture output, start head projection
        S_LN_F_WAIT: begin
          if (lnf_done) begin
            x_reg      <= lnf_y;
            head_start <= 1'b1;
            state      <= S_HEAD_PROJ;
          end
        end

        // Wait for head projection to complete
        S_HEAD_PROJ: begin
          if (head_done) begin
            samp_start <= 1'b1;
            state      <= S_SAMPLE;
          end
        end

        // Wait for sampler to complete
        S_SAMPLE: begin
          if (samp_done) begin
            token_o <= samp_token;
            state   <= S_TOKEN_OUT;
          end
        end

        // Emit token, decide loop or stop
        S_TOKEN_OUT: begin
          token_valid_o <= 1'b1;
          pos_r         <= pos_r + 8'd1;
          if (pos_r == 8'd255) begin
            done_o     <= 1'b1;
            busy_o     <= 1'b0;
            generating <= 1'b0;
            state      <= S_IDLE;
          end else begin
            // Autoregressive: feed output token back
            cur_token <= samp_token;
            emb_start <= 1'b1;
            state     <= S_EMBED;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule