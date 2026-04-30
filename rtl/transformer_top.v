// Transformer top: full inference pipeline (W8A16)
//
// Embedding -> 4 transformer layers -> ln_f -> head projection -> sampler
// Autoregressive generation loop: output token feeds back as next input
//
// Protocol:
//   Prompt token: set token_i, pulse start_i with generate_i=0
//     Runs embedding + 4 layers, populates KV cache. done_o pulses.
//   Generate: set token_i, pulse start_i + generate_i
//     Full pipeline through sampler. token_valid_o pulses with output token.
//     Loops autoregressive until pos=255 or external stop.

module transformer_top (
  input  wire        clk_i,
  input  wire        rst_i,

  // Token interface
  input  wire [7:0]  token_i,
  input  wire        start_i,
  input  wire        generate_i,

  // 8-bit weight store interface for pos_emb and layernorm
  output reg  [5:0]  w_sel_o,
  output reg  [15:0] w_addr_o,
  input  wire [7:0]  w_data_i,
  input  wire [15:0] w_scale_i,

  // 64-bit packed weight store interface for per-layer attention and FF matvecs
  output reg  [3:0]  w8_sel_o,
  output reg  [15:0] w8_addr_o,
  input  wire [63:0] w8_data_i,
  input  wire [15:0] w8_scale_i,

  // Dedicated tok_emb bus shared between embedding and head_proj
  output reg  [11:0] tok_emb_addr_o,
  input  wire [63:0] tok_emb_data_i,
  input  wire [15:0] tok_emb_scale_i,

  // K cache (fp16)
  output reg         k_we_o,
  output reg  [15:0] k_wdata_o,
  output reg  [1:0]  k_layer_o,
  output reg  [2:0]  k_head_o,
  output reg  [7:0]  k_pos_o,
  output reg  [3:0]  k_dim_o,
  input  wire [31:0] k_rdata_i,

  // V cache (fp16)
  output reg         v_we_o,
  output reg  [15:0] v_wdata_o,
  output reg  [1:0]  v_layer_o,
  output reg  [2:0]  v_head_o,
  output reg  [7:0]  v_pos_o,
  output reg  [3:0]  v_dim_o,
  input  wire [31:0] v_rdata_i,

  // Output
  output reg  [7:0]  token_o,
  output reg         token_valid_o,
  output reg         busy_o,
  output reg         done_o
);


  // FSM states
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

  // Shared activation RAM: 256 entries for head proj logits, 128 for hidden state
  reg [15:0] act_ram [0:255];

  reg [7:0]    cur_token;
  reg [7:0]    pos_r;
  reg [1:0]    layer_idx;
  reg          generating;

  // Embedding: writes to act_ram
  reg         emb_start;
  wire [5:0]  emb_w_sel;
  wire [15:0] emb_w_addr;
  wire [11:0] emb_tok_addr;
  wire        emb_res_we;
  wire [6:0]  emb_res_waddr;
  wire [15:0] emb_res_wdata;
  wire        emb_done;

  embedding u_emb (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .start_i    (emb_start),
    .token_id_i (cur_token),
    .position_i (pos_r),
    .w_scale_i  (w_scale_i),
    .w_sel_o    (emb_w_sel),
    .w_addr_o   (emb_w_addr),
    .w_data_i   (w_data_i),
    .tok_addr_o (emb_tok_addr),
    .tok_data_i (tok_emb_data_i),
    .tok_scale_i(tok_emb_scale_i),
    .res_we_o   (emb_res_we),
    .res_waddr_o(emb_res_waddr),
    .res_wdata_o(emb_res_wdata),
    .done_o     (emb_done),
    .busy_o     ()
  );

  // Transformer layer: reads/writes act_ram[0:127]
  reg          tl_start;
  wire [5:0]   tl_w_sel;
  wire [15:0]  tl_w_addr;
  wire [3:0]   tl_w8_sel;
  wire [15:0]  tl_w8_addr;
  wire [6:0]   tl_act_raddr;
  wire         tl_res_we;
  wire [6:0]   tl_res_waddr;
  wire [15:0]  tl_res_wdata;
  wire         tl_done;

  wire         tl_k_we;
  wire [15:0]  tl_k_wdata;
  wire [1:0]   tl_k_layer;
  wire [2:0]   tl_k_head;
  wire [7:0]   tl_k_pos;
  wire [3:0]   tl_k_dim;
  wire         tl_v_we;
  wire [15:0]  tl_v_wdata;
  wire [1:0]   tl_v_layer;
  wire [2:0]   tl_v_head;
  wire [7:0]   tl_v_pos;
  wire [3:0]   tl_v_dim;

  transformer_layer u_tl (
    .clk_i       (clk_i),
    .rst_i       (rst_i),
    .start_i     (tl_start),
    .layer_i     (layer_idx),
    .pos_i       (pos_r),
    .act_raddr_o (tl_act_raddr),
    .act_rdata_i (act_ram[tl_act_raddr]),
    .res_we_o    (tl_res_we),
    .res_waddr_o (tl_res_waddr),
    .res_wdata_o (tl_res_wdata),
    .w_sel_o     (tl_w_sel),
    .w_addr_o    (tl_w_addr),
    .w_data_i    (w_data_i),
    .w_scale_i   (w_scale_i),
    .w8_sel_o    (tl_w8_sel),
    .w8_addr_o   (tl_w8_addr),
    .w8_data_i   (w8_data_i),
    .w8_scale_i  (w8_scale_i),
    .k_we_o      (tl_k_we),
    .k_wdata_o   (tl_k_wdata),
    .k_layer_o   (tl_k_layer),
    .k_head_o    (tl_k_head),
    .k_pos_o     (tl_k_pos),
    .k_dim_o     (tl_k_dim),
    .k_rdata_i   (k_rdata_i),
    .v_we_o      (tl_v_we),
    .v_wdata_o   (tl_v_wdata),
    .v_layer_o   (tl_v_layer),
    .v_head_o    (tl_v_head),
    .v_pos_o     (tl_v_pos),
    .v_dim_o     (tl_v_dim),
    .v_rdata_i   (v_rdata_i),
    .done_o      (tl_done)
  );

  // KV boundary register: splits the FSM-state -> 32-bank BRAM fanout
  reg kv_active;
  always @(posedge clk_i) begin
    if (rst_i) begin
      k_we_o    <= 1'b0;
      k_wdata_o <= 16'd0;
      k_layer_o <= 2'd0;
      k_head_o  <= 3'd0;
      k_pos_o   <= 8'd0;
      k_dim_o   <= 4'd0;
      v_we_o    <= 1'b0;
      v_wdata_o <= 16'd0;
      v_layer_o <= 2'd0;
      v_head_o  <= 3'd0;
      v_pos_o   <= 8'd0;
      v_dim_o   <= 4'd0;
    end else begin
      k_we_o    <= kv_active ? tl_k_we : 1'b0;
      k_wdata_o <= tl_k_wdata;
      k_layer_o <= tl_k_layer;
      k_head_o  <= tl_k_head;
      k_pos_o   <= tl_k_pos;
      k_dim_o   <= tl_k_dim;
      v_we_o    <= kv_active ? tl_v_we : 1'b0;
      v_wdata_o <= tl_v_wdata;
      v_layer_o <= tl_v_layer;
      v_head_o  <= tl_v_head;
      v_pos_o   <= tl_v_pos;
      v_dim_o   <= tl_v_dim;
    end
  end

  // Final LayerNorm: reads/writes act_ram[0:127]
  reg         lnf_start;
  wire [5:0]  lnf_w_sel;
  wire [6:0]  lnf_w_addr;
  wire [6:0]  lnf_x_raddr;
  wire        lnf_y_we;
  wire [6:0]  lnf_y_waddr;
  wire [15:0] lnf_y_wdata;
  wire        lnf_done;

  layernorm u_ln_f (
    .clk_i       (clk_i),
    .rst_i       (rst_i),
    .start_i     (lnf_start),
    .x_raddr_o   (lnf_x_raddr),
    .x_rdata_i   (act_ram[lnf_x_raddr]),
    .y_we_o      (lnf_y_we),
    .y_waddr_o   (lnf_y_waddr),
    .y_wdata_o   (lnf_y_wdata),
    .w_sel_o     (lnf_w_sel),
    .w_addr_o    (lnf_w_addr),
    .w_data_i    (w_data_i),
    .gamma_sel_i (6'd34),
    .w_scale_i   (w_scale_i),
    .done_o      (lnf_done),
    .busy_o      ()
  );

  // Head projection: reads act_ram[0:127], writes act_ram[0:255]
  reg          head_start;
  wire [11:0]  head_w8_addr;
  wire [6:0]   head_act_raddr;
  wire         head_res_we;
  wire [7:0]   head_res_waddr;
  wire [15:0]  head_res_wdata;
  wire         head_done;

  matvec_fp16_w8 #(.IN_DIM(128), .OUT_DIM(256)) u_head_proj (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (head_start),
    .scale_i      (tok_emb_scale_i),
    .weight_addr_o(head_w8_addr),
    .weight_data_i(tok_emb_data_i),
    .act_raddr_o  (head_act_raddr),
    .act_rdata_i  (act_ram[head_act_raddr]),
    .res_we_o     (head_res_we),
    .res_waddr_o  (head_res_waddr),
    .res_wdata_o  (head_res_wdata),
    .done_o       (head_done)
  );

  // act_ram write mux: only one writer active at a time
  always @(posedge clk_i) begin
    if (emb_res_we)
      act_ram[emb_res_waddr] <= emb_res_wdata;
    else if (tl_res_we)
      act_ram[tl_res_waddr] <= tl_res_wdata;
    else if (lnf_y_we)
      act_ram[lnf_y_waddr] <= lnf_y_wdata;
    else if (head_res_we)
      act_ram[head_res_waddr] <= head_res_wdata;
  end

  // Sampler: reads act_ram[0:255]
  reg          samp_start;
  wire [7:0]   samp_logit_raddr;
  wire [7:0]   samp_token;
  wire         samp_done;

  sampler u_samp (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (samp_start),
    .logit_raddr_o(samp_logit_raddr),
    .logit_rdata_i(act_ram[samp_logit_raddr]),
    .token_o      (samp_token),
    .done_o       (samp_done)
  );

  // Combinational mux for the 8-bit weight store inputs.
  // Embedding only drives w_* during S_EMBED for pos_emb access; tok_emb has
  // its own dedicated bus
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
      default: begin
        w_sel_o  = 6'd0;
        w_addr_o = 16'd0;
      end
    endcase
  end

  // Combinational mux for the 64-bit packed weight store inputs.
  // Only active during transformer_layer execution
  always @(*) begin
    case (state)
      S_LAYER_START, S_LAYER_WAIT: begin
        w8_sel_o  = tl_w8_sel;
        w8_addr_o = tl_w8_addr;
      end
      default: begin
        w8_sel_o  = 4'd0;
        w8_addr_o = 16'd0;
      end
    endcase
  end

  // Mux the tok_emb bus address between embedding and head_proj
  always @(*) begin
    case (state)
      S_EMBED:     tok_emb_addr_o = emb_tok_addr;
      S_HEAD_PROJ: tok_emb_addr_o = head_w8_addr;
      default:     tok_emb_addr_o = 12'd0;
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