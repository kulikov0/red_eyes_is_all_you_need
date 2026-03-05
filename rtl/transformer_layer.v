// Single transformer block: LN1 -> Attention -> Residual -> LN2 -> FF -> Residual
//
// W8A16: int8 weights in BRAM, fp16 activations throughout
// Submodules: layernorm (flat fp16 bus), attention (fp16), 2x matvec_fp16, gelu (fp16 PWL)
// Weight store and KV cache are external, muxed here
//
// Weight store tensor_sel mapping per layer L = {layer_r, 3'b000}:
//   LN1 gamma = L+2, LN1 beta = L+3
//   QKV = L+4, Proj = L+5 (handled inside attention)
//   LN2 gamma = L+6, LN2 beta = L+7
//   FF_up = L+8, FF_down = L+9
//
// Latency: 2*656 + attention + 65538 + 515 + 65538 + 7 cycles
//   At P=255: ~276,588 cycles

module transformer_layer (
  input  wire          clk_i,
  input  wire          rst_i,
  input  wire          start_i,
  input  wire [1:0]    layer_i,
  input  wire [7:0]    pos_i,
  input  wire [2047:0] x_i,

  // Weight store interface
  output reg  [5:0]    w_sel_o,
  output reg  [15:0]   w_addr_o,
  input  wire [7:0]    w_data_i,

  // K cache (fp16)
  output wire          k_we_o,
  output wire [15:0]   k_wdata_o,
  input  wire [15:0]   k_rdata_i,

  // V cache (fp16)
  output wire          v_we_o,
  output wire [15:0]   v_wdata_o,
  input  wire [15:0]   v_rdata_i,

  // KV cache address (shared K/V)
  output wire [1:0]    kv_layer_o,
  output wire [2:0]    kv_head_o,
  output wire [7:0]    kv_pos_o,
  output wire [3:0]    kv_dim_o,

  // Output
  output reg  [2047:0] out_vec_o,
  output reg           done_o
);

  `include "weight_scales.vh"

  // FSM states
  localparam [3:0] S_IDLE     = 4'd0,
                   S_LN_START = 4'd1,
                   S_LN_WAIT  = 4'd2,
                   S_ATTN     = 4'd3,
                   S_RES1     = 4'd4,
                   S_FF_UP    = 4'd5,
                   S_GELU     = 4'd6,
                   S_FF_DOWN  = 4'd7,
                   S_RES2     = 4'd8,
                   S_DONE     = 4'd9;

  reg [3:0] state;

  // Latched inputs
  reg [1:0]    layer_r;
  reg [7:0]    pos_r;

  // Residual register (preserved across LN+attn and LN+FF)
  reg [2047:0] x_reg;

  // LN/attn output buffer (128 x fp16)
  reg [2047:0] sub_out;

  // FF intermediate buffer (512 x fp16)
  reg [8191:0] ff_buf;

  // LN which: 0=LN1, 1=LN2
  reg          ln_which;

  // GELU index
  reg [9:0]    gelu_idx;

  // Per-layer scale muxes
  reg [15:0] ln_gamma_scale;
  reg [15:0] ln_beta_scale;
  reg [15:0] ff_up_scale;
  reg [15:0] ff_down_scale;

  always @(*) begin
    case (layer_r)
      2'd0: begin
        ff_up_scale   = SCALE_BLOCK0_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK0_FF_DOWN_WEIGHT;
      end
      2'd1: begin
        ff_up_scale   = SCALE_BLOCK1_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK1_FF_DOWN_WEIGHT;
      end
      2'd2: begin
        ff_up_scale   = SCALE_BLOCK2_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK2_FF_DOWN_WEIGHT;
      end
      2'd3: begin
        ff_up_scale   = SCALE_BLOCK3_FF_UP_WEIGHT;
        ff_down_scale = SCALE_BLOCK3_FF_DOWN_WEIGHT;
      end
    endcase
  end

  // LN scale mux: depends on layer_r and ln_which
  always @(*) begin
    case ({layer_r, ln_which})
      3'b000: begin
        ln_gamma_scale = SCALE_BLOCK0_LN1_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK0_LN1_BIAS;
      end
      3'b001: begin
        ln_gamma_scale = SCALE_BLOCK0_LN2_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK0_LN2_BIAS;
      end
      3'b010: begin
        ln_gamma_scale = SCALE_BLOCK1_LN1_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK1_LN1_BIAS;
      end
      3'b011: begin
        ln_gamma_scale = SCALE_BLOCK1_LN2_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK1_LN2_BIAS;
      end
      3'b100: begin
        ln_gamma_scale = SCALE_BLOCK2_LN1_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK2_LN1_BIAS;
      end
      3'b101: begin
        ln_gamma_scale = SCALE_BLOCK2_LN2_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK2_LN2_BIAS;
      end
      3'b110: begin
        ln_gamma_scale = SCALE_BLOCK3_LN1_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK3_LN1_BIAS;
      end
      3'b111: begin
        ln_gamma_scale = SCALE_BLOCK3_LN2_WEIGHT;
        ln_beta_scale  = SCALE_BLOCK3_LN2_BIAS;
      end
    endcase
  end

  // LayerNorm interface (flat fp16 bus)
  reg          ln_start;
  reg  [5:0]   ln_gamma_sel;
  wire [5:0]   ln_w_sel;
  wire [6:0]   ln_w_addr;
  wire [2047:0] ln_y;
  wire         ln_done;

  layernorm u_ln (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ln_start),
    .x_i          (x_reg),
    .w_sel_o      (ln_w_sel),
    .w_addr_o     (ln_w_addr),
    .w_data_i     (w_data_i),
    .gamma_sel_i  (ln_gamma_sel),
    .gamma_scale_i(ln_gamma_scale),
    .beta_scale_i (ln_beta_scale),
    .y_o          (ln_y),
    .done_o       (ln_done),
    .busy_o       ()
  );

  // Attention interface
  reg          attn_start;
  wire [5:0]   attn_w_sel;
  wire [15:0]  attn_w_addr;
  wire [2047:0] attn_out;
  wire         attn_done;

  // KV wires from attention
  wire         attn_k_we;
  wire [15:0]  attn_k_wdata;
  wire         attn_v_we;
  wire [15:0]  attn_v_wdata;
  wire [1:0]   attn_kv_layer;
  wire [2:0]   attn_kv_head;
  wire [7:0]   attn_kv_pos;
  wire [3:0]   attn_kv_dim;

  attention u_attn (
    .clk_i     (clk_i),
    .rst_i     (rst_i),
    .start_i   (attn_start),
    .layer_i   (layer_r),
    .pos_i     (pos_r),
    .x_i       (sub_out),
    .w_sel_o   (attn_w_sel),
    .w_addr_o  (attn_w_addr),
    .w_data_i  (w_data_i),
    .k_we_o    (attn_k_we),
    .k_wdata_o (attn_k_wdata),
    .k_rdata_i (k_rdata_i),
    .v_we_o    (attn_v_we),
    .v_wdata_o (attn_v_wdata),
    .v_rdata_i (v_rdata_i),
    .kv_layer_o(attn_kv_layer),
    .kv_head_o (attn_kv_head),
    .kv_pos_o  (attn_kv_pos),
    .kv_dim_o  (attn_kv_dim),
    .out_vec_o (attn_out),
    .done_o    (attn_done)
  );

  // KV cache pass-through from attention
  assign k_we_o     = attn_k_we;
  assign k_wdata_o  = attn_k_wdata;
  assign v_we_o     = attn_v_we;
  assign v_wdata_o  = attn_v_wdata;
  assign kv_layer_o = attn_kv_layer;
  assign kv_head_o  = attn_kv_head;
  assign kv_pos_o   = attn_kv_pos;
  assign kv_dim_o   = attn_kv_dim;

  // FF_up matvec: 128 -> 512, fp16
  reg          ff_up_start;
  wire [15:0]  ff_up_addr;
  wire [8191:0] ff_up_out;
  wire         ff_up_done;

  matvec_fp16 #(.IN_DIM(128), .OUT_DIM(512)) u_ff_up (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ff_up_start),
    .in_vec_i     (sub_out),
    .scale_i      (ff_up_scale),
    .weight_addr_o(ff_up_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (ff_up_out),
    .done_o       (ff_up_done)
  );

  // FF_down matvec: 512 -> 128, fp16
  reg          ff_down_start;
  wire [15:0]  ff_down_addr;
  wire [2047:0] ff_down_out;
  wire         ff_down_done;

  matvec_fp16 #(.IN_DIM(512), .OUT_DIM(128)) u_ff_down (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ff_down_start),
    .in_vec_i     (ff_buf),
    .scale_i      (ff_down_scale),
    .weight_addr_o(ff_down_addr),
    .weight_data_i(w_data_i),
    .out_vec_o    (ff_down_out),
    .done_o       (ff_down_done)
  );

  // GELU (fp16 PWL, 2-cycle pipeline)
  reg         gelu_valid_in;
  reg  [15:0] gelu_in;
  wire [15:0] gelu_out;

  gelu u_gelu (
    .clk_i  (clk_i),
    .valid_i(gelu_valid_in),
    .x_i    (gelu_in),
    .valid_o(),
    .y_o    (gelu_out)
  );

  // Residual 1: fp16 add (combinational)
  wire [2047:0] res1_sum;
  genvar gi;
  generate
    for (gi = 0; gi < 128; gi = gi + 1) begin : gen_res1
      wire [15:0] res1_fp16;
      fp16_add_comb u_res1 (
        .a_i  (sub_out[gi*16 +: 16]),
        .b_i  (x_reg[gi*16 +: 16]),
        .sum_o(res1_fp16)
      );
      assign res1_sum[gi*16 +: 16] = res1_fp16;
    end
  endgenerate

  // Residual 2: fp16 add (combinational)
  wire [2047:0] res2_sum;
  generate
    for (gi = 0; gi < 128; gi = gi + 1) begin : gen_res2
      wire [15:0] res2_fp16;
      fp16_add_comb u_res2 (
        .a_i  (ff_down_out[gi*16 +: 16]),
        .b_i  (x_reg[gi*16 +: 16]),
        .sum_o(res2_fp16)
      );
      assign res2_sum[gi*16 +: 16] = res2_fp16;
    end
  endgenerate

  // Weight store mux (active sel depends on FSM state)
  always @(*) begin
    case (state)
      S_LN_START, S_LN_WAIT: begin
        w_sel_o  = ln_w_sel;
        w_addr_o = {9'd0, ln_w_addr};
      end
      S_ATTN: begin
        w_sel_o  = attn_w_sel;
        w_addr_o = attn_w_addr;
      end
      S_FF_UP: begin
        w_sel_o  = {layer_r, 3'b000} + 6'd8;
        w_addr_o = ff_up_addr;
      end
      S_FF_DOWN: begin
        w_sel_o  = {layer_r, 3'b000} + 6'd9;
        w_addr_o = ff_down_addr;
      end
      default: begin
        w_sel_o  = 6'd0;
        w_addr_o = 16'd0;
      end
    endcase
  end

  // Main FSM
  always @(posedge clk_i) begin
    if (rst_i) begin
      state         <= S_IDLE;
      done_o        <= 1'b0;
      ln_start      <= 1'b0;
      attn_start    <= 1'b0;
      ff_up_start   <= 1'b0;
      ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;
    end else begin
      done_o        <= 1'b0;
      ln_start      <= 1'b0;
      attn_start    <= 1'b0;
      ff_up_start   <= 1'b0;
      ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            layer_r  <= layer_i;
            pos_r    <= pos_i;
            x_reg    <= x_i;
            ln_which <= 1'b0;
            state    <= S_LN_START;
          end
        end

        // Assert ln_start for 1 cycle, set gamma_sel based on ln_which
        S_LN_START: begin
          ln_start <= 1'b1;
          if (ln_which == 1'b0) begin
            ln_gamma_sel <= {layer_r, 3'b000} + 6'd2;
          end else begin
            ln_gamma_sel <= {layer_r, 3'b000} + 6'd6;
          end
          state <= S_LN_WAIT;
        end

        // Wait for layernorm to complete, capture flat bus output
        S_LN_WAIT: begin
          if (ln_done) begin
            sub_out <= ln_y;
            if (ln_which == 1'b0) begin
              state      <= S_ATTN;
              attn_start <= 1'b1;
            end else begin
              state       <= S_FF_UP;
              ff_up_start <= 1'b1;
            end
          end
        end

        // Wait for attention to complete
        S_ATTN: begin
          if (attn_done) begin
            sub_out <= attn_out;
            state   <= S_RES1;
          end
        end

        // Residual add: x_reg = fp16_add(sub_out, x_reg)
        S_RES1: begin
          x_reg    <= res1_sum;
          ln_which <= 1'b1;
          state    <= S_LN_START;
        end

        // Wait for FF_up to complete
        S_FF_UP: begin
          if (ff_up_done) begin
            ff_buf   <= ff_up_out;
            state    <= S_GELU;
            gelu_idx <= 10'd0;
          end
        end

        // GELU: 3-cycle pipeline (FSM reg + 2 GELU stages)
        // Cycle 0: gelu_in <= ff_buf[0]
        // Cycle 1: gelu_in <= ff_buf[1], GELU stage 1 latches ff_buf[0]
        // Cycle 2: gelu_in <= ff_buf[2], GELU stage 2 latches, y_o updating
        // Cycle 3: gelu_in <= ff_buf[3], gelu_out = GELU(ff_buf[0]) -> capture
        S_GELU: begin
          if (gelu_idx <= 10'd511) begin
            gelu_in       <= ff_buf[gelu_idx*16 +: 16];
            gelu_valid_in <= 1'b1;
          end

          if (gelu_idx >= 10'd3) begin
            ff_buf[(gelu_idx - 10'd3)*16 +: 16] <= gelu_out;
          end

          gelu_idx <= gelu_idx + 10'd1;

          if (gelu_idx == 10'd514) begin
            state         <= S_FF_DOWN;
            ff_down_start <= 1'b1;
          end
        end

        // Wait for FF_down to complete
        S_FF_DOWN: begin
          if (ff_down_done) begin
            out_vec_o <= res2_sum;
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