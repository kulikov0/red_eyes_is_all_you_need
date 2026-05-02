// Single transformer block: LN1, Attention, Residual, LN2, FF, Residual.
//
// W8A16: int8 weights in BRAM, fp16 activations throughout. Two weight buses:
// 8-bit shared bank for LN, 128-bit packed bank for the per-layer
// matvec_fp16_w16 instances inside attention and FF.
//
// w_sel_o tensor_sel mapping per layer L = {layer_r, 3'b000}:
//   LN1 gamma = L+2, LN1 beta = L+3
//   LN2 gamma = L+6, LN2 beta = L+7
//
// w16_sel_o encoding: {layer, type}
//   type 00: qkv,    01: proj    driven by attention
//   type 10: ff_up,  11: ff_down driven by transformer_layer

module transformer_layer (
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

  // 8-bit weight store interface, only used by LN
  output reg  [5:0]    w_sel_o,
  output reg  [15:0]   w_addr_o,
  input  wire [7:0]    w_data_i,
  input  wire [15:0]   w_scale_i,

  // Per-tensor weight buses, each backed by its own dedicated bank
  output wire [11:0]   qkv_addr_o,
  input  wire [127:0]  qkv_data_i,
  input  wire [15:0]   qkv_scale_i,
  output wire [9:0]    proj_addr_o,
  input  wire [127:0]  proj_data_i,
  input  wire [15:0]   proj_scale_i,
  output wire [11:0]   ff_up_addr_o,
  input  wire [127:0]  ff_up_data_i,
  input  wire [15:0]   ff_up_scale_i,
  output wire [11:0]   ff_down_addr_o,
  input  wire [127:0]  ff_down_data_i,
  input  wire [15:0]   ff_down_scale_i,

  // K cache (fp16)
  output wire          k_we_o,
  output wire [15:0]   k_wdata_o,
  output wire [1:0]    k_layer_o,
  output wire [2:0]    k_head_o,
  output wire [7:0]    k_pos_o,
  output wire [3:0]    k_dim_o,
  input  wire [31:0]   k_rdata_i,

  // V cache (fp16)
  output wire          v_we_o,
  output wire [15:0]   v_wdata_o,
  output wire [1:0]    v_layer_o,
  output wire [2:0]    v_head_o,
  output wire [7:0]    v_pos_o,
  output wire [3:0]    v_dim_o,
  input  wire [31:0]   v_rdata_i,

  output reg           done_o
);

  // FSM states
  localparam [3:0] S_IDLE      = 4'd0,
                   S_LOAD      = 4'd1,
                   S_SAVE_RES  = 4'd2,
                   S_LN_START  = 4'd3,
                   S_LN_WAIT   = 4'd4,
                   S_ATTN      = 4'd5,
                   S_RES_ADD   = 4'd6,
                   S_FF_UP     = 4'd7,
                   S_GELU      = 4'd8,
                   S_FF_DOWN   = 4'd9,
                   S_STORE_OUT = 4'd10,
                   S_DONE      = 4'd11;

  reg [3:0] state;

  // Latched inputs
  reg [1:0]    layer_r;
  reg [7:0]    pos_r;

  // Shared activation RAM: holds current vector
  reg [15:0] act_ram [0:511];

  // Residual RAM: preserved across LN+attn and LN+FF
  reg [15:0] res_ram [0:127];

  // LN which: 0=LN1, 1=LN2
  reg          ln_which;

  // GELU index
  reg [9:0]    gelu_idx;

  // Load/store index for parent RAM transfers
  reg [7:0]    io_idx;
  assign act_raddr_o = io_idx[6:0];

  // LayerNorm: reads/writes act_ram[0:127]
  reg          ln_start;
  reg  [5:0]   ln_gamma_sel;
  wire [5:0]   ln_w_sel;
  wire [6:0]   ln_w_addr;
  wire [6:0]   ln_x_raddr;
  wire         ln_y_we;
  wire [6:0]   ln_y_waddr;
  wire [15:0]  ln_y_wdata;
  wire         ln_done;

  layernorm u_ln (
    .clk_i       (clk_i),
    .rst_i       (rst_i),
    .start_i     (ln_start),
    .x_raddr_o   (ln_x_raddr),
    .x_rdata_i   (act_ram[ln_x_raddr]),
    .y_we_o      (ln_y_we),
    .y_waddr_o   (ln_y_waddr),
    .y_wdata_o   (ln_y_wdata),
    .w_sel_o     (ln_w_sel),
    .w_addr_o    (ln_w_addr),
    .w_data_i    (w_data_i),
    .gamma_sel_i (ln_gamma_sel),
    .w_scale_i   (w_scale_i),
    .done_o      (ln_done),
    .busy_o      ()
  );

  // Attention: reads/writes act_ram[0:127]
  reg          attn_start;
  wire [6:0]   attn_act_raddr;
  wire         attn_res_we;
  wire [6:0]   attn_res_waddr;
  wire [15:0]  attn_res_wdata;
  wire         attn_done;

  wire         attn_k_we;
  wire [15:0]  attn_k_wdata;
  wire [1:0]   attn_k_layer;
  wire [2:0]   attn_k_head;
  wire [7:0]   attn_k_pos;
  wire [3:0]   attn_k_dim;
  wire         attn_v_we;
  wire [15:0]  attn_v_wdata;
  wire [1:0]   attn_v_layer;
  wire [2:0]   attn_v_head;
  wire [7:0]   attn_v_pos;
  wire [3:0]   attn_v_dim;

  attention u_attn (
    .clk_i       (clk_i),
    .rst_i       (rst_i),
    .start_i     (attn_start),
    .layer_i     (layer_r),
    .pos_i       (pos_r),
    .act_raddr_o (attn_act_raddr),
    .act_rdata_i (act_ram[attn_act_raddr]),
    .res_we_o    (attn_res_we),
    .res_waddr_o (attn_res_waddr),
    .res_wdata_o (attn_res_wdata),
    .qkv_addr_o  (qkv_addr_o),
    .qkv_data_i  (qkv_data_i),
    .qkv_scale_i (qkv_scale_i),
    .proj_addr_o (proj_addr_o),
    .proj_data_i (proj_data_i),
    .proj_scale_i(proj_scale_i),
    .k_we_o      (attn_k_we),
    .k_wdata_o   (attn_k_wdata),
    .k_layer_o   (attn_k_layer),
    .k_head_o    (attn_k_head),
    .k_pos_o     (attn_k_pos),
    .k_dim_o     (attn_k_dim),
    .k_rdata_i   (k_rdata_i),
    .v_we_o      (attn_v_we),
    .v_wdata_o   (attn_v_wdata),
    .v_layer_o   (attn_v_layer),
    .v_head_o    (attn_v_head),
    .v_pos_o     (attn_v_pos),
    .v_dim_o     (attn_v_dim),
    .v_rdata_i   (v_rdata_i),
    .done_o      (attn_done)
  );


  // KV cache pass-through from attention
  assign k_we_o    = attn_k_we;
  assign k_wdata_o = attn_k_wdata;
  assign k_layer_o = attn_k_layer;
  assign k_head_o  = attn_k_head;
  assign k_pos_o   = attn_k_pos;
  assign k_dim_o   = attn_k_dim;
  assign v_we_o    = attn_v_we;
  assign v_wdata_o = attn_v_wdata;
  assign v_layer_o = attn_v_layer;
  assign v_head_o  = attn_v_head;
  assign v_pos_o   = attn_v_pos;
  assign v_dim_o   = attn_v_dim;

  // FF_up matvec: 128 -> 512, reads act_ram[0:127], writes act_ram[0:511]
  reg          ff_up_start;
  wire [6:0]   ff_up_act_raddr;
  wire         ff_up_res_we;
  wire [8:0]   ff_up_res_waddr;
  wire [15:0]  ff_up_res_wdata;
  wire         ff_up_done;

  matvec_fp16_w16 #(.IN_DIM(128), .OUT_DIM(512)) u_ff_up (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ff_up_start),
    .scale_i      (ff_up_scale_i),
    .weight_addr_o(ff_up_addr_o),
    .weight_data_i(ff_up_data_i),
    .act_raddr_o  (ff_up_act_raddr),
    .act_rdata_i  (act_ram[ff_up_act_raddr]),
    .res_we_o     (ff_up_res_we),
    .res_waddr_o  (ff_up_res_waddr),
    .res_wdata_o  (ff_up_res_wdata),
    .done_o       (ff_up_done)
  );


  // FF_down matvec: 512 -> 128, reads act_ram[0:511], writes act_ram[0:127]
  reg          ff_down_start;
  wire [8:0]   ff_down_act_raddr;
  wire         ff_down_res_we;
  wire [6:0]   ff_down_res_waddr;
  wire [15:0]  ff_down_res_wdata;
  wire         ff_down_done;

  matvec_fp16_w16 #(.IN_DIM(512), .OUT_DIM(128)) u_ff_down (
    .clk_i        (clk_i),
    .rst_i        (rst_i),
    .start_i      (ff_down_start),
    .scale_i      (ff_down_scale_i),
    .weight_addr_o(ff_down_addr_o),
    .weight_data_i(ff_down_data_i),
    .act_raddr_o  (ff_down_act_raddr),
    .act_rdata_i  (act_ram[ff_down_act_raddr]),
    .res_we_o     (ff_down_res_we),
    .res_waddr_o  (ff_down_res_waddr),
    .res_wdata_o  (ff_down_res_wdata),
    .done_o       (ff_down_done)
  );


  // GELU: fp16 PWL, 3-cycle pipeline
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

  // Sequential residual add: act_ram[i] + res_ram[i] -> act_ram[i]
  // Track write address through pipe to match fp16_add latency
  reg [7:0] res_idx;
  wire res_add_feed = (state == S_RES_ADD) && (res_idx < 8'd128);
  wire res_add_valid_out;
  wire [15:0] res_add_out;
  fp16_add u_res_add (
    .clk_i  (clk_i),
    .valid_i(res_add_feed),
    .a_i    (act_ram[res_idx[6:0]]),
    .b_i    (res_ram[res_idx[6:0]]),
    .valid_o(res_add_valid_out),
    .sum_o  (res_add_out)
  );

  reg [6:0] res_wr_pipe [0:3];
  integer i;
  always @(posedge clk_i) begin
    res_wr_pipe[0] <= res_idx[6:0];
    for (i = 1; i < 4; i = i + 1) res_wr_pipe[i] <= res_wr_pipe[i-1];
  end

  // 8-bit weight store mux for LN gamma/beta
  always @(*) begin
    case (state)
      S_LN_START, S_LN_WAIT: begin
        w_sel_o  = ln_w_sel;
        w_addr_o = {9'd0, ln_w_addr};
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
      res_we_o      <= 1'b0;
      ln_start      <= 1'b0;
      attn_start    <= 1'b0;
      ff_up_start   <= 1'b0;
      ff_down_start <= 1'b0;
      gelu_valid_in <= 1'b0;
    end else begin
      done_o        <= 1'b0;
      res_we_o      <= 1'b0;
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
            ln_which <= 1'b0;
            io_idx   <= 8'd0;
            state    <= S_LOAD;
          end
        end

        // Load 128 elements from parent shared RAM into act_ram
        S_LOAD: begin
          act_ram[io_idx[6:0]] <= act_rdata_i;
          io_idx <= io_idx + 8'd1;
          if (io_idx == 8'd127) begin
            state  <= S_SAVE_RES;
            io_idx <= 8'd0;
          end
        end

        // Copy act_ram[0:127] to res_ram for residual add later
        S_SAVE_RES: begin
          res_ram[io_idx[6:0]] <= act_ram[io_idx[6:0]];
          io_idx <= io_idx + 8'd1;
          if (io_idx == 8'd127) begin
            state <= S_LN_START;
          end
        end

        S_LN_START: begin
          ln_start <= 1'b1;
          if (ln_which == 1'b0) begin
            ln_gamma_sel <= {layer_r, 3'b000} + 6'd2;
          end else begin
            ln_gamma_sel <= {layer_r, 3'b000} + 6'd6;
          end
          state <= S_LN_WAIT;
        end

        S_LN_WAIT: begin
          if (ln_y_we)
            act_ram[ln_y_waddr] <= ln_y_wdata;
          if (ln_done) begin
            if (ln_which == 1'b0) begin
              state      <= S_ATTN;
              attn_start <= 1'b1;
            end else begin
              state       <= S_FF_UP;
              ff_up_start <= 1'b1;
            end
          end
        end

        S_ATTN: begin
          if (attn_res_we)
            act_ram[attn_res_waddr] <= attn_res_wdata;
          if (attn_done) begin
            state   <= S_RES_ADD;
            res_idx <= 8'd0;
          end
        end

        // Sequential residual add with 4-cycle pipelined fp16_add
        S_RES_ADD: begin
          if (res_add_valid_out) begin
            act_ram[res_wr_pipe[3]] <= res_add_out;
          end
          res_idx <= res_idx + 8'd1;
          if (res_idx == 8'd131) begin
            if (ln_which == 1'b0) begin
              // After res1: save new residual, do LN2
              ln_which <= 1'b1;
              io_idx   <= 8'd0;
              state    <= S_SAVE_RES;
            end else begin
              // After res2: store output to parent RAM
              io_idx <= 8'd0;
              state  <= S_STORE_OUT;
            end
          end
        end

        S_FF_UP: begin
          if (ff_up_res_we)
            act_ram[ff_up_res_waddr] <= ff_up_res_wdata;
          if (ff_up_done) begin
            state    <= S_GELU;
            gelu_idx <= 10'd0;
          end
        end

        // GELU: 15-cycle pipeline, reads/writes act_ram in-place
        S_GELU: begin
          if (gelu_idx <= 10'd511) begin
            gelu_in       <= act_ram[gelu_idx[8:0]];
            gelu_valid_in <= 1'b1;
          end

          if (gelu_idx >= 10'd15) begin
            act_ram[gelu_idx[8:0] - 9'd15] <= gelu_out;
          end

          gelu_idx <= gelu_idx + 10'd1;

          if (gelu_idx == 10'd526) begin
            state         <= S_FF_DOWN;
            ff_down_start <= 1'b1;
          end
        end

        S_FF_DOWN: begin
          if (ff_down_res_we)
            act_ram[ff_down_res_waddr] <= ff_down_res_wdata;
          if (ff_down_done) begin
            state   <= S_RES_ADD;
            res_idx <= 8'd0;
          end
        end

        // Copy act_ram[0:127] to parent shared RAM
        S_STORE_OUT: begin
          res_we_o    <= 1'b1;
          res_waddr_o <= io_idx[6:0];
          res_wdata_o <= act_ram[io_idx[6:0]];
          io_idx <= io_idx + 8'd1;
          if (io_idx == 8'd127) begin
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