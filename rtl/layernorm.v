// LayerNorm: y_i = (x_i - mean) / sqrt(var) * gamma + beta
// Ref: https://www.mdpi.com/2072-666X/17/1/84
//
// Reads gamma/beta int8 from weight_store, dequants to fp16
//
// MEAN feeds x[idx] into fp16_reduce_k4
// VAR feeds (x[idx]-mean)^2 via pipelined sub+mul into fp16_reduce_k4
// NORM pipelined: sub, mul_rsqrt, mul_gamma, add_beta
//
// FSM: IDLE -> MEAN_ACC -> MEAN_DIV -> VAR_ACC -> VAR_DIV ->
//      INV_SQRT -> LOAD_GAMMA -> LOAD_BETA -> NORM_FEED -> NORM_DRAIN

module layernorm #(
  parameter DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,

  output wire [$clog2(DIM)-1:0] x_raddr_o,
  input  wire [15:0]            x_rdata_i,

  output reg                     y_we_o,
  output reg  [$clog2(DIM)-1:0]  y_waddr_o,
  output reg  [15:0]             y_wdata_o,

  // Weight store interface for gamma/beta
  output reg  [5:0]  w_sel_o,
  output reg  [6:0]  w_addr_o,
  input  wire [7:0]  w_data_i,

  input  wire [5:0]  gamma_sel_i,
  input  wire [15:0] w_scale_i,

  output reg         done_o,
  output reg         busy_o
);

  localparam S_IDLE       = 4'd0;
  localparam S_MEAN_ACC   = 4'd1;
  localparam S_MEAN_DIV   = 4'd2;
  localparam S_VAR_ACC    = 4'd3;
  localparam S_VAR_DIV    = 4'd4;
  localparam S_INV_SQRT   = 4'd5;
  localparam S_LOAD_GAMMA = 4'd6;
  localparam S_LOAD_BETA  = 4'd7;
  localparam S_NORM_FEED  = 4'd8;
  localparam S_NORM_DRAIN = 4'd9;
  localparam S_LN_DONE    = 4'd10;

  reg [3:0] state;
  reg [7:0] idx;
  wire [7:0] prev = idx - 8'd1;

  // FP16 accumulators
  reg [15:0] sum_acc;
  reg [15:0] neg_mean;
  reg [15:0] var_acc;
  reg [15:0] inv_std;

  // Gamma/beta dequanted to fp16
  reg [15:0] gamma_buf [0:DIM-1];
  reg [15:0] beta_buf  [0:DIM-1];

  assign x_raddr_o = idx[$clog2(DIM)-1:0];
  wire [15:0] x_elem = x_rdata_i;

  // fp16(1/128) = 0x2000: sign=0, exp=8, frac=0 -> 2^(8-15) = 2^-7 = 1/128
  localparam [15:0] INV_N = 16'h2000;

  // State-edge detector: clear pulses on first cycle of MEAN/VAR_ACC,
  // valid_i fires from the next cycle. act_ram has combinational read so
  // x_rdata equals x[idx] at the same cycle, no read latency
  reg [3:0] state_d;
  always @(posedge clk_i) state_d <= state;

  // MEAN: K=4 reduce of x[0..DIM-1]
  wire mean_clear   = (state == S_MEAN_ACC) && (state_d != S_MEAN_ACC);
  wire mean_valid_i = (state == S_MEAN_ACC) && (state_d == S_MEAN_ACC) && (idx <= DIM[7:0] - 8'd1);
  wire mean_flush   = mean_valid_i && (idx == DIM[7:0] - 8'd1);

  wire        mean_done;
  wire [15:0] mean_sum;
  fp16_reduce_k4 u_mean_red (
    .clk_i  (clk_i),
    .rst_i  (rst_i),
    .clear_i(mean_clear),
    .valid_i(mean_valid_i),
    .data_i (x_elem),
    .flush_i(mean_flush),
    .done_o (mean_done),
    .sum_o  (mean_sum)
  );

  // MEAN_DIV: sum * (1/128)
  wire [15:0] mean_div_out;
  fp16_mul_comb u_mean_div (.a_i(sum_acc), .b_i(INV_N), .prod_o(mean_div_out));

  // Variance: diff = x - mean. Gated by state_d so first cycle is the clear
  wire        var_issue = (state == S_VAR_ACC) && (state_d == S_VAR_ACC) && (idx <= DIM[7:0] - 8'd1);
  wire        var_sub_valid_out;
  wire [15:0] var_diff;
  fp16_add u_var_sub (
    .clk_i(clk_i),
    .valid_i(var_issue),
    .a_i(x_elem),
    .b_i(neg_mean),
    .valid_o(var_sub_valid_out),
    .sum_o(var_diff)
  );

  // sq = diff * diff
  wire        var_sq_valid;
  wire [15:0] var_sq;
  fp16_mul u_var_sq (
    .clk_i(clk_i),
    .valid_i(var_sub_valid_out),
    .a_i(var_diff),
    .b_i(var_diff),
    .valid_o(var_sq_valid),
    .prod_o(var_sq)
  );

  // VAR: K=4 reduce of squared diffs
  wire var_clear = (state == S_VAR_ACC) && (state_d != S_VAR_ACC);

  reg [7:0] var_red_in_cnt;
  wire var_flush = var_sq_valid && (var_red_in_cnt == DIM[7:0] - 8'd1);

  wire        var_done;
  wire [15:0] var_sum;
  fp16_reduce_k4 u_var_red (
    .clk_i  (clk_i),
    .rst_i  (rst_i),
    .clear_i(var_clear),
    .valid_i(var_sq_valid),
    .data_i (var_sq),
    .flush_i(var_flush),
    .done_o (var_done),
    .sum_o  (var_sum)
  );

  // VAR_DIV: var_acc * (1/128)
  wire [15:0] var_div_out;
  fp16_mul_comb u_var_div (.a_i(var_acc), .b_i(INV_N), .prod_o(var_div_out));

  // fp16_rsqrt interface
  reg         rsqrt_valid;
  wire        rsqrt_done;
  wire [15:0] rsqrt_result;

  fp16_rsqrt u_rsqrt (
    .clk_i   (clk_i),
    .valid_i (rsqrt_valid),
    .val_i   (var_div_out),
    .valid_o (rsqrt_done),
    .result_o(rsqrt_result)
  );

  // Boundary register on w_data_i breaks the long route from weight_store
  // BRAM through the tensor_sel mux into our DSP. Adds one cycle of latency
  reg [7:0] w_data_r;
  always @(posedge clk_i) w_data_r <= w_data_i;

  // LOAD_GAMMA/BETA: dequant int8 -> fp16
  wire [15:0] dequant_fp16;
  fp16_from_int8 u_dequant (.val_i(w_data_r), .fp16_o(dequant_fp16));

  wire deq_valid_in = ((state == S_LOAD_GAMMA) || (state == S_LOAD_BETA)) &&
                      (idx >= 8'd3) && (idx <= DIM[7:0] + 8'd2);

  wire        deq_valid_out;
  wire [15:0] dequant_scaled;
  fp16_mul u_deq_mul (
    .clk_i(clk_i),
    .valid_i(deq_valid_in),
    .a_i(dequant_fp16),
    .b_i(w_scale_i),
    .valid_o(deq_valid_out),
    .prod_o(dequant_scaled)
  );

  // Track gamma/beta write address through dequant pipeline. Source is idx-3
  // to match weight_store + boundary register delay through w_data_r
  wire [7:0] prev3 = idx - 8'd3;
  reg [$clog2(DIM)-1:0] deq_addr_r1, deq_addr_r2;
  always @(posedge clk_i) begin
    deq_addr_r1 <= prev3[$clog2(DIM)-1:0];
    deq_addr_r2 <= deq_addr_r1;
  end

  wire norm_feed = (state == S_NORM_FEED);

  // diff = x - mean
  wire        norm_sub_v_out;
  wire [15:0] norm_diff;
  fp16_add u_norm_sub (
    .clk_i(clk_i),
    .valid_i(norm_feed),
    .a_i(x_elem),
    .b_i(neg_mean),
    .valid_o(norm_sub_v_out),
    .sum_o(norm_diff)
  );

  // scaled = diff * inv_std
  wire        norm_mul1_v_out;
  wire [15:0] norm_scaled;
  fp16_mul u_norm_mul1 (
    .clk_i(clk_i),
    .valid_i(norm_sub_v_out),
    .a_i(norm_diff),
    .b_i(inv_std),
    .valid_o(norm_mul1_v_out),
    .prod_o(norm_scaled)
  );

  // Track idx through the NORM pipeline for gamma/beta/writeback addressing
  reg [$clog2(DIM)-1:0] idx_pipe [0:9];
  integer i;
  always @(posedge clk_i) begin
    idx_pipe[0] <= idx[$clog2(DIM)-1:0];
    for (i = 1; i < 10; i = i + 1) idx_pipe[i] <= idx_pipe[i-1];
  end

  // gamma_applied = scaled * gamma
  wire        norm_mul2_v_out;
  wire [15:0] norm_gamma;
  fp16_mul u_norm_mul2 (
    .clk_i(clk_i),
    .valid_i(norm_mul1_v_out),
    .a_i(norm_scaled),
    .b_i(gamma_buf[idx_pipe[4]]),
    .valid_o(norm_mul2_v_out),
    .prod_o(norm_gamma)
  );

  // out = gamma_applied + beta
  wire        norm_add_v_out;
  wire [15:0] norm_out;
  fp16_add u_norm_add (
    .clk_i(clk_i),
    .valid_i(norm_mul2_v_out),
    .a_i(norm_gamma),
    .b_i(beta_buf[idx_pipe[6]]),
    .valid_o(norm_add_v_out),
    .sum_o(norm_out)
  );

  // Drain counter: used by S_NORM_DRAIN
  reg [3:0] drain_cnt;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state        <= S_IDLE;
      idx          <= 8'd0;
      sum_acc      <= 16'd0;
      var_acc      <= 16'd0;
      neg_mean     <= 16'd0;
      inv_std      <= 16'd0;
      rsqrt_valid  <= 1'b0;
      done_o       <= 1'b0;
      busy_o       <= 1'b0;
      y_we_o       <= 1'b0;
      w_sel_o      <= 6'd0;
      w_addr_o     <= 7'd0;
      drain_cnt    <= 4'd0;
      var_red_in_cnt <= 8'd0;

    end else begin
      done_o      <= 1'b0;
      rsqrt_valid <= 1'b0;
      y_we_o      <= 1'b0;

      if (var_clear)         var_red_in_cnt <= 8'd0;
      else if (var_sq_valid) var_red_in_cnt <= var_red_in_cnt + 8'd1;

      if (norm_add_v_out) begin
        y_we_o    <= 1'b1;
        y_waddr_o <= idx_pipe[9];
        y_wdata_o <= norm_out;
      end

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state   <= S_MEAN_ACC;
            idx     <= 8'd0;
            busy_o  <= 1'b1;
          end
        end

        S_MEAN_ACC: begin
          if (mean_valid_i) idx <= idx + 8'd1;
          if (mean_done) begin
            sum_acc <= mean_sum;
            state   <= S_MEAN_DIV;
          end
        end

        S_MEAN_DIV: begin
          neg_mean <= {~mean_div_out[15], mean_div_out[14:0]};
          idx      <= 8'd0;
          state    <= S_VAR_ACC;
        end

        S_VAR_ACC: begin
          if (var_issue) idx <= idx + 8'd1;
          if (var_done) begin
            var_acc <= var_sum;
            state   <= S_VAR_DIV;
          end
        end

        S_VAR_DIV: begin
          rsqrt_valid <= 1'b1;
          idx         <= 8'd0;
          state       <= S_INV_SQRT;
        end

        S_INV_SQRT: begin
          if (rsqrt_done) begin
            inv_std  <= rsqrt_result;
            state    <= S_LOAD_GAMMA;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i;
            w_addr_o <= 7'd0;
          end
        end

        S_LOAD_GAMMA: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= idx[6:0] + 7'd1;
          end
          if (deq_valid_out) begin
            gamma_buf[deq_addr_r2] <= dequant_scaled;
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0] + 8'd4) begin
            state    <= S_LOAD_BETA;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i + 6'd1;
            w_addr_o <= 7'd0;
          end
        end

        S_LOAD_BETA: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= idx[6:0] + 7'd1;
          end
          if (deq_valid_out) begin
            beta_buf[deq_addr_r2] <= dequant_scaled;
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0] + 8'd4) begin
            state <= S_NORM_FEED;
            idx   <= 8'd0;
          end
        end

        // Feed elements into norm pipeline
        S_NORM_FEED: begin
          idx <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1) begin
            state     <= S_NORM_DRAIN;
            drain_cnt <= 4'd0;
          end
        end

        // Flush remaining pipeline results, write captured by add valid_o
        S_NORM_DRAIN: begin
          drain_cnt <= drain_cnt + 4'd1;
          if (drain_cnt == 4'd10) begin
            state <= S_LN_DONE;
          end
        end

        S_LN_DONE: begin
          done_o <= 1'b1;
          busy_o <= 1'b0;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule