// LayerNorm: y_i = (x_i - mean) / sqrt(var) * gamma + beta
// Ref: https://www.mdpi.com/2072-666X/17/1/84
//
// Reads gamma/beta int8 from weight_store, dequants to fp16
//
// VAR stage uses pipelined fp16_add (sub) and fp16_mul (sq), feeding into
// a combinational accumulator add. 5-cycle drain after last issue
// NORM pipelined: sub | mul_rsqrt | mul_gamma | add_beta, 4 stages
//
// FSM: IDLE -> MEAN_ACC -> MEAN_DIV -> VAR_ACC -> VAR_DRAIN -> VAR_DIV ->
//      INV_SQRT -> LOAD_GAMMA -> LOAD_BETA -> NORM_FEED -> NORM_DRAIN
// Latency: ~657 cycles

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
  localparam S_VAR_DRAIN  = 4'd4;
  localparam S_VAR_DIV    = 4'd5;
  localparam S_INV_SQRT   = 4'd6;
  localparam S_LOAD_GAMMA = 4'd7;
  localparam S_LOAD_BETA  = 4'd8;
  localparam S_NORM_FEED  = 4'd9;
  localparam S_NORM_DRAIN = 4'd10;
  localparam S_LN_DONE    = 4'd11;

  reg [3:0] state;
  reg [7:0] idx;
  wire [7:0] prev = idx - 8'd1;

  // FP16 accumulators
  reg [15:0] sum_acc;
  reg [15:0] neg_mean;
  reg [15:0] var_acc;
  reg [15:0] inv_std;

  // Gamma/beta buffers (fp16, dequanted)
  reg [15:0] gamma_buf [0:DIM-1];
  reg [15:0] beta_buf  [0:DIM-1];

  assign x_raddr_o = idx[$clog2(DIM)-1:0];
  wire [15:0] x_elem = x_rdata_i;

  // fp16(1/128) = 0x2000: sign=0, exp=8, frac=0 -> 2^(8-15) = 2^-7 = 1/128
  localparam [15:0] INV_N = 16'h2000;

  // Combinational fp16 arithmetic

  // MEAN_ACC: sum_acc + x[idx]
  wire [15:0] mean_add_out;
  fp16_add_comb u_mean_add (.a_i(sum_acc), .b_i(x_elem), .sum_o(mean_add_out));

  // MEAN_DIV: sum * (1/128)
  wire [15:0] mean_div_out;
  fp16_mul_comb u_mean_div (.a_i(sum_acc), .b_i(INV_N), .prod_o(mean_div_out));

  // Variance: diff = x - mean
  wire        var_issue = (state == S_VAR_ACC);
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

  // Variance accumulator: single acc, combinational feedback
  wire [15:0] var_add_out;
  fp16_add_comb u_var_add (.a_i(var_acc), .b_i(var_sq), .sum_o(var_add_out));

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

  // LOAD_GAMMA/BETA: dequant int8 -> fp16
  wire [15:0] dequant_fp16;
  fp16_from_int8 u_dequant (.val_i(w_data_i), .fp16_o(dequant_fp16));

  wire deq_valid_in = ((state == S_LOAD_GAMMA) || (state == S_LOAD_BETA)) &&
                      (idx >= 8'd1) && (idx <= DIM[7:0]);

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

  // Track gamma/beta write address through 2-cycle dequant pipeline
  reg [$clog2(DIM)-1:0] deq_addr_r1, deq_addr_r2;
  always @(posedge clk_i) begin
    deq_addr_r1 <= prev[$clog2(DIM)-1:0];
    deq_addr_r2 <= deq_addr_r1;
  end

  // Normalize: diff = x - mean
  wire [15:0] norm_diff;
  fp16_add_comb u_norm_sub (.a_i(x_elem), .b_i(neg_mean), .sum_o(norm_diff));

  reg [15:0]             p1_diff;
  reg [$clog2(DIM)-1:0]  p1_idx;
  reg                    p1_valid;

  // scaled = diff * inv_std
  wire [15:0] norm_scaled;
  fp16_mul_comb u_norm_mul1 (.a_i(p1_diff), .b_i(inv_std), .prod_o(norm_scaled));

  reg [15:0]             p2_scaled;
  reg [$clog2(DIM)-1:0]  p2_idx;
  reg                    p2_valid;

  // gamma_applied = scaled * gamma
  wire [15:0] norm_gamma;
  fp16_mul_comb u_norm_mul2 (.a_i(p2_scaled), .b_i(gamma_buf[p2_idx]), .prod_o(norm_gamma));

  reg [15:0]             p3_gamma;
  reg [$clog2(DIM)-1:0]  p3_idx;
  reg                    p3_valid;

  // out = gamma_applied + beta
  wire [15:0] norm_out;
  fp16_add_comb u_norm_add (.a_i(p3_gamma), .b_i(beta_buf[p3_idx]), .sum_o(norm_out));

  // Drain counter: used by S_VAR_DRAIN and S_NORM_DRAIN
  reg [2:0] drain_cnt;

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
      p1_valid     <= 1'b0;
      p2_valid     <= 1'b0;
      p3_valid     <= 1'b0;
      drain_cnt    <= 3'd0;

    end else begin
      done_o      <= 1'b0;
      rsqrt_valid <= 1'b0;
      y_we_o      <= 1'b0;

      // NORM pipeline advance runs during NORM_FEED and NORM_DRAIN
      if (state == S_NORM_FEED || state == S_NORM_DRAIN) begin
        p1_diff  <= norm_diff;
        p1_idx   <= idx[$clog2(DIM)-1:0];
        p1_valid <= (state == S_NORM_FEED);

        p2_scaled <= norm_scaled;
        p2_idx    <= p1_idx;
        p2_valid  <= p1_valid;

        p3_gamma <= norm_gamma;
        p3_idx   <= p2_idx;
        p3_valid <= p2_valid;

        if (p3_valid) begin
          y_we_o    <= 1'b1;
          y_waddr_o <= p3_idx;
          y_wdata_o <= norm_out;
        end
      end

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state   <= S_MEAN_ACC;
            idx     <= 8'd0;
            sum_acc <= 16'd0;
            busy_o  <= 1'b1;
          end
        end

        S_MEAN_ACC: begin
          sum_acc <= mean_add_out;
          idx     <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1) begin
            state <= S_MEAN_DIV;
          end
        end

        S_MEAN_DIV: begin
          neg_mean  <= {~mean_div_out[15], mean_div_out[14:0]};
          var_acc   <= 16'd0;
          idx       <= 8'd0;
          state     <= S_VAR_ACC;
        end

        // Issue one (x[idx] - mean) sub per cycle, let sub->mul pipeline flow.
        // Accumulate var_sq into var_acc combinationally whenever mul output is valid.
        S_VAR_ACC: begin
          if (var_sq_valid) begin
            var_acc <= var_add_out;
          end
          if (idx == DIM[7:0] - 8'd1) begin
            state     <= S_VAR_DRAIN;
            drain_cnt <= 3'd0;
          end else begin
            idx <= idx + 8'd1;
          end
        end

        // Wait for last 5 cycles of in-flight sub+mul to finish, keep accumulating
        S_VAR_DRAIN: begin
          if (var_sq_valid) begin
            var_acc <= var_add_out;
          end
          if (drain_cnt == 3'd5) begin
            state <= S_VAR_DIV;
          end else begin
            drain_cnt <= drain_cnt + 3'd1;
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
          if (idx == DIM[7:0] + 8'd2) begin
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
          if (idx == DIM[7:0] + 8'd2) begin
            state    <= S_NORM_FEED;
            idx      <= 8'd0;
            p1_valid <= 1'b0;
            p2_valid <= 1'b0;
            p3_valid <= 1'b0;
          end
        end

        // Feed elements into norm pipeline
        S_NORM_FEED: begin
          idx <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1) begin
            state     <= S_NORM_DRAIN;
            drain_cnt <= 3'd0;
          end
        end

        // Flush remaining pipeline results
        S_NORM_DRAIN: begin
          drain_cnt <= drain_cnt + 3'd1;
          if (drain_cnt == 3'd2) begin
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