// LayerNorm: y_i = (x_i - mean) / sqrt(var) * gamma + beta
// Ref: https://www.mdpi.com/2072-666X/17/1/84 (LOD-LUT rsqrt)
//
// Input: 128 fp16 values as flat bus (DIM*16 bits)
// Output: 128 fp16 values as flat bus (DIM*16 bits)
//
// Reads gamma and beta int8 from weight_store, dequants to fp16 via
// fp16_from_int8(byte) * scale
//
// FSM: IDLE -> MEAN_ACC(128) -> MEAN_DIV(1) -> VAR_ACC(128) ->
//      VAR_DIV(1) -> INV_SQRT(2) -> LOAD_GAMMA(129) -> LOAD_BETA(129) ->
//      NORM(128)
// Latency: ~646 cycles
//
// TODO: optimize - LOAD_GAMMA can overlap with MEAN_ACC, LOAD_BETA with
// VAR_ACC when weight_store port is free during input loading, saving ~258 cycles

module layernorm #(
  parameter DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,

  // Input: DIM x fp16 (flat bus)
  input  wire [DIM*16-1:0] x_i,

  // Weight store interface for gamma/beta
  output reg  [5:0]  w_sel_o,
  output reg  [6:0]  w_addr_o,
  input  wire [7:0]  w_data_i,

  // Which LN instance: tensor_sel for gamma, gamma+1 for beta
  input  wire [5:0]  gamma_sel_i,

  // FP16 dequant scales for gamma and beta
  input  wire [15:0] gamma_scale_i,
  input  wire [15:0] beta_scale_i,

  // Output: DIM x fp16 (flat bus)
  output reg  [DIM*16-1:0] y_o,
  output reg               done_o,
  output reg               busy_o
);

  localparam S_IDLE       = 4'd0;
  localparam S_MEAN_ACC   = 4'd1;
  localparam S_MEAN_DIV   = 4'd2;
  localparam S_VAR_ACC    = 4'd3;
  localparam S_VAR_DIV    = 4'd4;
  localparam S_INV_SQRT   = 4'd5;
  localparam S_LOAD_GAMMA = 4'd6;
  localparam S_LOAD_BETA  = 4'd7;
  localparam S_NORM       = 4'd8;

  reg [3:0] state;
  reg [7:0] idx;

  // FP16 accumulators
  reg [15:0] sum_acc;
  reg [15:0] neg_mean;
  reg [15:0] var_acc;
  reg [15:0] inv_std;

  // Gamma/beta buffers (fp16, dequanted)
  reg [15:0] gamma_buf [0:DIM-1];
  reg [15:0] beta_buf  [0:DIM-1];

  // Current input element
  wire [15:0] x_elem = x_i[idx[6:0]*16 +: 16];

  // fp16(1/128) = 0x2000: sign=0, exp=8, frac=0 -> 2^(8-15) = 2^-7 = 1/128
  localparam [15:0] INV_N = 16'h2000;

  // Combinational fp16 arithmetic

  // MEAN_ACC: sum_acc + x[idx]
  wire [15:0] mean_add_out;
  fp16_add_comb u_mean_add (.a_i(sum_acc), .b_i(x_elem), .sum_o(mean_add_out));

  // MEAN_DIV: sum * (1/128)
  wire [15:0] mean_div_out;
  fp16_mul_comb u_mean_div (.a_i(sum_acc), .b_i(INV_N), .prod_o(mean_div_out));

  // VAR_ACC: diff = x - mean, sq = diff * diff, var_acc + sq
  wire [15:0] var_diff;
  fp16_add_comb u_var_sub (.a_i(x_elem), .b_i(neg_mean), .sum_o(var_diff));

  wire [15:0] var_sq;
  fp16_mul_comb u_var_sq (.a_i(var_diff), .b_i(var_diff), .prod_o(var_sq));

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

  wire [15:0] dequant_gamma;
  fp16_mul_comb u_deq_gamma (.a_i(dequant_fp16), .b_i(gamma_scale_i), .prod_o(dequant_gamma));

  wire [15:0] dequant_beta;
  fp16_mul_comb u_deq_beta (.a_i(dequant_fp16), .b_i(beta_scale_i), .prod_o(dequant_beta));

  // NORM: y = (x - mean) * inv_std * gamma + beta
  wire [15:0] norm_diff;
  fp16_add_comb u_norm_sub (.a_i(x_elem), .b_i(neg_mean), .sum_o(norm_diff));

  wire [15:0] norm_scaled;
  fp16_mul_comb u_norm_mul1 (.a_i(norm_diff), .b_i(inv_std), .prod_o(norm_scaled));

  wire [15:0] norm_gamma;
  fp16_mul_comb u_norm_mul2 (.a_i(norm_scaled), .b_i(gamma_buf[idx[6:0]]), .prod_o(norm_gamma));

  wire [15:0] norm_out;
  fp16_add_comb u_norm_add (.a_i(norm_gamma), .b_i(beta_buf[idx[6:0]]), .sum_o(norm_out));

  // BRAM pipeline index for gamma/beta capture
  wire [7:0] prev = idx - 8'd1;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      idx         <= 8'd0;
      sum_acc     <= 16'd0;
      var_acc     <= 16'd0;
      neg_mean    <= 16'd0;
      inv_std     <= 16'd0;
      rsqrt_valid <= 1'b0;
      done_o      <= 1'b0;
      busy_o      <= 1'b0;
      w_sel_o     <= 6'd0;
      w_addr_o    <= 7'd0;

    end else begin
      done_o      <= 1'b0;
      rsqrt_valid <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state   <= S_MEAN_ACC;
            idx     <= 8'd0;
            sum_acc <= 16'd0;
            busy_o  <= 1'b1;
          end
        end

        // Pass 1: accumulate fp16 sum of all inputs
        S_MEAN_ACC: begin
          sum_acc <= mean_add_out;
          idx     <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1) begin
            state <= S_MEAN_DIV;
          end
        end

        // Compute mean = sum * (1/128), neg_mean = -mean
        S_MEAN_DIV: begin
          neg_mean <= {~mean_div_out[15], mean_div_out[14:0]};
          var_acc  <= 16'd0;
          idx      <= 8'd0;
          state    <= S_VAR_ACC;
        end

        // Pass 2: accumulate variance = sum((x - mean)^2)
        S_VAR_ACC: begin
          var_acc <= var_add_out;
          idx     <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1) begin
            state <= S_VAR_DIV;
          end
        end

        // Compute var = var_acc * (1/128), launch rsqrt
        S_VAR_DIV: begin
          rsqrt_valid <= 1'b1;
          idx         <= 8'd0;
          state       <= S_INV_SQRT;
        end

        // Wait for fp16_rsqrt result (2 cycles)
        S_INV_SQRT: begin
          if (rsqrt_done) begin
            inv_std  <= rsqrt_result;
            state    <= S_LOAD_GAMMA;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i;
            w_addr_o <= 7'd0;
          end
        end

        // Load gamma from weight_store, dequant to fp16
        // 2-cycle BRAM pipeline: addr registered + data registered
        S_LOAD_GAMMA: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= idx[6:0] + 7'd1;
          end
          if (idx > 0) begin
            gamma_buf[prev[6:0]] <= dequant_gamma;
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0]) begin
            state    <= S_LOAD_BETA;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i + 6'd1;
            w_addr_o <= 7'd0;
          end
        end

        // Load beta from weight_store, dequant to fp16
        S_LOAD_BETA: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= idx[6:0] + 7'd1;
          end
          if (idx > 0) begin
            beta_buf[prev[6:0]] <= dequant_beta;
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0]) begin
            state <= S_NORM;
            idx   <= 8'd0;
          end
        end

        // Pass 3: normalize and output
        S_NORM: begin
          y_o[idx[6:0]*16 +: 16] <= norm_out;
          idx <= idx + 8'd1;
          if (idx == DIM[7:0] - 8'd1) begin
            state  <= S_IDLE;
            done_o <= 1'b1;
            busy_o <= 1'b0;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule