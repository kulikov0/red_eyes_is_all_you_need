// LayerNorm: y_i = (x_i - mean) / sqrt(var) * gamma + beta
//
// Input: 128 int8 values streamed one per cycle via x_data_i
// Output: 128 int8 values streamed one per cycle via y_data_o
//
// Reads gamma and beta from weight_store via w_sel_o / w_addr_o / w_data_i
//
// FSM: MEAN_ACC(128) -> VAR_ACC(128) -> INV_SQRT(2) ->
//      LOAD_GAMMA(129) -> LOAD_BETA(129) -> NORM(128)
//
// TODO - optimize: LOAD_GAMMA can overlap with MEAN_ACC, LOAD_BETA with
// VAR_ACC when weight_store port is free during input loading, which could reduce cycles

module layernorm #(
  parameter DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,

  // Input stream 
  input  wire signed [7:0] x_data_i,

  // Weight store interface for gamma/beta
  output reg  [5:0]  w_sel_o,
  output reg  [6:0]  w_addr_o,
  input  wire [7:0]  w_data_i,

  // Which LN instance: tensor_sel for gamma, gamma+1 for beta
  input  wire [5:0]  gamma_sel_i,

  // Output stream
  output reg  signed [7:0] y_data_o,
  output reg         y_valid_o,
  output reg         done_o,
  output reg         busy_o
);

  localparam S_IDLE       = 3'd0;
  localparam S_MEAN_ACC   = 3'd1;
  localparam S_VAR_ACC    = 3'd2;
  localparam S_INV_SQRT   = 3'd3;
  localparam S_LOAD_GAMMA = 3'd4;
  localparam S_LOAD_BETA  = 3'd5;
  localparam S_NORM       = 3'd6;

  reg [2:0] state;
  reg [7:0] idx;

  // Mean accumulator: 128 x int8, max sum = 16256, needs 15 bits signed
  reg signed [14:0] mean_acc;
  reg signed [7:0]  mean;

  // Variance accumulator: max = 128 x 255^2 = 8,323,200, needs 24 bits
  reg signed [23:0] var_acc;

  // Input buffer: stores raw inputs, then centered values (9-bit signed)
  reg signed [8:0] x_buf [0:DIM-1];

  // Gamma/beta buffers
  reg signed [7:0] gamma_buf [0:DIM-1];
  reg signed [7:0] beta_buf  [0:DIM-1];

  // inv_sqrt interface
  reg         isqrt_valid;
  reg  [16:0] isqrt_d;
  wire        isqrt_done;
  wire [15:0] isqrt_result;

  inv_sqrt #(
    .D_W(17)
  ) u_isqrt (
    .clk_i   (clk_i),
    .valid_i (isqrt_valid),
    .d_i     (isqrt_d),
    .valid_o (isqrt_done),
    .result_o(isqrt_result)
  );

  reg [15:0] inv_std;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      idx         <= 8'd0;
      mean_acc    <= 15'sd0;
      var_acc     <= 24'd0;
      mean        <= 8'sd0;
      inv_std     <= 16'd0;
      isqrt_valid <= 1'b0;
      y_valid_o   <= 1'b0;
      done_o      <= 1'b0;
      busy_o      <= 1'b0;
      w_sel_o     <= 6'd0;
      w_addr_o    <= 7'd0;

    end else begin
      // Defaults
      y_valid_o   <= 1'b0;
      done_o      <= 1'b0;
      isqrt_valid <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state    <= S_MEAN_ACC;
            idx      <= 8'd0;
            mean_acc <= 15'sd0;
            busy_o   <= 1'b1;
          end
        end

        // Pass 1: accumulate sum, store inputs
        S_MEAN_ACC: begin
          x_buf[idx[6:0]] <= x_data_i;
          mean_acc      <= mean_acc + x_data_i;
          idx           <= idx + 1;
          if (idx == DIM - 1) begin
            state   <= S_VAR_ACC;
            idx     <= 8'd0;
            mean    <= (mean_acc + x_data_i) >>> 7;
            var_acc <= 24'd0;
          end
        end

        // Pass 2: subtract mean, square, accumulate variance
        S_VAR_ACC: begin
          begin : var_blk
            reg signed [8:0] diff;
            diff = x_buf[idx[6:0]] - mean;
            x_buf[idx[6:0]] <= diff;
            var_acc       <= var_acc + diff * diff;
          end
          idx <= idx + 1;
          if (idx == DIM - 1) begin
            state <= S_INV_SQRT;
            idx   <= 8'd0;
          end
        end

        // Feed variance to inv_sqrt, wait for result
        S_INV_SQRT: begin
          if (idx == 8'd0) begin
            isqrt_d     <= var_acc[23:7];
            isqrt_valid <= 1'b1;
            idx         <= 8'd1;
          end else if (isqrt_done) begin
            inv_std  <= isqrt_result;
            state    <= S_LOAD_GAMMA;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i;
            w_addr_o <= 7'd0;
          end
        end

        // Load gamma from weight_store
        // 2-cycle pipeline: registered w_addr_o + registered BRAM read
        // Transition pre-issues addr 0, so loop issues addr 1..127
        // Cycle 0: addr 0 in flight, issue addr 1
        // Cycle 1: capture [0], issue addr 2. ...
        // Cycle 128: capture [127], transition.
        S_LOAD_GAMMA: begin
          if (idx < DIM - 1) begin
            w_sel_o  <= gamma_sel_i;
            w_addr_o <= idx[6:0] + 7'd1;
          end
          if (idx > 0) begin
            gamma_buf[idx[6:0] - 1] <= $signed(w_data_i);
          end
          idx <= idx + 1;
          if (idx == DIM) begin
            state    <= S_LOAD_BETA;
            idx      <= 8'd0;
            w_sel_o  <= gamma_sel_i + 6'd1;
            w_addr_o <= 7'd0;
          end
        end

        // Load beta from weight_store
        S_LOAD_BETA: begin
          if (idx < DIM - 1) begin
            w_sel_o  <= gamma_sel_i + 6'd1;
            w_addr_o <= idx[6:0] + 7'd1;
          end
          if (idx > 0) begin
            beta_buf[idx[6:0] - 1] <= $signed(w_data_i);
          end
          idx <= idx + 1;
          if (idx == DIM) begin
            state <= S_NORM;
            idx   <= 8'd0;
          end
        end

        // Pass 3: normalize and output
        // Combined: (diff * inv_std * gamma + 16384) >>> 15 + beta
        S_NORM: begin
          begin : norm_blk
            reg signed [8:0]  diff;
            reg signed [32:0] full_prod;
            reg signed [17:0] biased;
            diff      = x_buf[idx[6:0]];
            full_prod = diff * $signed({1'b0, inv_std}) * gamma_buf[idx[6:0]];
            biased    = ((full_prod + 33'sd16384) >>> 15) + beta_buf[idx[6:0]];
            y_data_o  <= (biased > 18'sd127)  ?  8'sd127 :
                         (biased < -18'sd128) ? -8'sd128 :
                          biased[7:0];
          end
          y_valid_o <= 1'b1;
          idx       <= idx + 1;
          if (idx == DIM - 1) begin
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
