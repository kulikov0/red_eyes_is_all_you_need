// Sampler: fp16 argmax stub
//
// Scans 256 fp16 logits and outputs the index of the maximum value
// FSM: IDLE -> SCAN (256 cycles) -> DONE (1 cycle)
// Latency: 258 cycles
//
//
// TODO: replace with full sampler (temperature scaling,
// repetition penalty, top-k, LFSR sampling)

module sampler (
  input  wire             clk_i,
  input  wire             rst_i,
  input  wire             start_i,
  input  wire [256*16-1:0] logits_i,
  output reg  [7:0]       token_o,
  output reg              done_o
);

  localparam S_IDLE = 2'd0;
  localparam S_SCAN = 2'd1;
  localparam S_DONE = 2'd2;

  reg [1:0]  state;
  reg [8:0]  scan_idx;
  reg [15:0] best_val;
  reg [7:0]  best_idx;

  wire [15:0] cur_logit = logits_i[scan_idx[7:0]*16 +: 16];

  // FP16 greater-than comparison (a > b)
  // Handles positive and negative values correctly
  wire a_neg  = cur_logit[15];
  wire b_neg  = best_val[15];
  wire mag_gt = cur_logit[14:0] > best_val[14:0];
  wire a_gt_b = (a_neg != b_neg) ? b_neg :
                a_neg ? ~mag_gt : mag_gt;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state    <= S_IDLE;
      done_o   <= 1'b0;
      token_o  <= 8'd0;
      scan_idx <= 9'd0;
      best_val <= 16'hFC00;
      best_idx <= 8'd0;

    end else begin
      done_o <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state    <= S_SCAN;
            scan_idx <= 9'd0;
            best_val <= 16'hFC00;
            best_idx <= 8'd0;
          end
        end

        S_SCAN: begin
          if (a_gt_b) begin
            best_val <= cur_logit;
            best_idx <= scan_idx[7:0];
          end
          scan_idx <= scan_idx + 9'd1;
          if (scan_idx == 9'd255) begin
            state <= S_DONE;
          end
        end

        S_DONE: begin
          token_o <= best_idx;
          done_o  <= 1'b1;
          state   <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule