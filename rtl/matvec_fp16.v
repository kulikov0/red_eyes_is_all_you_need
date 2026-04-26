// Matrix-vector multiply: int8 weights x fp16 input -> fp16 output
//
// Snapshots input from shared RAM into internal distributed RAM, then computes
// Writes result to shared RAM. Safe when input/output share the same RAM
//
// Pipeline: BRAM -> dequant fp16_mul -> MAC fp16_mul -> accumulate fp16_add
// Accumulator feedback forces K=4 interleaved rows to cover add feedback
// Assumes OUT_DIM % 4 == 0 and IN_DIM is a power of 2

module matvec_fp16 #(
  parameter IN_DIM  = 128,
  parameter OUT_DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,
  input  wire [15:0] scale_i,

  // Weight BRAM
  output wire [$clog2(OUT_DIM*IN_DIM)-1:0] weight_addr_o,
  input  wire signed [7:0]                 weight_data_i,

  output reg [$clog2(IN_DIM)-1:0] act_raddr_o,
  input  wire [15:0]              act_rdata_i,

  output reg                        res_we_o,
  output reg  [$clog2(OUT_DIM)-1:0] res_waddr_o,
  output reg  [15:0]                res_wdata_o,

  output reg  done_o
);

  localparam K     = 4;
  localparam K_LOG = 2;
  localparam COL_W = $clog2(IN_DIM);
  localparam GRP_N = OUT_DIM / K;
  localparam GRP_W = (GRP_N <= 1) ? 1 : $clog2(GRP_N);

  // Input activation snapshot
  reg [15:0] in_snap [0:IN_DIM-1];

  // K interleaved accumulators
  reg [15:0] acc [0:K-1];

  // FSM
  localparam S_IDLE    = 3'd0;
  localparam S_LOAD    = 3'd1;
  localparam S_ZERO    = 3'd2;
  localparam S_COMPUTE = 3'd3;
  localparam S_DRAIN   = 3'd4;
  localparam S_WRITE   = 3'd5;
  localparam S_DONE    = 3'd6;

  reg [2:0] state;

  reg [GRP_W-1:0]   group;
  reg [COL_W-1:0]   col;
  reg [K_LOG-1:0]   k;
  reg [3:0]         drain_cnt;
  reg [K_LOG-1:0]   write_k;

  // Combinational weight address: BRAM samples this at posedge, data appears next cycle
  assign weight_addr_o = group * (K * IN_DIM) + k * IN_DIM + col;

  // Dequant valid: 1-cycle delay of state==S_COMPUTE to align with BRAM output
  reg state_compute_r;
  always @(posedge clk_i) begin
    if (rst_i) state_compute_r <= 1'b0;
    else       state_compute_r <= (state == S_COMPUTE);
  end
  wire dq_valid_in = state_compute_r;

  // Dequant: int8 weight -> fp16 -> multiply by scale
  wire [15:0] w_fp16;
  fp16_from_int8 u_from (.val_i(weight_data_i), .fp16_o(w_fp16));

  wire        dq_valid_out;
  wire [15:0] w_dequant;
  fp16_mul u_dq (
    .clk_i(clk_i),
    .valid_i(dq_valid_in),
    .a_i(w_fp16),
    .b_i(scale_i),
    .valid_o(dq_valid_out),
    .prod_o(w_dequant)
  );

  // Metadata shift registers track (col, k) through each pipeline stage
  // col needed at u_mac input (3 cycles after BRAM sample = 3 stages from col)
  // k needed at u_add input (5 cycles from k sample) and acc writeback (8 cycles)
  reg [COL_W-1:0] col_pipe [0:2];
  reg [K_LOG-1:0] k_pipe   [0:7];
  integer i;
  always @(posedge clk_i) begin
    col_pipe[0] <= col;
    col_pipe[1] <= col_pipe[0];
    col_pipe[2] <= col_pipe[1];

    k_pipe[0] <= k;
    for (i = 1; i < 8; i = i + 1) k_pipe[i] <= k_pipe[i-1];
  end

  // MAC multiply: dequanted weight * matching input element
  wire        mac_valid_out;
  wire [15:0] mac_prod;
  fp16_mul u_mac (
    .clk_i(clk_i),
    .valid_i(dq_valid_out),
    .a_i(w_dequant),
    .b_i(in_snap[col_pipe[2]]),
    .valid_o(mac_valid_out),
    .prod_o(mac_prod)
  );

  // Accumulator add: feeds acc[k] back through 3-cycle fp16_add
  // K=4 row interleaving covers the 3-cycle feedback with 1 cycle to spare
  wire        add_valid_out;
  wire [15:0] add_sum;
  fp16_add u_add (
    .clk_i(clk_i),
    .valid_i(mac_valid_out),
    .a_i(acc[k_pipe[4]]),
    .b_i(mac_prod),
    .valid_o(add_valid_out),
    .sum_o(add_sum)
  );


  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      done_o      <= 1'b0;
      res_we_o    <= 1'b0;
      act_raddr_o <= 0;
      group       <= 0;
      col         <= 0;
      k           <= 0;
      drain_cnt   <= 0;
      write_k     <= 0;

    end else begin
      done_o   <= 1'b0;
      res_we_o <= 1'b0;

      // Write back accumulator when add output is valid
      if (add_valid_out) begin
        acc[k_pipe[7]] <= add_sum;
      end

      case (state)
        S_IDLE: begin
          if (start_i) begin
            state       <= S_LOAD;
            act_raddr_o <= 0;
            group       <= 0;
          end
        end

        S_LOAD: begin
          in_snap[act_raddr_o] <= act_rdata_i;
          if (act_raddr_o == IN_DIM - 1) begin
            state <= S_ZERO;
          end else begin
            act_raddr_o <= act_raddr_o + 1;
          end
        end

        S_ZERO: begin
          for (i = 0; i < K; i = i + 1) acc[i] <= 16'd0;
          col   <= 0;
          k     <= 0;
          state <= S_COMPUTE;
        end

        // Issue one MAC per cycle, interleaving K rows across consecutive cycles
        // Order: (col=0,k=0), (col=0,k=1), ..., (col=0,k=3), (col=1,k=0), ...
        // Each row sees a new MAC every K cycles
        S_COMPUTE: begin
          if (k == K - 1) begin
            k <= 0;
            if (col == IN_DIM - 1) begin
              state     <= S_DRAIN;
              drain_cnt <= 0;
            end else begin
              col <= col + 1;
            end
          end else begin
            k <= k + 1;
          end
        end

        // Wait for last in-flight MACs to propagate through BRAM+mul+mul+add (8 cycles)
        S_DRAIN: begin
          if (drain_cnt == 4'd7) begin
            state   <= S_WRITE;
            write_k <= 0;
          end else begin
            drain_cnt <= drain_cnt + 1;
          end
        end

        S_WRITE: begin
          res_we_o    <= 1'b1;
          res_waddr_o <= group * K + write_k;
          res_wdata_o <= acc[write_k];
          if (write_k == K - 1) begin
            if (group == GRP_N - 1) begin
              state <= S_DONE;
            end else begin
              group <= group + 1;
              state <= S_ZERO;
            end
          end else begin
            write_k <= write_k + 1;
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

// Combinational fp16 adder (same logic as fp16_add but no output register)
module fp16_add_comb (
  input  wire [15:0] a_i,
  input  wire [15:0] b_i,
  output wire [15:0] sum_o
);

  wire        a_sign = a_i[15];
  wire [4:0]  a_exp  = a_i[14:10];
  wire [9:0]  a_mant = a_i[9:0];
  wire        b_sign = b_i[15];
  wire [4:0]  b_exp  = b_i[14:10];
  wire [9:0]  b_mant = b_i[9:0];

  wire a_is_zero = (a_exp == 5'd0);
  wire b_is_zero = (b_exp == 5'd0);
  wire a_is_inf  = (a_exp == 5'd31) && (a_mant == 10'd0);
  wire b_is_inf  = (b_exp == 5'd31) && (b_mant == 10'd0);
  wire a_is_nan  = (a_exp == 5'd31) && (a_mant != 10'd0);
  wire b_is_nan  = (b_exp == 5'd31) && (b_mant != 10'd0);

  wire [10:0] a_full = a_is_zero ? 11'd0 : {1'b1, a_mant};
  wire [10:0] b_full = b_is_zero ? 11'd0 : {1'b1, b_mant};

  wire a_ge_b = (a_exp > b_exp) || ((a_exp == b_exp) && (a_full >= b_full));

  wire        lg_sign = a_ge_b ? a_sign : b_sign;
  wire [4:0]  lg_exp  = a_ge_b ? a_exp  : b_exp;
  wire [10:0] lg_mant = a_ge_b ? a_full : b_full;
  wire        sm_sign = a_ge_b ? b_sign : a_sign;
  wire [10:0] sm_mant = a_ge_b ? b_full : a_full;
  wire [4:0]  sm_exp  = a_ge_b ? b_exp  : a_exp;

  wire [4:0]  exp_diff = lg_exp - sm_exp;
  wire [13:0] lg_ext = {1'b0, lg_mant, 2'b00};
  wire [26:0] sm_wide = {1'b0, sm_mant, 2'b00, 13'b0};
  wire [26:0] sm_shifted = sm_wide >> exp_diff;
  wire [13:0] sm_ext = sm_shifted[26:13];
  wire        sticky  = |sm_shifted[12:0];

  wire eff_sub = lg_sign ^ sm_sign;
  wire [14:0] mant_sum = eff_sub ? ({1'b0, lg_ext} - {1'b0, sm_ext}) :
                                   ({1'b0, lg_ext} + {1'b0, sm_ext});

  reg [3:0] lod;
  reg       sum_is_zero;
  integer i;
  always @(*) begin
    lod = 4'd0;
    sum_is_zero = (mant_sum[14:0] == 15'd0);
    for (i = 0; i < 15; i = i + 1) begin
      if (mant_sum[i]) lod = i[3:0];
    end
  end

  wire overflow = (lod == 4'd13) || (lod == 4'd14);
  wire [3:0] rshift_amt = (lod > 4'd12) ? (lod - 4'd12) : 4'd0;
  wire [3:0] lshift_amt = (lod < 4'd12) ? (4'd12 - lod) : 4'd0;

  wire [14:0] norm_mant = sum_is_zero ? 15'd0 :
                          overflow    ? (mant_sum >> rshift_amt) :
                                        (mant_sum << lshift_amt);

  wire signed [6:0] lg_exp_s = $signed({2'b0, lg_exp});
  wire signed [6:0] rsh_s    = $signed({3'b0, rshift_amt});
  wire signed [6:0] lsh_s    = $signed({3'b0, lshift_amt});
  wire signed [6:0] exp_adj_s = sum_is_zero ? 7'sd0 :
                                overflow    ? (lg_exp_s + rsh_s) :
                                              (lg_exp_s - lsh_s);

  wire [9:0] trunc_mant = norm_mant[11:2];
  wire       guard_bit  = norm_mant[1];
  wire       round_bit  = norm_mant[0];
  wire       extra_sticky = overflow ? |mant_sum[0] : 1'b0;
  wire       sticky_bit   = sticky | extra_sticky;
  wire       use_sticky = sticky_bit & ~eff_sub;
  wire       round_up = guard_bit & (round_bit | use_sticky | trunc_mant[0]);

  wire [10:0] rounded_mant = {1'b0, trunc_mant} + {10'd0, round_up};
  wire        round_ovf = rounded_mant[10];
  wire signed [6:0] final_exp_s = round_ovf ? (exp_adj_s + 7'sd1) : exp_adj_s;

  wire [15:0] normal_result = {lg_sign, final_exp_s[4:0], rounded_mant[9:0]};
  wire exp_overflow = (final_exp_s >= 7'sd31);
  wire [15:0] inf_result = {lg_sign, 5'd31, 10'd0};
  wire exp_underflow = (final_exp_s <= 7'sd0) && !sum_is_zero;
  wire [15:0] zero_result = {lg_sign, 15'd0};
  wire [15:0] nan_result = 16'h7E00;

  reg [15:0] result;
  always @(*) begin
    if (a_is_nan || b_is_nan) begin
      result = nan_result;
    end else if (a_is_inf && b_is_inf && eff_sub) begin
      result = nan_result;
    end else if (a_is_inf || b_is_inf) begin
      result = a_is_inf ? {a_sign, 5'd31, 10'd0} : {b_sign, 5'd31, 10'd0};
    end else if (a_is_zero && b_is_zero) begin
      result = {a_sign & b_sign, 15'd0};
    end else if (a_is_zero) begin
      result = b_i;
    end else if (b_is_zero) begin
      result = a_i;
    end else if (sum_is_zero) begin
      result = 16'd0;
    end else if (exp_underflow) begin
      result = zero_result;
    end else if (exp_overflow) begin
      result = inf_result;
    end else begin
      result = normal_result;
    end
  end
  assign sum_o = result;

endmodule