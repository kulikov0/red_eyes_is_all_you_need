// Matrix-vector multiply: int8 weights x fp16 input -> fp16 output
//
// Snapshots input from shared RAM into internal distributed RAM, then computes
// Writes result to shared RAM. Safe when input/output share the same RAM
//
// Dequant pipelined: BRAM+int8+mul_scale | mul_input+acc, 2 stages
//
// Latency: IN_DIM + OUT_DIM * (IN_DIM + 1) + 3 cycles

module matvec_fp16 #(
  parameter IN_DIM  = 128,
  parameter OUT_DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,
  input  wire [15:0] scale_i,

  // Weight BRAM
  output reg [$clog2(OUT_DIM*IN_DIM)-1:0] weight_addr_o,
  input  wire signed [7:0]                weight_data_i,

  output reg [$clog2(IN_DIM)-1:0] act_raddr_o,
  input  wire [15:0]              act_rdata_i,

  output reg                        res_we_o,
  output reg  [$clog2(OUT_DIM)-1:0] res_waddr_o,
  output reg  [15:0]                res_wdata_o,

  output reg  done_o
);

  localparam ADDR_W = $clog2(OUT_DIM * IN_DIM);
  localparam COL_W  = $clog2(IN_DIM) + 1;
  localparam ROW_W  = $clog2(OUT_DIM) + 1;

  // Internal snapshot of input vector
  reg [15:0] in_snap [0:IN_DIM-1];

  // Dequant: int8 -> fp16 -> fp16*scale
  wire [15:0] w_fp16;
  fp16_from_int8 u_dequant_cvt (
    .val_i(weight_data_i),
    .fp16_o(w_fp16)
  );

  wire [15:0] w_dequant;
  fp16_mul_comb u_dequant_mul (
    .a_i(w_fp16),
    .b_i(scale_i),
    .prod_o(w_dequant)
  );

  // Pipeline register between dequant and MAC
  reg [15:0]      dq_r;
  reg [COL_W-1:0] dq_col;
  reg             dq_valid;
  reg             dq_last;

  // MAC from registered dequant: dq_r * in_snap[dq_col] + acc
  wire [15:0] mac_prod;
  fp16_mul_comb u_mac_mul (
    .a_i(dq_r),
    .b_i(in_snap[dq_col]),
    .prod_o(mac_prod)
  );

  reg [15:0] acc;
  wire [15:0] acc_sum;
  fp16_add_comb u_acc_add (
    .a_i(acc),
    .b_i(mac_prod),
    .sum_o(acc_sum)
  );

  reg [COL_W-1:0] col;
  reg [ROW_W-1:0] row;      // output row counter, incremented on dq_last
  reg [ROW_W-1:0] feed_row; // input row counter, leads row by 1 pipeline cycle
  reg loading;
  reg prefetch;
  reg running;
  reg draining;

  always @(posedge clk_i) begin
    if (rst_i) begin
      acc           <= 16'd0;
      col           <= {COL_W{1'b0}};
      row           <= {ROW_W{1'b0}};
      loading       <= 1'b0;
      prefetch      <= 1'b0;
      running       <= 1'b0;
      draining      <= 1'b0;
      done_o        <= 1'b0;
      res_we_o      <= 1'b0;
      act_raddr_o   <= {$clog2(IN_DIM){1'b0}};
      weight_addr_o <= {ADDR_W{1'b0}};
      dq_valid      <= 1'b0;
      dq_last       <= 1'b0;

    end else if (start_i) begin
      loading       <= 1'b1;
      prefetch      <= 1'b0;
      running       <= 1'b0;
      draining      <= 1'b0;
      act_raddr_o   <= {$clog2(IN_DIM){1'b0}};
      col           <= {COL_W{1'b0}};
      done_o        <= 1'b0;
      res_we_o      <= 1'b0;
      dq_valid      <= 1'b0;

    end else if (loading) begin
      in_snap[act_raddr_o] <= act_rdata_i;
      if (act_raddr_o == (IN_DIM - 1)) begin
        loading       <= 1'b0;
        prefetch      <= 1'b1;
        col           <= {COL_W{1'b0}};
        row           <= {ROW_W{1'b0}};
        feed_row      <= {ROW_W{1'b0}};
        acc           <= 16'd0;
        weight_addr_o <= {ADDR_W{1'b0}};
      end else begin
        act_raddr_o <= act_raddr_o + 1;
      end

    end else if (prefetch) begin
      // First BRAM read latency cycle
      prefetch      <= 1'b0;
      running       <= 1'b1;
      weight_addr_o <= weight_addr_o + 1;
      dq_valid      <= 1'b0;

    end else if (running || draining) begin
      res_we_o <= 1'b0;

      // Register dequant result and col index
      if (running) begin
        dq_r     <= w_dequant;
        dq_col   <= col;
        dq_valid <= 1'b1;
        dq_last  <= (col == IN_DIM[COL_W-1:0] - 1);
      end else begin
        dq_valid <= 1'b0;
      end

      // MAC accumulate from pipeline register
      if (dq_valid) begin
        if (dq_last) begin
          res_we_o    <= 1'b1;
          res_waddr_o <= row[$clog2(OUT_DIM)-1:0];
          res_wdata_o <= acc_sum;
          acc         <= 16'd0;
          row         <= row + 1;
        end else begin
          acc <= acc_sum;
        end
      end

      // Advance weight address and column
      if (running) begin
        if (col == IN_DIM[COL_W-1:0] - 1) begin
          col <= {COL_W{1'b0}};
          if (feed_row == OUT_DIM[ROW_W-1:0] - 1) begin
            running  <= 1'b0;
            draining <= 1'b1;
          end else begin
            feed_row      <= feed_row + 1;
            weight_addr_o <= weight_addr_o + 1;
          end
        end else begin
          col           <= col + 1;
          weight_addr_o <= weight_addr_o + 1;
        end
      end

      // Drain: last dq_valid consumed, emit done
      if (draining && !dq_valid) begin
        draining <= 1'b0;
        done_o   <= 1'b1;
      end

    end else begin
      done_o   <= 1'b0;
      res_we_o <= 1'b0;
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