// Hardware SafeSoftmax with bipartite exp(-d) LUT
//
// Based on https://www.mdpi.com/2072-666X/17/1/84
//
// Algorithm (SafeSoftmax):
//   softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
//
// Division-free normalization via log domain:
//   ln(sum) computed from LOD decomposition
//   softmax(x_i) = exp(-(d_i + ln(sum)))  where d_i = max - x_i
//
// Bipartite LUT for exp(-d):
//   11-bit Q4.7 input split as (x0,x1,x2) = (5,3,3)
//   exp(-d) ~ LUT_0[{x0,x1}] + LUT_1[{x0,x2}]
//   Output Q1.15 unsigned (16-bit)
//
// Pipeline per element (fully pipelined, 1 element/cycle throughput):
//   Posedge T:   buf_raddr <= j, p0_valid <= 1
//   Posedge T+1: BRAM: buf_rdata = input_buf[j]
//                Combinational: d, LUT addrs from buf_rdata (element j)
//                LUT BRAM captures: lut0_out <= exp_lut0[addr_j]
//                Register: clip_r <= d_clipped(j), p1_valid <= p0_valid
//   Posedge T+2: lut0_out = LUT[j], clip_r = clip[j], p1_valid = 1
//                -> accumulate exp_val or output
//
// Total latency: N+2 cycles per pass x 2 passes + 1 LN_SUM = ~2N+5 cycles
// Resources: 3 BRAM18 (input buffer + 2 exp LUTs)

module softmax #(
  parameter N        = 256,
  parameter IN_W     = 16,
  parameter FRAC_W   = 7,
  parameter OUT_W    = 16,
  parameter LUT0_HEX = "/home/user/red_eyes_is_all_you_need/mem/exp_lut0.hex",
  parameter LUT1_HEX = "/home/user/red_eyes_is_all_you_need/mem/exp_lut1.hex"
) (
  input  wire              clk_i,
  input  wire              rst_i,
  input  wire              start_i,
  input  wire              in_valid_i,
  input  wire [IN_W-1:0]   in_data_i,
  output wire              in_ready_o,
  output reg               out_valid_o,
  output reg  [OUT_W-1:0]  out_data_o,
  output reg               done_o
);

  localparam ADDR_W = $clog2(N);
  localparam CNT_W  = ADDR_W + 1;  // counter needs to hold value N

  // Q4.7 constants
  localparam [11:0] D_CLIP = 12'd2048;  // 16.0 in Q4.7
  localparam [7:0]  LN2_Q7 = 8'd89;     // round(ln(2) * 128)

  // FSM states
  localparam [2:0] S_IDLE    = 3'd0,
                   S_LOAD    = 3'd1,
                   S_EXP_ACC = 3'd2,
                   S_LN_SUM  = 3'd3,
                   S_NORM    = 3'd4;

  reg [2:0] state;


  // Input buffer BRAM (N x IN_W)
  (* ram_style = "block" *) reg [IN_W-1:0] input_buf [0:N-1];
  reg  [ADDR_W-1:0] buf_waddr;
  reg  [ADDR_W-1:0] buf_raddr;
  reg  [IN_W-1:0]   buf_rdata;
  always @(posedge clk_i) begin
    if (in_valid_i && state == S_LOAD)
      input_buf[buf_waddr] <= in_data_i;
    buf_rdata <= input_buf[buf_raddr];
  end


  // Bipartite exp LUTs (256 x 16-bit each)
  // Addresses driven combinationally from buf_rdata -> d -> bipartite split
  (* ram_style = "block" *) reg [15:0] exp_lut0 [0:255];
  (* ram_style = "block" *) reg signed [15:0] exp_lut1 [0:255];
  initial begin
    $readmemh(LUT0_HEX, exp_lut0);
    $readmemh(LUT1_HEX, exp_lut1);
  end

  reg [15:0]        lut0_out;
  reg signed [15:0] lut1_out;
  always @(posedge clk_i) begin
    lut0_out <= exp_lut0[{x0_sel, x1_sel}];
    lut1_out <= exp_lut1[{x0_sel, x2_sel}];
  end


  // ln(1 + s/16) * 128 lookup (distributed RAM)
  reg [7:0] ln1ps_lut [0:15];
  initial begin
    ln1ps_lut[ 0] = 8'd0;
    ln1ps_lut[ 1] = 8'd8;
    ln1ps_lut[ 2] = 8'd15;
    ln1ps_lut[ 3] = 8'd22;
    ln1ps_lut[ 4] = 8'd29;
    ln1ps_lut[ 5] = 8'd35;
    ln1ps_lut[ 6] = 8'd41;
    ln1ps_lut[ 7] = 8'd47;
    ln1ps_lut[ 8] = 8'd53;
    ln1ps_lut[ 9] = 8'd58;
    ln1ps_lut[10] = 8'd63;
    ln1ps_lut[11] = 8'd68;
    ln1ps_lut[12] = 8'd73;
    ln1ps_lut[13] = 8'd78;
    ln1ps_lut[14] = 8'd82;
    ln1ps_lut[15] = 8'd87;
  end


  // State registers
  reg [CNT_W-1:0]       cnt;
  reg signed [IN_W-1:0] max_val;
  reg [23:0]            sum_acc;
  reg [10:0]            ln_offset;

  // Pipeline: 2 valid stages (matching 2 BRAM registered read stages)
  reg p0_valid, p1_valid;
  reg p0_last,  p1_last;
  reg clip_r;  // d_clipped captured at same posedge as LUT BRAM read

  assign in_ready_o = (state == S_LOAD);


  // Combinational difference & bipartite addressing
  // These wires are valid whenever buf_rdata is valid (after its
  // registered read). They drive the LUT BRAM addresses directly.
  wire signed [IN_W-1:0] in_signed = in_data_i;
  wire signed [IN_W-1:0] buf_rdata_signed = buf_rdata;
  wire [IN_W-1:0] d_raw = max_val - buf_rdata_signed;

  wire [IN_W-FRAC_W-1:0] d_int  = d_raw[IN_W-1:FRAC_W];
  wire [FRAC_W-1:0]      d_frac = d_raw[FRAC_W-1:0];
  wire                    d_overflow = (d_int >= (1 << 4));
  wire [10:0]             d_q47 = d_overflow ? D_CLIP : {d_int[3:0], d_frac};

  wire [11:0] d_plus_ln  = {1'b0, d_q47} + {1'b0, ln_offset};
  wire [10:0] d_norm_q47 = (d_plus_ln >= {1'b0, D_CLIP}) ? D_CLIP : d_plus_ln[10:0];

  wire [10:0] d_sel     = (state == S_NORM) ? d_norm_q47 : d_q47;
  wire        d_clipped = (state == S_NORM) ?
                          (d_overflow || d_plus_ln >= {1'b0, D_CLIP}) : d_overflow;

  wire [4:0] x0_sel = d_sel[10:6];
  wire [2:0] x1_sel = d_sel[5:3];
  wire [2:0] x2_sel = d_sel[2:0];


  // exp_val (valid when p1_valid, uses clip_r and LUT outputs)
  wire signed [16:0] exp_sum_raw = {1'b0, lut0_out} + {{1{lut1_out[15]}}, lut1_out};
  wire [15:0] exp_val = clip_r ? 16'd0 :
                         (exp_sum_raw[16] ? 16'd0 :
                          (exp_sum_raw[15:0] > 16'd32768 ? 16'd32768 :
                           exp_sum_raw[15:0]));


  // LOD for sum_acc (combinational)
  reg [4:0] sum_lod;
  reg [3:0] sum_mantissa;
  reg       sum_is_zero;
  integer i;
  always @(*) begin
    sum_lod = 5'd0;
    sum_is_zero = (sum_acc == 24'd0);
    for (i = 0; i < 24; i = i + 1) begin
      if (sum_acc[i]) sum_lod = i[4:0];
    end
    if (sum_lod >= 5'd4)
      sum_mantissa = sum_acc[sum_lod-1 -: 4];
    else
      sum_mantissa = (sum_acc << (4 - sum_lod)) & 4'hF;
  end


  // LN_SUM (combinational from sum_acc)
  wire signed [5:0]  k_minus_15 = {1'b0, sum_lod} - 6'sd15;
  wire signed [12:0] kln2_term  = k_minus_15 * $signed({1'b0, LN2_Q7});
  wire [7:0]         ln1ps_val  = ln1ps_lut[sum_mantissa];
  wire signed [12:0] ln_raw     = kln2_term + $signed({1'b0, ln1ps_val});


  // FSM
  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      out_valid_o <= 1'b0;
      done_o      <= 1'b0;
      p0_valid    <= 1'b0;
      p1_valid    <= 1'b0;
    end else begin
      out_valid_o <= 1'b0;
      done_o      <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state     <= S_LOAD;
            cnt       <= {CNT_W{1'b0}};
            max_val   <= {1'b1, {(IN_W-1){1'b0}}};
            buf_waddr <= {ADDR_W{1'b0}};
          end
        end


        S_LOAD: begin
          if (in_valid_i) begin
            buf_waddr <= buf_waddr + 1'b1;
            if (in_signed > max_val)
              max_val <= in_signed;
            if (cnt == N - 1) begin
              state     <= S_EXP_ACC;
              cnt       <= {CNT_W{1'b0}};
              buf_raddr <= {ADDR_W{1'b0}};
              sum_acc   <= 24'd0;
              p0_valid  <= 1'b0;
              p1_valid  <= 1'b0;
            end else begin
              cnt <= cnt + 1'b1;
            end
          end
        end


        // EXP_ACC pipeline:
        //   Each cycle issues buf_raddr for the next element.
        //   The BRAM read from the PREVIOUS cycle's buf_raddr produces
        //   buf_rdata, which drives d/LUT-addr combinationally.
        //   The LUT BRAM captures these addresses this posedge.
        //   clip_r also captures d_clipped this posedge.
        //   p1_valid gates accumulation: when true, lut0_out/clip_r
        //   from the previous posedge are valid for the element.
        S_EXP_ACC: begin
          // Stage 0: issue BRAM read for element cnt
          // buf_raddr was set to 0 during LOAD->EXP_ACC transition,
          // so the BRAM already has element 0's read in flight.
          // First cycle: cnt=0, we set buf_raddr=1, p0_valid=1 for element 0.
          if (cnt < N) begin
            buf_raddr <= (cnt[ADDR_W-1:0] == {ADDR_W{1'b1}}) ?
                          {ADDR_W{1'b0}} : cnt[ADDR_W-1:0] + 1'b1;
            cnt       <= cnt + 1'b1;
            p0_valid  <= 1'b1;
            p0_last   <= (cnt == N - 1);
          end else begin
            p0_valid  <= 1'b0;
          end

          // Stage 1: buf_rdata valid from previous buf_raddr.
          // Combinationally: d_clipped and LUT addrs are driven.
          // Only capture clip when buf_rdata is valid (p0_valid).
          p1_valid <= p0_valid;
          p1_last  <= p0_last;
          if (p0_valid)
            clip_r <= d_clipped;

          // Result stage: lut0_out/lut1_out and clip_r from prev posedge
          // are valid. Accumulate exp_val.
          if (p1_valid) begin
            sum_acc <= sum_acc + {8'd0, exp_val};
            if (p1_last) begin
              state <= S_LN_SUM;
              cnt   <= {CNT_W{1'b0}};
            end
          end
        end


        S_LN_SUM: begin
          if (sum_is_zero) begin
            ln_offset <= D_CLIP;
          end else if (ln_raw < 13'sd0) begin
            ln_offset <= 11'd0;
          end else if (ln_raw >= $signed({2'b0, D_CLIP})) begin
            ln_offset <= D_CLIP;
          end else begin
            ln_offset <= ln_raw[10:0];
          end
          state     <= S_NORM;
          cnt       <= {CNT_W{1'b0}};
          buf_raddr <= {ADDR_W{1'b0}};
          p0_valid  <= 1'b0;
          p1_valid  <= 1'b0;
        end


        // NORM: same pipeline, output instead of accumulate
        S_NORM: begin
          if (cnt < N) begin
            buf_raddr <= (cnt[ADDR_W-1:0] == {ADDR_W{1'b1}}) ?
                          {ADDR_W{1'b0}} : cnt[ADDR_W-1:0] + 1'b1;
            cnt       <= cnt + 1'b1;
            p0_valid  <= 1'b1;
            p0_last   <= (cnt == N - 1);
          end else begin
            p0_valid  <= 1'b0;
          end

          p1_valid <= p0_valid;
          p1_last  <= p0_last;
          if (p0_valid)
            clip_r <= d_clipped;

          if (p1_valid) begin
            out_valid_o <= 1'b1;
            out_data_o  <= exp_val;
            if (p1_last) begin
              done_o <= 1'b1;
              state  <= S_IDLE;
            end
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
