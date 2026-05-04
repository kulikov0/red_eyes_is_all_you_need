// 32-way SIMD matrix-vector multiply: int8 weights packed 16/word x fp16 input.
//
// Reads one 256-bit BRAM word per cycle from a single-port weight bank that
// pre-pairs col and col+IN_DIM/2 weights for the same 16 rows. Low 128 bits
// feed lane group A, high 128 bits feed lane group B. K=8 keeps the slot
// revisit period larger than fp16_add latency so NBA semantics settle in
// time, unlike a K=4 layout which races.

module matvec_fp16_w32 #(
  parameter IN_DIM  = 128,
  parameter OUT_DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,
  input  wire [15:0] scale_i,

  output wire [$clog2((OUT_DIM/16)*(IN_DIM/2))-1:0] weight_addr_o,
  input  wire [255:0]                               weight_data_i,

  output reg [$clog2(IN_DIM)-1:0] act_raddr_o,
  input  wire [15:0]              act_rdata_i,

  output reg                        res_we_o,
  output reg  [$clog2(OUT_DIM)-1:0] res_waddr_o,
  output reg  [15:0]                res_wdata_o,

  output reg  done_o
);

  localparam K        = 8;
  localparam K_LOG    = 3;
  localparam IN_HALF  = IN_DIM / 2;
  localparam COL_W    = $clog2(IN_HALF);
  localparam GRP_N    = OUT_DIM / (16 * K);
  localparam GRP_W    = (GRP_N <= 1) ? 1 : $clog2(GRP_N);
  localparam WRITE_W  = 7;

  (* ram_style = "distributed" *) reg [15:0] in_snap [0:IN_DIM-1];
  reg [15:0] acc [0:15][0:K-1];

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
  reg [WRITE_W-1:0] write_idx;

  assign weight_addr_o = group * (K * IN_HALF) + k * IN_HALF + col;

  reg [255:0] weight_data_r;
  always @(posedge clk_i) weight_data_r <= weight_data_i;

  wire [127:0] weight_data_a_r = weight_data_r[127:0];
  wire [127:0] weight_data_b_r = weight_data_r[255:128];

  reg [2:0] state_compute_r;
  always @(posedge clk_i) begin
    if (rst_i) state_compute_r <= 3'b000;
    else       state_compute_r <= {state_compute_r[1:0], (state == S_COMPUTE)};
  end
  wire dq_valid_in = state_compute_r[2];

  wire [15:0] w_fp16_a    [0:15];
  wire [15:0] w_fp16_b    [0:15];
  wire        dq_valid_a  [0:15];
  wire        dq_valid_b  [0:15];
  wire [15:0] w_dequant_a [0:15];
  wire [15:0] w_dequant_b [0:15];

  genvar L;
  generate
    for (L = 0; L < 16; L = L + 1) begin : g_dq
      fp16_from_int8 u_from_a (.val_i(weight_data_a_r[L*8 +: 8]), .fp16_o(w_fp16_a[L]));
      fp16_from_int8 u_from_b (.val_i(weight_data_b_r[L*8 +: 8]), .fp16_o(w_fp16_b[L]));
      fp16_mul u_dq_a (
        .clk_i  (clk_i), .valid_i(dq_valid_in),
        .a_i    (w_fp16_a[L]), .b_i(scale_i),
        .valid_o(dq_valid_a[L]), .prod_o(w_dequant_a[L])
      );
      fp16_mul u_dq_b (
        .clk_i  (clk_i), .valid_i(dq_valid_in),
        .a_i    (w_fp16_b[L]), .b_i(scale_i),
        .valid_o(dq_valid_b[L]), .prod_o(w_dequant_b[L])
      );
    end
  endgenerate

  reg [COL_W-1:0] col_pipe [0:4];
  reg [K_LOG-1:0] k_pipe   [0:14];
  integer i;
  always @(posedge clk_i) begin
    col_pipe[0] <= col;
    for (i = 1; i < 5; i = i + 1) col_pipe[i] <= col_pipe[i-1];

    k_pipe[0] <= k;
    for (i = 1; i < 15; i = i + 1) k_pipe[i] <= k_pipe[i-1];
  end

  wire        mac_valid_a [0:15];
  wire        mac_valid_b [0:15];
  wire [15:0] mac_prod_a  [0:15];
  wire [15:0] mac_prod_b  [0:15];
  wire [$clog2(IN_DIM)-1:0] col_b_pipe = {1'b1, col_pipe[4]};
  generate
    for (L = 0; L < 16; L = L + 1) begin : g_mac
      fp16_mul u_mac_a (
        .clk_i  (clk_i), .valid_i(dq_valid_a[L]),
        .a_i    (w_dequant_a[L]), .b_i(in_snap[col_pipe[4]]),
        .valid_o(mac_valid_a[L]), .prod_o(mac_prod_a[L])
      );
      fp16_mul u_mac_b (
        .clk_i  (clk_i), .valid_i(dq_valid_b[L]),
        .a_i    (w_dequant_b[L]), .b_i(in_snap[col_b_pipe]),
        .valid_o(mac_valid_b[L]), .prod_o(mac_prod_b[L])
      );
    end
  endgenerate

  wire        sum_ab_valid [0:15];
  wire [15:0] sum_ab       [0:15];
  generate
    for (L = 0; L < 16; L = L + 1) begin : g_sum_ab
      fp16_add u_sum_ab (
        .clk_i  (clk_i), .valid_i(mac_valid_a[L]),
        .a_i    (mac_prod_a[L]), .b_i(mac_prod_b[L]),
        .valid_o(sum_ab_valid[L]), .sum_o(sum_ab[L])
      );
    end
  endgenerate

  wire        add_valid [0:15];
  wire [15:0] add_sum   [0:15];
  generate
    for (L = 0; L < 16; L = L + 1) begin : g_add
      fp16_add u_add (
        .clk_i  (clk_i), .valid_i(sum_ab_valid[L]),
        .a_i    (acc[L][k_pipe[10]]), .b_i(sum_ab[L]),
        .valid_o(add_valid[L]), .sum_o(add_sum[L])
      );
    end
  endgenerate

  integer j;
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
      write_idx   <= 0;

    end else begin
      done_o   <= 1'b0;
      res_we_o <= 1'b0;

      for (j = 0; j < 16; j = j + 1) begin
        if (add_valid[j]) acc[j][k_pipe[14]] <= add_sum[j];
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
          for (i = 0; i < K; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
              acc[j][i] <= 16'd0;
            end
          end
          col   <= 0;
          k     <= 0;
          state <= S_COMPUTE;
        end

        S_COMPUTE: begin
          if (k == K - 1) begin
            k <= 0;
            if (col == IN_HALF - 1) begin
              state     <= S_DRAIN;
              drain_cnt <= 0;
            end else begin
              col <= col + 1;
            end
          end else begin
            k <= k + 1;
          end
        end

        S_DRAIN: begin
          if (drain_cnt == 4'd14) begin
            state     <= S_WRITE;
            write_idx <= 0;
          end else begin
            drain_cnt <= drain_cnt + 1;
          end
        end

        S_WRITE: begin
          res_we_o    <= 1'b1;
          res_waddr_o <= group * (16 * K) + write_idx;
          res_wdata_o <= acc[write_idx[3:0]][write_idx[6:4]];
          if (write_idx == (16 * K - 1)) begin
            if (group == GRP_N - 1) begin
              state <= S_DONE;
            end else begin
              group <= group + 1;
              state <= S_ZERO;
            end
          end else begin
            write_idx <= write_idx + 1;
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
