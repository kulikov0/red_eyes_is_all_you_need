// 8-way SIMD matrix-vector multiply: int8 weights packed 8/word x fp16 input
//
// weight_data_i is a 64-bit BRAM word holding 8 adjacent rows of the matrix
// at the same column. Byte L carries row[group*32 + slot*8 + L][col]. Eight
// parallel lanes process the bytes in lockstep; lane L accumulates rows whose
// low 3 bits equal L within the current group. All lanes share the activation
// read and the K=4 row-octet interleaving that covers fp16_add feedback
//
// Throughput is 8 weights per BRAM read. Writeback emits 32 results per group
// of 32 rows
//
// Assumes OUT_DIM % 32 == 0 and IN_DIM is a power of 2

module matvec_fp16_w8 #(
  parameter IN_DIM  = 128,
  parameter OUT_DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,
  input  wire [15:0] scale_i,

  output wire [$clog2((OUT_DIM/8)*IN_DIM)-1:0] weight_addr_o,
  input  wire [63:0]                           weight_data_i,

  output reg [$clog2(IN_DIM)-1:0] act_raddr_o,
  input  wire [15:0]              act_rdata_i,

  output reg                        res_we_o,
  output reg  [$clog2(OUT_DIM)-1:0] res_waddr_o,
  output reg  [15:0]                res_wdata_o,

  output reg  done_o
);

  localparam K       = 4;
  localparam K_LOG   = 2;
  localparam COL_W   = $clog2(IN_DIM);
  localparam GRP_N   = OUT_DIM / (8 * K);
  localparam GRP_W   = (GRP_N <= 1) ? 1 : $clog2(GRP_N);
  localparam WRITE_W = 5;

  reg [15:0] in_snap [0:IN_DIM-1];
  reg [15:0] acc [0:7][0:K-1];

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

  assign weight_addr_o = group * (K * IN_DIM) + k * IN_DIM + col;

  // Boundary register on weight_data_i breaks the long route from weight_store_w8
  reg [63:0] weight_data_r;
  always @(posedge clk_i) weight_data_r <= weight_data_i;

  // Dequant valid delays state==S_COMPUTE to align with weight_store latency
  // plus the local weight_data_r register
  reg [2:0] state_compute_r;
  always @(posedge clk_i) begin
    if (rst_i) state_compute_r <= 3'b000;
    else       state_compute_r <= {state_compute_r[1:0], (state == S_COMPUTE)};
  end
  wire dq_valid_in = state_compute_r[2];

  wire [15:0] w_fp16    [0:7];
  wire        dq_valid  [0:7];
  wire [15:0] w_dequant [0:7];

  genvar L;
  generate
    for (L = 0; L < 8; L = L + 1) begin : g_dq
      fp16_from_int8 u_from (
        .val_i (weight_data_r[L*8 +: 8]),
        .fp16_o(w_fp16[L])
      );
      fp16_mul u_dq (
        .clk_i  (clk_i),
        .valid_i(dq_valid_in),
        .a_i    (w_fp16[L]),
        .b_i    (scale_i),
        .valid_o(dq_valid[L]),
        .prod_o (w_dequant[L])
      );
    end
  endgenerate

  reg [COL_W-1:0] col_pipe [0:4];
  reg [K_LOG-1:0] k_pipe   [0:9];
  integer i;
  always @(posedge clk_i) begin
    col_pipe[0] <= col;
    for (i = 1; i < 5; i = i + 1) col_pipe[i] <= col_pipe[i-1];

    k_pipe[0] <= k;
    for (i = 1; i < 10; i = i + 1) k_pipe[i] <= k_pipe[i-1];
  end

  wire        mac_valid [0:7];
  wire [15:0] mac_prod  [0:7];
  generate
    for (L = 0; L < 8; L = L + 1) begin : g_mac
      fp16_mul u_mac (
        .clk_i  (clk_i),
        .valid_i(dq_valid[L]),
        .a_i    (w_dequant[L]),
        .b_i    (in_snap[col_pipe[4]]),
        .valid_o(mac_valid[L]),
        .prod_o (mac_prod[L])
      );
    end
  endgenerate

  wire        add_valid [0:7];
  wire [15:0] add_sum   [0:7];
  generate
    for (L = 0; L < 8; L = L + 1) begin : g_add
      fp16_add u_add (
        .clk_i  (clk_i),
        .valid_i(mac_valid[L]),
        .a_i    (acc[L][k_pipe[6]]),
        .b_i    (mac_prod[L]),
        .valid_o(add_valid[L]),
        .sum_o  (add_sum[L])
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

      for (j = 0; j < 8; j = j + 1) begin
        if (add_valid[j]) acc[j][k_pipe[9]] <= add_sum[j];
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
            for (j = 0; j < 8; j = j + 1) begin
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

        S_DRAIN: begin
          if (drain_cnt == 4'd9) begin
            state     <= S_WRITE;
            write_idx <= 0;
          end else begin
            drain_cnt <= drain_cnt + 1;
          end
        end

        // write_idx[2:0]=lane, write_idx[4:3]=k slot.
        // Row index = group*32 + write_idx
        S_WRITE: begin
          res_we_o    <= 1'b1;
          res_waddr_o <= group * (8 * K) + write_idx;
          res_wdata_o <= acc[write_idx[2:0]][write_idx[4:3]];
          if (write_idx == (8 * K - 1)) begin
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
