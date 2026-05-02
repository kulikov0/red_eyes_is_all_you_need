// Embedding lookup: tok_emb[token_id] * tok_scale + pos_emb[position] * pos_scale.
//
// tok_emb lives in weight_store_tok_emb packed. Each 128-bit word holds rows
// 16g..16g+15 at the same column, so embedding byte-extracts via token_id[3:0].
// pos_emb lives in weight_store as bytes.

module embedding #(
  parameter DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,
  input  wire [7:0]  token_id_i,
  input  wire [7:0]  position_i,

  input  wire [15:0] w_scale_i,

  // Byte weight store interface for pos_emb
  output reg  [5:0]  w_sel_o,
  output reg  [15:0] w_addr_o,
  input  wire [7:0]  w_data_i,

  // 128-bit packed tok_emb bus from weight_store_tok_emb
  output reg  [10:0]  tok_addr_o,
  input  wire [127:0] tok_data_i,
  input  wire [15:0]  tok_scale_i,

  // Result write to shared RAM
  output reg                     res_we_o,
  output reg  [$clog2(DIM)-1:0]  res_waddr_o,
  output reg  [15:0]             res_wdata_o,

  output reg               done_o,
  output reg               busy_o
);

  localparam S_IDLE     = 2'd0;
  localparam S_READ_TOK = 2'd1;
  localparam S_READ_POS = 2'd2;
  localparam S_DONE     = 2'd3;

  reg [1:0] state;
  reg [7:0] idx;

  // pos_emb base address: position * 128. tok_emb packed addr stride is
  // {token_id[7:4], dim[6:0]} since each 128-bit word covers 16 rows at one col
  wire [14:0] pos_base = {position_i[7:0], 7'd0};
  wire [10:0] tok_base = {token_id_i[7:4], 7'd0};

  // Buffer for tok_emb values read in first pass, used in second
  reg signed [7:0] tok_buf [0:DIM-1];

  reg [15:0] tok_scale_r;

  // Boundary register on w_data_i breaks the long route from weight_store
  // BRAM through the tensor_sel mux into our DSP
  reg signed [7:0] w_data_r;
  always @(posedge clk_i) w_data_r <= w_data_i;

  // Boundary register on tok_data_i, byte-extract by token_id[3:0]
  reg [127:0] tok_data_r;
  always @(posedge clk_i) tok_data_r <= tok_data_i;
  wire signed [7:0] tok_byte = tok_data_r[token_id_i[3:0]*8 +: 8];

  wire [7:0] prev3 = idx - 8'd3;

  // Combinational int8 -> fp16 conversion
  wire [15:0] tok_fp16;
  fp16_from_int8 u_tok_cvt (.val_i(tok_buf[prev3[6:0]]), .fp16_o(tok_fp16));

  wire [15:0] pos_fp16;
  fp16_from_int8 u_pos_cvt (.val_i(w_data_r), .fp16_o(pos_fp16));

  // Feed pipeline only while consuming pos reads in S_READ_POS, shifted to
  // wait for w_data_r to settle behind the weight_store boundary register
  wire feed_valid = (state == S_READ_POS) && (idx >= 8'd3) && (idx <= DIM[7:0] + 8'd2);

  // tok dequant: fp16 * tok_scale
  wire        tok_mv_out;
  wire [15:0] tok_dq;
  fp16_mul u_tok_mul (
    .clk_i(clk_i),
    .valid_i(feed_valid),
    .a_i(tok_fp16),
    .b_i(tok_scale_r),
    .valid_o(tok_mv_out),
    .prod_o(tok_dq)
  );

  // pos dequant: fp16 * pos_scale
  wire        pos_mv_out;
  wire [15:0] pos_dq;
  fp16_mul u_pos_mul (
    .clk_i(clk_i),
    .valid_i(feed_valid),
    .a_i(pos_fp16),
    .b_i(w_scale_i),
    .valid_o(pos_mv_out),
    .prod_o(pos_dq)
  );

  // Sum: tok_dq + pos_dq
  wire        add_v_out;
  wire [15:0] sum_fp16;
  fp16_add u_add (
    .clk_i(clk_i),
    .valid_i(tok_mv_out),
    .a_i(tok_dq),
    .b_i(pos_dq),
    .valid_o(add_v_out),
    .sum_o(sum_fp16)
  );

  // Track write address through the dequant pipeline
  reg [$clog2(DIM)-1:0] feed_addr_pipe [0:5];
  integer i;
  always @(posedge clk_i) begin
    feed_addr_pipe[0] <= prev3[$clog2(DIM)-1:0];
    for (i = 1; i < 6; i = i + 1) feed_addr_pipe[i] <= feed_addr_pipe[i-1];
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      state       <= S_IDLE;
      idx         <= 8'd0;
      done_o      <= 1'b0;
      busy_o      <= 1'b0;
      res_we_o    <= 1'b0;
      w_sel_o     <= 6'd0;
      w_addr_o    <= 16'd0;
      tok_addr_o  <= 11'd0;
      tok_scale_r <= 16'd0;

    end else begin
      done_o   <= 1'b0;
      res_we_o <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state      <= S_READ_TOK;
            idx        <= 8'd0;
            busy_o     <= 1'b1;
            // Pre-issue first tok_emb packed address
            tok_addr_o <= tok_base;
          end
        end

        // Read 128 packed words at {token_id[7:3], dim} and byte-extract into tok_buf
        S_READ_TOK: begin
          if (idx == 8'd2)
            tok_scale_r <= tok_scale_i;
          if (idx < DIM[7:0] - 8'd1) begin
            tok_addr_o <= tok_base + {3'd0, idx} + 11'd1;
          end
          if (idx > 2) begin
            tok_buf[prev3[6:0]] <= tok_byte;
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0] + 8'd2) begin
            state    <= S_READ_POS;
            idx      <= 8'd0;
            w_sel_o  <= 6'd1;
            w_addr_o <= {1'b0, pos_base};
          end
        end

        // Feed fp16 dequant pipeline, drain afterwards
        S_READ_POS: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= {1'b0, pos_base} + {8'd0, idx} + 16'd1;
          end

          if (add_v_out) begin
            res_we_o    <= 1'b1;
            res_waddr_o <= feed_addr_pipe[5];
            res_wdata_o <= sum_fp16;
          end

          idx <= idx + 8'd1;
          if (idx == DIM[7:0] + 8'd8) begin
            state <= S_DONE;
          end
        end

        S_DONE: begin
          done_o <= 1'b1;
          busy_o <= 1'b0;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule