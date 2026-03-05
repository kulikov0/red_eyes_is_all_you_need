// Embedding lookup: tok_emb[token_id] * tok_scale + pos_emb[position] * pos_scale
//
// Reads 128 bytes of tok_emb from weight_store, then 128 bytes of pos_emb,
// dequantizes each int8 to fp16 (* scale), adds element-wise
//
// FSM: IDLE -> READ_TOK(129 cyc) -> READ_POS(129 cyc) -> DONE
// Latency: 258 cycles
//
// TODO - optimize: add second data output to weight_store so tok_emb and
// pos_emb can be read in parallel on the same addr, halving latency to ~130

module embedding #(
  parameter DIM = 128
) (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        start_i,
  input  wire [7:0]  token_id_i,
  input  wire [7:0]  position_i,

  // FP16 dequant scales
  input  wire [15:0] tok_scale_i,
  input  wire [15:0] pos_scale_i,

  // Weight store interface
  output reg  [5:0]  w_sel_o,
  output reg  [15:0] w_addr_o,
  input  wire [7:0]  w_data_i,

  // Output: 128 x fp16
  output reg  [DIM*16-1:0] embed_o,
  output reg               done_o,
  output reg               busy_o
);

  localparam S_IDLE     = 2'd0;
  localparam S_READ_TOK = 2'd1;
  localparam S_READ_POS = 2'd2;

  reg [1:0] state;
  reg [7:0] idx;
  wire [7:0] prev = idx - 8'd1;

  // Base addresses: token_id * 128, position * 128
  wire [14:0] tok_base = {token_id_i[7:0], 7'd0};
  wire [14:0] pos_base = {position_i[7:0], 7'd0};

  // Buffer for tok_emb values (read in first pass, used in second)
  reg signed [7:0] tok_buf [0:DIM-1];

  // Dequant pipeline (all combinational):
  // int8 -> fp16 -> * scale -> fp16_add

  // tok: int8 -> fp16
  wire [15:0] tok_fp16;
  fp16_from_int8 u_tok_cvt (.val_i(tok_buf[prev[6:0]]), .fp16_o(tok_fp16));

  // tok: fp16 * tok_scale
  wire [15:0] tok_dequant;
  fp16_mul_comb u_tok_mul (.a_i(tok_fp16), .b_i(tok_scale_i), .prod_o(tok_dequant));

  // pos: int8 -> fp16
  wire [15:0] pos_fp16;
  fp16_from_int8 u_pos_cvt (.val_i(w_data_i), .fp16_o(pos_fp16));

  // pos: fp16 * pos_scale
  wire [15:0] pos_dequant;
  fp16_mul_comb u_pos_mul (.a_i(pos_fp16), .b_i(pos_scale_i), .prod_o(pos_dequant));

  // tok_dequant + pos_dequant
  wire [15:0] sum_fp16;
  fp16_add_comb u_add (.a_i(tok_dequant), .b_i(pos_dequant), .sum_o(sum_fp16));

  always @(posedge clk_i) begin
    if (rst_i) begin
      state    <= S_IDLE;
      idx      <= 8'd0;
      done_o   <= 1'b0;
      busy_o   <= 1'b0;
      w_sel_o  <= 6'd0;
      w_addr_o <= 16'd0;

    end else begin
      done_o <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start_i) begin
            state    <= S_READ_TOK;
            idx      <= 8'd0;
            busy_o   <= 1'b1;
            // Pre-issue first tok_emb address
            w_sel_o  <= 6'd0;
            w_addr_o <= {1'b0, tok_base};
          end
        end

        // Read 128 bytes of tok_emb[token_id] into tok_buf
        S_READ_TOK: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= {1'b0, tok_base} + {8'd0, idx} + 16'd1;
          end
          if (idx > 0) begin
            tok_buf[prev[6:0]] <= $signed(w_data_i);
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0]) begin
            state    <= S_READ_POS;
            idx      <= 8'd0;
            // Pre-issue first pos_emb address
            w_sel_o  <= 6'd1;
            w_addr_o <= {1'b0, pos_base};
          end
        end

        // Read 128 bytes of pos_emb[position], dequant both, add, store
        S_READ_POS: begin
          if (idx < DIM[7:0] - 8'd1) begin
            w_addr_o <= {1'b0, pos_base} + {8'd0, idx} + 16'd1;
          end
          if (idx > 0) begin
            embed_o[prev[6:0]*16 +: 16] <= sum_fp16;
          end
          idx <= idx + 8'd1;
          if (idx == DIM[7:0]) begin
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