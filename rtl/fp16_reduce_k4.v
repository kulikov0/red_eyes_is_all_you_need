// FP16 K=4 interleaved partial-sum reducer with balanced tree reduction
//
// Streams an fp16 sequence in, returns the running sum after flush.
// 4 partial accumulators round-robin'd through one pipelined fp16_add.
// Tree-reduces s01=p0+p1 and s23=p2+p3, then sum=s01+s23.
//
// Caller protocol: pulse clear_i to start a new reduction, drive valid_i +
// data_i for each input with gaps OK, pulse flush_i on the same cycle as
// the last valid_i, wait for done_o then sample sum_o.

module fp16_reduce_k4 (
  input  wire        clk_i,
  input  wire        rst_i,
  input  wire        clear_i,
  input  wire        valid_i,
  input  wire [15:0] data_i,
  input  wire        flush_i,
  output reg         done_o,
  output reg  [15:0] sum_o
);

  localparam [2:0] S_IDLE  = 3'd0,
                   S_ACCUM = 3'd1,
                   S_DRAIN = 3'd2,
                   S_RED   = 3'd3;

  reg [2:0]  state;
  reg [1:0]  rr_cnt;
  reg [1:0]  slot_pipe [0:2];
  reg [3:0]  drain_cnt;
  reg [3:0]  red_cnt;

  reg [15:0] partials [0:3];
  reg [15:0] s01_r;
  reg [15:0] s23_r;

  // Add input mux: round-robin during S_ACCUM, tree reduce during S_RED
  reg [15:0] add_a;
  reg [15:0] add_b;
  reg        add_v_in;
  always @(*) begin
    add_a    = 16'd0;
    add_b    = 16'd0;
    add_v_in = 1'b0;
    case (state)
      S_ACCUM: begin
        add_a    = partials[rr_cnt];
        add_b    = data_i;
        add_v_in = valid_i;
      end
      S_RED: begin
        case (red_cnt)
          4'd0: begin add_a = partials[0]; add_b = partials[1]; add_v_in = 1'b1; end
          4'd1: begin add_a = partials[2]; add_b = partials[3]; add_v_in = 1'b1; end
          4'd5: begin add_a = s01_r;       add_b = s23_r;       add_v_in = 1'b1; end
          default: ;
        endcase
      end
      default: ;
    endcase
  end

  wire        add_v_out;
  wire [15:0] add_sum;
  fp16_add u_add (
    .clk_i  (clk_i),
    .valid_i(add_v_in),
    .a_i    (add_a),
    .b_i    (add_b),
    .valid_o(add_v_out),
    .sum_o  (add_sum)
  );

  integer i;
  always @(posedge clk_i) begin
    if (rst_i) begin
      state     <= S_IDLE;
      rr_cnt    <= 2'd0;
      drain_cnt <= 4'd0;
      red_cnt   <= 4'd0;
      done_o    <= 1'b0;
      sum_o     <= 16'd0;
      for (i = 0; i < 4; i = i + 1) partials[i] <= 16'd0;
      for (i = 0; i < 3; i = i + 1) slot_pipe[i] <= 2'd0;
    end else begin
      done_o <= 1'b0;

      slot_pipe[0] <= rr_cnt;
      for (i = 1; i < 3; i = i + 1) slot_pipe[i] <= slot_pipe[i-1];

      // Writeback to partial slot during accumulation/drain
      if ((state == S_ACCUM || state == S_DRAIN) && add_v_out) begin
        partials[slot_pipe[2]] <= add_sum;
      end

      case (state)
        S_IDLE: begin
          if (clear_i) begin
            state  <= S_ACCUM;
            rr_cnt <= 2'd0;
            for (i = 0; i < 4; i = i + 1) partials[i] <= 16'd0;
          end
        end

        S_ACCUM: begin
          if (valid_i) rr_cnt <= rr_cnt + 2'd1;
          if (flush_i) begin
            state     <= S_DRAIN;
            drain_cnt <= 4'd0;
          end
        end

        // Wait for in-flight adds to settle into partials
        S_DRAIN: begin
          if (drain_cnt == 4'd4) begin
            state   <= S_RED;
            red_cnt <= 4'd0;
          end else begin
            drain_cnt <= drain_cnt + 4'd1;
          end
        end

        // Balanced tree: feed (p0+p1) then (p2+p3), capture both intermediates,
        // feed s01+s23, capture final result and signal done
        S_RED: begin
          red_cnt <= red_cnt + 4'd1;
          if (red_cnt == 4'd3) s01_r <= add_sum;
          if (red_cnt == 4'd4) s23_r <= add_sum;
          if (red_cnt == 4'd8) begin
            sum_o  <= add_sum;
            done_o <= 1'b1;
            state  <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
