module matvec_int8 #(
  parameter IN_DIM  = 128,
  parameter OUT_DIM = 128
) (
  input  wire                             clk_i,
  input  wire                             rst_i,
  input  wire                             start_i,
  input  wire [IN_DIM*8-1:0]              in_vec_i,
  output reg [$clog2(OUT_DIM*IN_DIM)-1:0] weight_addr_o,
  input  wire signed [7:0]                weight_data_i,
  output reg [OUT_DIM*8-1:0]              out_vec_o,
  output reg                              done_o
);

  reg signed [23:0] acc;
  reg [$clog2(IN_DIM):0]  col;
  reg [$clog2(OUT_DIM):0] row;
  reg running;
  reg prefetch;

  always @(posedge clk_i) begin

    if (rst_i) begin
      acc           <= 24'd0;
      col           <= {($clog2(IN_DIM)+1){1'b0}};
      row           <= {($clog2(OUT_DIM)+1){1'b0}};
      running       <= 1'b0;
      prefetch      <= 1'b0;
      done_o        <= 1'b0;
      weight_addr_o <= {$clog2(OUT_DIM*IN_DIM){1'b0}};

    end else if (start_i) begin
      acc           <= 24'd0;
      col           <= {($clog2(IN_DIM)+1){1'b0}};
      row           <= {($clog2(OUT_DIM)+1){1'b0}};
      running       <= 1'b0;
      prefetch      <= 1'b1;   // wait 1 cycle for BRAM read latency
      done_o        <= 1'b0;
      weight_addr_o <= {$clog2(OUT_DIM*IN_DIM){1'b0}};

    end else if (prefetch) begin
      prefetch      <= 1'b0;
      running       <= 1'b1;
      weight_addr_o <= weight_addr_o + 1;

    end else if (running) begin

      if (col == IN_DIM - 1) begin
        // Last element of row - compute final MAC combinatorially
        // to avoid acc<=acc+... and acc<=0 fighting each other
        begin : requant
          reg signed [23:0] final_acc;
          reg signed [16:0] shifted;
          final_acc = acc + ($signed(in_vec_i[col*8 +: 8]) * $signed(weight_data_i));
          shifted   = $signed(final_acc) >>> 7;
          out_vec_o[row*8 +: 8] <=
            (shifted > 17'sd127)  ?  8'sd127 :
            (shifted < -17'sd128) ? -8'sd128 :
             shifted[7:0];
        end

        col           <= {($clog2(IN_DIM)+1){1'b0}};
        acc           <= 24'd0;
        weight_addr_o <= weight_addr_o + 1;
        row           <= row + 1;

        if (row == OUT_DIM - 1) begin
          running <= 1'b0;
          done_o  <= 1'b1;
        end

      end else begin
        acc           <= acc + ($signed(in_vec_i[col*8 +: 8]) * $signed(weight_data_i));
        col           <= col + 1;
        weight_addr_o <= weight_addr_o + 1;
      end

    end else begin
      done_o <= 1'b0;
    end
  end

endmodule