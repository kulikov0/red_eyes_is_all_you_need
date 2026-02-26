`timescale 1ns/1ps
module matvec_int8 #(
    parameter IN_DIM  = 128,
    parameter OUT_DIM = 128
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,
    input  wire [IN_DIM*8-1:0]              in_vec,
    output reg  [$clog2(OUT_DIM*IN_DIM)-1:0] weight_addr,
    input  wire signed [7:0]                 weight_data,
    output reg  [OUT_DIM*8-1:0]             out_vec,
    output reg                              done
);

    reg signed [23:0] acc;
    reg [$clog2(IN_DIM):0]  col;
    reg [$clog2(OUT_DIM):0] row;
    reg running;

    always @(posedge clk) begin

        if (rst) begin
            acc         <= 0;
            col         <= 0;
            row         <= 0;
            running     <= 0;
            done        <= 0;
            weight_addr <= 0;

        end else if (start) begin
            acc         <= 0;
            col         <= 0;
            row         <= 0;
            running     <= 1;   // start immediately, no prefetch needed
            done        <= 0;
            weight_addr <= 0;

        end else if (running) begin

            if (col == IN_DIM - 1) begin
                // Last element of row - compute final MAC combinatorially
                // to avoid acc<=acc+... and acc<=0 fighting each other
                begin : requant
                    reg signed [23:0] final_acc;
                    reg signed [16:0] shifted;
                    final_acc = acc + ($signed(in_vec[col*8 +: 8]) * $signed(weight_data));
                    shifted   = $signed(final_acc) >>> 7;
                    out_vec[row*8 +: 8] <=
                        (shifted > 17'sd127)  ?  8'sd127 :
                        (shifted < -17'sd128) ? -8'sd128 :
                         shifted[7:0];
                end

                col         <= 0;
                acc         <= 0;
                weight_addr <= weight_addr + 1;
                row         <= row + 1;

                if (row == OUT_DIM - 1) begin
                    running <= 0;
                    done    <= 1;
                end

            end else begin
                acc         <= acc + ($signed(in_vec[col*8 +: 8]) * $signed(weight_data));
                col         <= col + 1;
                weight_addr <= weight_addr + 1;
            end

        end else begin
            done <= 0;
        end
    end

endmodule