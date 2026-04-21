// UART receiver, 8N1
//
// Oversamples at 16x baud rate, samples at mid-bit
// FSM: IDLE -> START (wait half bit) -> DATA (8 bits) -> STOP -> IDLE
// valid_o pulses for 1 cycle when a byte is received

module uart_rx #(
  parameter CLK_FREQ = 100_000_000,
  parameter BAUD     = 115_200
) (
  input  wire       clk_i,
  input  wire       rst_i,
  input  wire       rx_i,
  output reg  [7:0] data_o,
  output reg        valid_o
);

  localparam CLKS_PER_BIT = CLK_FREQ / BAUD;
  localparam HALF_BIT     = CLKS_PER_BIT / 2;
  localparam CNT_W        = $clog2(CLKS_PER_BIT);

  localparam [1:0] S_IDLE  = 2'd0,
                   S_START = 2'd1,
                   S_DATA  = 2'd2,
                   S_STOP  = 2'd3;

  reg [1:0]       state;
  reg [CNT_W-1:0] cnt;
  reg [2:0]       bit_idx;
  reg [7:0]       shift;

  // Double-flop synchronizer
  reg rx_r1, rx_r2;
  always @(posedge clk_i) begin
    rx_r1 <= rx_i;
    rx_r2 <= rx_r1;
  end

  always @(posedge clk_i) begin
    if (rst_i) begin
      state   <= S_IDLE;
      valid_o <= 1'b0;
      data_o  <= 8'd0;
      cnt     <= 0;
      bit_idx <= 3'd0;
      shift   <= 8'd0;
    end else begin
      valid_o <= 1'b0;

      case (state)

        S_IDLE: begin
          if (~rx_r2) begin
            state <= S_START;
            cnt   <= 0;
          end
        end

        S_START: begin
          if (cnt == HALF_BIT[CNT_W-1:0] - 1) begin
            if (~rx_r2) begin
              state   <= S_DATA;
              cnt     <= 0;
              bit_idx <= 3'd0;
            end else begin
              state <= S_IDLE;
            end
          end else begin
            cnt <= cnt + 1;
          end
        end

        S_DATA: begin
          if (cnt == CLKS_PER_BIT[CNT_W-1:0] - 1) begin
            cnt   <= 0;
            shift <= {rx_r2, shift[7:1]};
            if (bit_idx == 3'd7) begin
              state <= S_STOP;
            end else begin
              bit_idx <= bit_idx + 3'd1;
            end
          end else begin
            cnt <= cnt + 1;
          end
        end

        S_STOP: begin
          if (cnt == CLKS_PER_BIT[CNT_W-1:0] - 1) begin
            state   <= S_IDLE;
            data_o  <= shift;
            valid_o <= 1'b1;
          end else begin
            cnt <= cnt + 1;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule