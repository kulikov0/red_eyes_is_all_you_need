// UART transmitter, 8N1
//
// Loads data on start_i pulse, shifts out start + 8 data + stop bits
// FSM: IDLE -> START -> DATA (8 bits) -> STOP -> IDLE
// busy_o high while transmitting

module uart_tx #(
  parameter CLK_FREQ = 100_000_000,
  parameter BAUD     = 115_200
) (
  input  wire       clk_i,
  input  wire       rst_i,
  input  wire [7:0] data_i,
  input  wire       start_i,
  output reg        tx_o,
  output reg        busy_o
);

  localparam CLKS_PER_BIT = CLK_FREQ / BAUD;
  localparam CNT_W        = $clog2(CLKS_PER_BIT);

  localparam [1:0] S_IDLE  = 2'd0,
                   S_START = 2'd1,
                   S_DATA  = 2'd2,
                   S_STOP  = 2'd3;

  reg [1:0]       state;
  reg [CNT_W-1:0] cnt;
  reg [2:0]       bit_idx;
  reg [7:0]       shift;

  always @(posedge clk_i) begin
    if (rst_i) begin
      state   <= S_IDLE;
      tx_o    <= 1'b1;
      busy_o  <= 1'b0;
      cnt     <= 0;
      bit_idx <= 3'd0;
      shift   <= 8'd0;
    end else begin
      case (state)

        S_IDLE: begin
          tx_o <= 1'b1;
          if (start_i) begin
            state  <= S_START;
            shift  <= data_i;
            busy_o <= 1'b1;
            cnt    <= 0;
          end
        end

        S_START: begin
          tx_o <= 1'b0;
          if (cnt == CLKS_PER_BIT[CNT_W-1:0] - 1) begin
            state   <= S_DATA;
            cnt     <= 0;
            bit_idx <= 3'd0;
          end else begin
            cnt <= cnt + 1;
          end
        end

        S_DATA: begin
          tx_o <= shift[0];
          if (cnt == CLKS_PER_BIT[CNT_W-1:0] - 1) begin
            cnt   <= 0;
            shift <= {1'b0, shift[7:1]};
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
          tx_o <= 1'b1;
          if (cnt == CLKS_PER_BIT[CNT_W-1:0] - 1) begin
            state  <= S_IDLE;
            busy_o <= 1'b0;
          end else begin
            cnt <= cnt + 1;
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule