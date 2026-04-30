// System top for ALINX AX7203 (XC7A200TFBG484-2)
//
// 200 MHz LVDS clock -> MMCM -> 100 MHz working clock
// UART (115200 8N1) <-> transformer_top
//
// Protocol:
//   PC sends 1 byte -> FPGA runs prompt pass (KV fill) for each byte
//   PC sends 0xFF   -> FPGA runs autoregressive generation, sends each
//                      token back over UART. Stops at pos=255 or next 0xFF
//
// LEDs (active-low accent):
//   led[0] = busy (generating)
//   led[1] = UART RX activity (blinks on receive)
//   led[2] = UART TX activity (blinks on transmit)
//   led[3] = heartbeat (toggles every ~0.5s)

module top (
  input  wire       sys_clk_p_i,
  input  wire       sys_clk_n_i,
  input  wire       rst_n_i,
  input  wire       uart_rx_i,
  output wire       uart_tx_o,
  output wire [3:0] led_n_o
);

  // MMCM: 200 MHz LVDS -> 90 MHz
  wire clk_200;
  wire clk_90;
  wire mmcm_locked;

  IBUFDS ibuf_clk (
    .I  (sys_clk_p_i),
    .IB (sys_clk_n_i),
    .O  (clk_200)
  );

  wire mmcm_fb;

  MMCME2_BASE #(
    .CLKIN1_PERIOD (5.0),
    .CLKFBOUT_MULT_F (4.5),
    .CLKOUT0_DIVIDE_F(10.0)
  ) mmcm_inst (
    .CLKIN1  (clk_200),
    .RST     (1'b0),
    .PWRDWN  (1'b0),
    .CLKFBOUT(mmcm_fb),
    .CLKFBIN (mmcm_fb),
    .CLKOUT0 (clk_90),
    .LOCKED  (mmcm_locked)
  );

  wire clk = clk_90;

  // Reset synchronizer (active-high internal reset)
  reg [3:0] rst_pipe;
  wire rst = rst_pipe[3];

  always @(posedge clk or negedge mmcm_locked) begin
    if (!mmcm_locked)
      rst_pipe <= 4'hF;
    else
      rst_pipe <= {rst_pipe[2:0], ~rst_n_i};
  end

  // UART RX
  wire [7:0] rx_data;
  wire       rx_valid;

  uart_rx #(.CLK_FREQ(90_000_000), .BAUD(115_200)) u_rx (
    .clk_i  (clk),
    .rst_i  (rst),
    .rx_i   (uart_rx_i),
    .data_o (rx_data),
    .valid_o(rx_valid)
  );

  // UART TX
  reg  [7:0] tx_data;
  reg        tx_start;
  wire       tx_busy;

  uart_tx #(.CLK_FREQ(90_000_000), .BAUD(115_200)) u_tx (
    .clk_i  (clk),
    .rst_i  (rst),
    .data_i (tx_data),
    .start_i(tx_start),
    .tx_o   (uart_tx_o),
    .busy_o (tx_busy)
  );

  // 8-bit weight store for embedding, layernorm, head_proj
  wire [5:0]  w_sel;
  wire [15:0] w_addr;
  wire [7:0]  w_data;
  wire [15:0] w_scale;

  weight_store u_ws (
    .clk_i       (clk),
    .tensor_sel_i(w_sel),
    .addr_i      (w_addr),
    .data_o      (w_data),
    .scale_o     (w_scale)
  );

  // 64-bit packed weight store for per-layer matvecs and tok_emb (head_proj + embedding)
  wire [3:0]  w8_sel;
  wire [15:0] w8_addr;
  wire [63:0] w8_data;
  wire [15:0] w8_scale;
  wire [11:0] tok_emb_addr;
  wire [63:0] tok_emb_data;
  wire [15:0] tok_emb_scale;

  weight_store_w8 u_ws_w8 (
    .clk_i           (clk),
    .w8_sel_i        (w8_sel),
    .w8_addr_i       (w8_addr),
    .data_o          (w8_data),
    .scale_o         (w8_scale),
    .tok_emb_addr_i  (tok_emb_addr),
    .tok_emb_data_o  (tok_emb_data),
    .tok_emb_scale_o (tok_emb_scale)
  );

  // KV caches with independent address ports
  wire        k_we, v_we;
  wire [15:0] k_wdata, v_wdata;
  wire [15:0] k_rdata, v_rdata;
  wire [1:0]  k_layer, v_layer;
  wire [2:0]  k_head, v_head;
  wire [7:0]  k_pos, v_pos;
  wire [3:0]  k_dim, v_dim;

  kv_cache #(.DATA_W(16)) u_kcache (
    .clk_i  (clk),
    .layer_i(k_layer),
    .head_i (k_head),
    .pos_i  (k_pos),
    .dim_i  (k_dim),
    .we_i   (k_we),
    .wdata_i(k_wdata),
    .rdata_o(k_rdata)
  );

  kv_cache #(.DATA_W(16)) u_vcache (
    .clk_i  (clk),
    .layer_i(v_layer),
    .head_i (v_head),
    .pos_i  (v_pos),
    .dim_i  (v_dim),
    .we_i   (v_we),
    .wdata_i(v_wdata),
    .rdata_o(v_rdata)
  );

  // Transformer top
  reg        tf_start;
  reg        tf_generate;
  reg  [7:0] tf_token_in;
  wire [7:0] tf_token_out;
  wire       tf_token_valid;
  wire       tf_busy;
  wire       tf_done;

  transformer_top u_tf (
    .clk_i       (clk),
    .rst_i       (rst),
    .token_i     (tf_token_in),
    .start_i     (tf_start),
    .generate_i  (tf_generate),
    .w_sel_o     (w_sel),
    .w_addr_o    (w_addr),
    .w_data_i    (w_data),
    .w_scale_i   (w_scale),
    .w8_sel_o        (w8_sel),
    .w8_addr_o       (w8_addr),
    .w8_data_i       (w8_data),
    .w8_scale_i      (w8_scale),
    .tok_emb_addr_o  (tok_emb_addr),
    .tok_emb_data_i  (tok_emb_data),
    .tok_emb_scale_i (tok_emb_scale),
    .k_we_o      (k_we),
    .k_wdata_o   (k_wdata),
    .k_layer_o   (k_layer),
    .k_head_o    (k_head),
    .k_pos_o     (k_pos),
    .k_dim_o     (k_dim),
    .k_rdata_i   (k_rdata),
    .v_we_o      (v_we),
    .v_wdata_o   (v_wdata),
    .v_layer_o   (v_layer),
    .v_head_o    (v_head),
    .v_pos_o     (v_pos),
    .v_dim_o     (v_dim),
    .v_rdata_i   (v_rdata),
    .token_o     (tf_token_out),
    .token_valid_o(tf_token_valid),
    .busy_o      (tf_busy),
    .done_o      (tf_done)
  );

  // Control FSM
  // S_WAIT_CMD: wait for UART byte
  //   0xFF -> start generation from accumulated prompt
  //   other -> feed as prompt token
  // S_PROMPT: wait for prompt pass to finish, then back to S_WAIT_CMD
  // S_GENERATE: autoregressive loop, send each token over UART
  // S_TX_WAIT: wait for UART TX to finish, then continue generating

  localparam [2:0] S_WAIT_CMD = 3'd0,
                   S_PROMPT   = 3'd1,
                   S_GENERATE = 3'd2,
                   S_TX_WAIT  = 3'd3,
                   S_TX_DONE  = 3'd4;

  reg [2:0] ctl_state;
  reg       gen_started;

  localparam CMD_GENERATE = 8'hFF;

  always @(posedge clk) begin
    if (rst) begin
      ctl_state   <= S_WAIT_CMD;
      tf_start    <= 1'b0;
      tf_generate <= 1'b0;
      tf_token_in <= 8'd0;
      tx_start    <= 1'b0;
      tx_data     <= 8'd0;
      gen_started <= 1'b0;
    end else begin
      tf_start <= 1'b0;
      tx_start <= 1'b0;

      case (ctl_state)

        S_WAIT_CMD: begin
          if (rx_valid) begin
            if (rx_data == CMD_GENERATE) begin
              // Start generation with last prompt token
              tf_token_in <= tf_token_in;
              tf_generate <= 1'b1;
              tf_start    <= 1'b1;
              gen_started <= 1'b0;
              ctl_state   <= S_GENERATE;
            end else begin
              // Prompt token: run forward pass to fill KV cache
              tf_token_in <= rx_data;
              tf_generate <= 1'b0;
              tf_start    <= 1'b1;
              ctl_state   <= S_PROMPT;
            end
          end
        end

        S_PROMPT: begin
          if (tf_done) begin
            ctl_state <= S_WAIT_CMD;
          end
        end

        S_GENERATE: begin
          if (tf_token_valid) begin
            // Got a generated token, send it over UART
            tx_data   <= tf_token_out;
            tx_start  <= 1'b1;
            ctl_state <= S_TX_WAIT;
          end
          if (tf_done) begin
            // Generation finished (pos=255)
            ctl_state <= S_WAIT_CMD;
          end
        end

        S_TX_WAIT: begin
          if (!tx_busy) begin
            ctl_state <= S_TX_DONE;
          end
        end

        // Return to generate state for next token
        S_TX_DONE: begin
          if (tf_done) begin
            ctl_state <= S_WAIT_CMD;
          end else if (tf_token_valid) begin
            tx_data   <= tf_token_out;
            tx_start  <= 1'b1;
            ctl_state <= S_TX_WAIT;
          end else begin
            ctl_state <= S_GENERATE;
          end
        end

        default: ctl_state <= S_WAIT_CMD;

      endcase
    end
  end

  // LEDs (active-low: drive 0 to light)
  reg [25:0] heartbeat_cnt;
  reg        heartbeat_r;
  reg [19:0] rx_blink, tx_blink;

  always @(posedge clk) begin
    if (rst) begin
      heartbeat_cnt <= 0;
      heartbeat_r   <= 0;
      rx_blink      <= 0;
      tx_blink      <= 0;
    end else begin
      // Heartbeat ~0.33s toggle at 90 MHz
      if (heartbeat_cnt == 24'd30_000_000) begin
        heartbeat_cnt <= 0;
        heartbeat_r   <= ~heartbeat_r;
      end else begin
        heartbeat_cnt <= heartbeat_cnt + 1;
      end

      // RX blink: set on valid, decrement to zero
      if (rx_valid)
        rx_blink <= 20'hFFFFF;
      else if (rx_blink != 0)
        rx_blink <= rx_blink - 1;

      // TX blink: set on start, decrement to zero
      if (tx_start)
        tx_blink <= 20'hFFFFF;
      else if (tx_blink != 0)
        tx_blink <= tx_blink - 1;
    end
  end

  assign led_n_o[0] = ~tf_busy;
  assign led_n_o[1] = ~(rx_blink != 0);
  assign led_n_o[2] = ~(tx_blink != 0);
  assign led_n_o[3] = ~heartbeat_r;

endmodule