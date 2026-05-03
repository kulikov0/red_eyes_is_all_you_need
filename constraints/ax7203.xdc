set_property -dict {PACKAGE_PIN R4 IOSTANDARD DIFF_SSTL15} [get_ports sys_clk_p_i]
set_property -dict {PACKAGE_PIN T4 IOSTANDARD DIFF_SSTL15} [get_ports sys_clk_n_i]
create_clock -period 5.000 -name sys_clk [get_ports sys_clk_p_i]

set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports uart_tx_o]
set_property -dict {PACKAGE_PIN P20 IOSTANDARD LVCMOS33} [get_ports uart_rx_i]

set_property -dict {PACKAGE_PIN J21 IOSTANDARD LVCMOS33} [get_ports rst_n_i]

set_property -dict {PACKAGE_PIN B13 IOSTANDARD LVCMOS33} [get_ports {led_n_o[0]}]
set_property -dict {PACKAGE_PIN C13 IOSTANDARD LVCMOS33} [get_ports {led_n_o[1]}]
set_property -dict {PACKAGE_PIN D14 IOSTANDARD LVCMOS33} [get_ports {led_n_o[2]}]
set_property -dict {PACKAGE_PIN D15 IOSTANDARD LVCMOS33} [get_ports {led_n_o[3]}]

create_generated_clock -name clk_90 -source [get_ports sys_clk_p_i] -multiply_by 9 -divide_by 20 [get_pins mmcm_inst/CLKOUT0]

# Pblock placement constraints. Each matvec is locked to its own slice/DSP/BRAM
# band so 16-lane internal routing stays short and fp16_add sticky paths,
# in_snap to DSP B, and weight_data fanout do not span the device.

create_pblock pb_qkv
add_cells_to_pblock pb_qkv [get_cells {u_tf/u_tl/u_attn/u_qkv u_ws_qkv}]
resize_pblock pb_qkv -add {SLICE_X0Y0:SLICE_X41Y149 \
                           DSP48_X0Y0:DSP48_X2Y59 \
                           RAMB36_X0Y0:RAMB36_X2Y29 \
                           RAMB18_X0Y0:RAMB18_X2Y59}

create_pblock pb_proj
add_cells_to_pblock pb_proj [get_cells {u_tf/u_tl/u_attn/u_proj u_ws_proj}]
resize_pblock pb_proj -add {SLICE_X42Y0:SLICE_X69Y149 \
                            DSP48_X3Y20:DSP48_X3Y79 \
                            RAMB36_X3Y10:RAMB36_X3Y39 \
                            RAMB18_X3Y20:RAMB18_X3Y79}

create_pblock pb_ff_up
add_cells_to_pblock pb_ff_up [get_cells {u_tf/u_tl/u_ff_up u_ws_ff_up}]
resize_pblock pb_ff_up -add {SLICE_X70Y0:SLICE_X109Y149 \
                             DSP48_X4Y20:DSP48_X5Y79 \
                             RAMB36_X4Y10:RAMB36_X5Y39 \
                             RAMB18_X4Y20:RAMB18_X5Y79}

create_pblock pb_ff_down
add_cells_to_pblock pb_ff_down [get_cells {u_tf/u_tl/u_ff_down u_ws_ff_down}]
resize_pblock pb_ff_down -add {SLICE_X110Y0:SLICE_X145Y149 \
                               DSP48_X6Y20:DSP48_X7Y99 \
                               RAMB36_X6Y10:RAMB36_X7Y45 \
                               RAMB18_X6Y20:RAMB18_X7Y99}

create_pblock pb_head_proj
add_cells_to_pblock pb_head_proj [get_cells u_tf/u_head_proj]
resize_pblock pb_head_proj -add {SLICE_X146Y0:SLICE_X163Y249 \
                                 DSP48_X8Y0:DSP48_X8Y99 \
                                 RAMB36_X8Y0:RAMB36_X8Y47 \
                                 RAMB18_X8Y0:RAMB18_X8Y99}

create_pblock pb_ln_f
add_cells_to_pblock pb_ln_f [get_cells u_tf/u_ln_f]
resize_pblock pb_ln_f -add {SLICE_X0Y150:SLICE_X100Y210 \
                            DSP48_X0Y75:DSP48_X5Y99 \
                            RAMB36_X0Y36:RAMB36_X4Y47 \
                            RAMB18_X0Y75:RAMB18_X4Y99}

create_pblock pb_softmax
add_cells_to_pblock pb_softmax [get_cells {u_tf/u_tl/u_attn/u_sm_a u_tf/u_tl/u_attn/u_sm_b}]
resize_pblock pb_softmax -add {SLICE_X100Y170:SLICE_X145Y210 \
                               RAMB36_X6Y36:RAMB36_X7Y43 \
                               RAMB18_X6Y75:RAMB18_X7Y89}

create_pblock pb_sampler
add_cells_to_pblock pb_sampler [get_cells u_tf/u_samp]
resize_pblock pb_sampler -add {SLICE_X100Y150:SLICE_X145Y169 \
                               SLICE_X100Y211:SLICE_X145Y249}
