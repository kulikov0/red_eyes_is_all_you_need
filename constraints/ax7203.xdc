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

create_generated_clock -name clk_80 -source [get_ports sys_clk_p_i] -multiply_by 2 -divide_by 5 [get_pins mmcm_inst/CLKOUT0]