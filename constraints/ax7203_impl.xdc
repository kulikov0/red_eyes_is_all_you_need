# Implementation-only timing exceptions
# RAMB36 cells are inferred during synth_design and only exist in the post-synth
# netlist, so this file is loaded with USED_IN_SYNTHESIS false from synth.tcl

# 3-cycle KV write FSM in attention.v holds wdata stable for 2 cycles before
# WE pulses, so the long route to scattered KV BRAMs has 2 clock periods
set kv_src_regs   [get_cells -hier -filter {NAME =~ *u_tf/k_wdata_o_reg* || NAME =~ *u_tf/v_wdata_o_reg* || NAME =~ *u_tf/k_pos_o_reg* || NAME =~ *u_tf/v_pos_o_reg*}]
set kv_ram_cells  [get_cells -hier -filter {(NAME =~ *u_kcache/banks* || NAME =~ *u_vcache/banks*) && REF_NAME =~ RAMB36*}]
set kv_write_pins [get_pins -of_objects $kv_ram_cells -filter {NAME =~ */DIADI* || NAME =~ */DIBDI* || NAME =~ */WEA* || NAME =~ */WEBWE*}]
set_multicycle_path 2 -setup -from $kv_src_regs -to $kv_write_pins
set_multicycle_path 1 -hold  -from $kv_src_regs -to $kv_write_pins