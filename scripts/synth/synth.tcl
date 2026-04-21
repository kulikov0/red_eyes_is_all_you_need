set proj_dir /home/user/red_eyes_is_all_you_need

create_project -force red_eyes /tmp/red_eyes -part xc7a200tfbg484-2

add_files [glob $proj_dir/rtl/*.v]
add_files $proj_dir/rtl/weight_scales.vh
add_files $proj_dir/rtl/gelu_pwl_coeffs.vh
set_property include_dirs $proj_dir/rtl [current_fileset]

add_files -fileset constrs_1 $proj_dir/constraints/ax7203.xdc
set_property top top [current_fileset]

synth_design -top top -part xc7a200tfbg484-2
report_utilization
report_timing_summary

write_checkpoint -force $proj_dir/build/post_synth.dcp

opt_design
place_design
route_design

report_utilization
report_timing_summary

write_checkpoint -force $proj_dir/build/post_route.dcp
write_bitstream -force $proj_dir/build/top.bit