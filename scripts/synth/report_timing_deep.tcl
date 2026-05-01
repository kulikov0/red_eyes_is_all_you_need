open_checkpoint /home/user/red_eyes_is_all_you_need/build/post_route.dcp

puts "=========================================================="
puts "TIMING SUMMARY"
puts "=========================================================="
report_timing_summary -no_header -no_detailed_paths

puts "=========================================================="
puts "CLOCKS"
puts "=========================================================="
report_clocks

puts "=========================================================="
puts "TOP 30 SETUP PATHS - max 1 per endpoint, with logic levels"
puts "=========================================================="
report_timing -delay_type max -max_paths 30 -unique_pids -path_type summary

puts "=========================================================="
puts "TOP 12 SETUP PATHS - full detail"
puts "=========================================================="
report_timing -delay_type max -max_paths 12 -unique_pids -nworst 1 -path_type full

puts "=========================================================="
puts "WORST PATHS BY HIERARCHY (top-level cells)"
puts "=========================================================="
foreach inst {u_tf/u_tl u_tf/u_attn u_tf/u_emb u_tf/u_ln_f u_tf/u_head_proj u_tf/u_samp u_ws u_ws_w8 u_kcache u_vcache} {
  if {[llength [get_cells -quiet $inst]] > 0} {
    puts "----- $inst -----"
    report_timing -from [get_cells $inst/*] -to [get_cells $inst/*] -max_paths 3 -path_type summary
  }
}

puts "=========================================================="
puts "WORST PATHS BY DEEP HIERARCHY (matvec, fp16 prims, attention internals)"
puts "=========================================================="
foreach pat {*/u_qkv */u_proj */u_ff_up */u_ff_down */u_head_proj */u_sm */u_ln */u_ln_f */u_attn/u_sc_mul* */u_attn/u_av_mul* */u_attn/u_sc_red* */u_attn/u_av_*_add */u_ln/u_norm_* */u_ln/u_var_* */u_ln/u_mean_red */u_ln/u_rsqrt} {
  set cells [get_cells -quiet -hierarchical -filter "NAME =~ $pat"]
  if {[llength $cells] > 0} {
    foreach c $cells {
      puts "----- $c -----"
      report_timing -through [get_cells $c/*] -max_paths 1 -path_type summary
    }
  }
}

puts "=========================================================="
puts "DSP / BRAM / FANOUT HOTSPOTS"
puts "=========================================================="
report_high_fanout_nets -max_nets 20 -fanout_greater_than 200

puts "=========================================================="
puts "UTILIZATION"
puts "=========================================================="
report_utilization
report_utilization -hierarchical -hierarchical_depth 3
