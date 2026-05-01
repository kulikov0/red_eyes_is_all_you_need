open_checkpoint /home/user/red_eyes_is_all_you_need/build/post_route.dcp

puts "=========================================================="
puts "TOP 30 WORST SETUP PATHS - summary (1 per endpoint)"
puts "=========================================================="
report_timing -delay_type max -max_paths 30 -nworst 1 -path_type short

puts "=========================================================="
puts "TOP 8 WORST SETUP PATHS - full details"
puts "=========================================================="
report_timing -delay_type max -max_paths 8 -nworst 1 -path_type full

puts "=========================================================="
puts "TOP 25 WORST PATHS - logic levels histogram"
puts "=========================================================="
set worst [get_timing_paths -max_paths 200 -nworst 1]
set ll_hist [dict create]
foreach p $worst {
  set ll [get_property LOGIC_LEVELS $p]
  dict incr ll_hist $ll
}
puts "Logic levels distribution (top 200 paths):"
foreach k [lsort -integer [dict keys $ll_hist]] {
  puts [format "  %3d levels : %d paths" $k [dict get $ll_hist $k]]
}

puts "=========================================================="
puts "WORST PATH PER TOP-LEVEL HIERARCHY"
puts "=========================================================="
set hier_paths [list]
foreach p $worst {
  set ep [get_property ENDPOINT_PIN $p]
  set ep_cell [get_cells -of_objects $ep]
  set hier [lindex [split $ep_cell /] 0]
  if {[lsearch $hier_paths $hier] < 0} {
    lappend hier_paths $hier
    set s [get_property SLACK $p]
    set ll [get_property LOGIC_LEVELS $p]
    set sp [get_property STARTPOINT_PIN $p]
    puts [format "  slack=%6.3f  logic=%2d  start=%s  end=%s" $s $ll $sp $ep]
  }
}

puts "=========================================================="
puts "WORST 50 PATHS - start/end summary"
puts "=========================================================="
set top50 [lrange $worst 0 49]
foreach p $top50 {
  set s  [get_property SLACK $p]
  set ll [get_property LOGIC_LEVELS $p]
  set sp [get_property STARTPOINT_PIN $p]
  set ep [get_property ENDPOINT_PIN $p]
  puts [format "slack=%6.3f LL=%2d  %s -> %s" $s $ll $sp $ep]
}

puts "=========================================================="
puts "HIGH FANOUT NETS"
puts "=========================================================="
report_high_fanout_nets -max_nets 25 -fanout_greater_than 200

puts "=========================================================="
puts "UTILIZATION (top-level + 2 levels deep)"
puts "=========================================================="
report_utilization
report_utilization -hierarchical -hierarchical_depth 3
